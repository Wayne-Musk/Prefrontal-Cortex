"""wm.proposal -- 候选动作提议器（skills 在新范式中的位置）。

**这是整套设计里最关键的角色重分配。**

在 skill + role-playing 范式里，技能是决策者：匹配到症状 → 执行套路。
在本架构里，技能降级为两件小事：

    1. candidate generator —— 向 planner 提议「值得考虑的动作」（缩小搜索分支）
    2. structured prior     —— 给这些候选一个先验分（加速搜索收敛）

它不再决定做什么。决定做什么的是「动力学 + 价值 + 反事实搜索」。
因此：技能写错、漏写、或者遇到技能没覆盖的新情况，系统退化到「动力学组合已知算子」，
而不是直接崩掉 —— 这就是泛化的机制。

两路提议：
    HeuristicProposal   —— 纯符号、零依赖、离线可跑（信息增益近似贪心）
    LLMProposal         —— 用 LLM 做语义层面的候选建议（需要 backend；无则自动降级）
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .inference import marginal
from .types import Action, Belief, Json


@dataclass
class ProposalConfig:
    max_branching: int = 12        # 每个搜索节点保留的候选数
    repair_mass_threshold: float = 0.35   # 某假设后验超过此值才值得赌一次修复
    top_services: int = 3
    probe_cost: float = 1.0


class Proposal:
    """提议器接口：belief → [(action, prior_score)]。"""

    def __call__(self, belief: Belief, model, actions: List[Action]) -> List[Tuple[Action, float]]:
        raise NotImplementedError


# ---------------------------------------------------------------- 启发式提议


class HeuristicProposal(Proposal):
    """无 LLM 的候选提议。

    策略（纯粹是「信息论启发式」，不含任何领域硬知识以外的东西）：
      * 为每个服务 × 每种故障模式计算后验
      * 探针的价值 = 该探针能把剩下的假设切成多均匀（越接近 50/50 越有价值）
      * 修复动作仅当某个 (service, mode) 假设已经足够可信时才进入候选
      * 信念熵很高且未做过全局巡检 → 强制加入全局观测动作
    """

    def __init__(self, cfg: Optional[ProposalConfig] = None,
                 probes: Tuple[str, ...] = (),
                 repairs: Tuple[str, ...] = (),
                 fix_map: Optional[Dict[str, str]] = None,
                 detect_map: Optional[Dict[str, str]] = None):
        self.cfg = cfg or ProposalConfig()
        self.probes = tuple(probes)
        self.repairs = tuple(repairs)
        self.fix_map = dict(fix_map or {})
        self.detect_map = dict(detect_map or {})

    # -- 分布的边缘化 -----------------------------------------------
    def _root_posterior(self, belief: Belief) -> Dict[str, float]:
        return marginal(belief, lambda s: s.get("root"))

    def _mode_posterior(self, belief: Belief) -> Dict[str, float]:
        return marginal(belief, lambda s: s.get("mode"))

    def _pair_posterior(self, belief: Belief) -> Dict[Tuple[str, str], float]:
        return marginal(belief, lambda s: (s.get("root"), s.get("mode")))

    # -- 主入口 -----------------------------------------------------
    def __call__(self, belief: Belief, model, actions: List[Action]) -> List[Tuple[Action, float]]:
        if not actions:
            return []
        scored: Dict[str, Tuple[Action, float]] = {}
        root_post = self._root_posterior(belief)
        mode_post = self._mode_posterior(belief)
        pair_post = self._pair_posterior(belief)

        global_mass = mode_post.get(None, 0.0) + mode_post.get("none", 0.0)
        entropy = belief.entropy()

        # 1) 修复类：只在假设足够尖锐时才提议（避免盲目动手）
        for act in actions:
            if act.name not in self.repairs or not act.params:
                continue
            svc = act.params[0]
            best_p = max((p for (r, m), p in pair_post.items() if r == svc), default=0.0)
            correct = self.fix_map.get(_mode_of(pair_post, svc)) == act.name
            base = 0.0
            if best_p >= self.cfg.repair_mass_threshold:
                base = best_p * (1.0 if correct else 0.35)
            elif entropy > 1.5:
                base = 0.05          # 完全没头绪时也留一条路，但分数极低
            if base > 0:
                scored[act.label] = (act, base - 0.02 * act.cost)

        # 2) 探针类：按「对分能力」排序，这是廉价的信息论启发式
        top_svcs = [s for s, _ in sorted(root_post.items(), key=lambda kv: -kv[1])
                    [:self.cfg.top_services] if s]
        for act in actions:
            if act.name != "probe" or len(act.params) < 2:
                continue
            svc, kind = act.params[0], act.params[1]
            if svc not in top_svcs:
                continue
            target_mode = self.detect_map.get(kind)
            p_target = 0.0
            if target_mode is not None:
                p_target = sum(p for (r, m), p in pair_post.items()
                               if r == svc and m == target_mode)
                p_target += mode_post.get(target_mode, 0.0) * 0.25  # 弱先验：模式本身可能
            balance = 1.0 - abs(2 * min(p_target, 1.0) - 1.0)       # 0.5 → 1.0
            score = 0.10 + 0.55 * balance + 0.15 * root_post.get(svc, 0.0)
            scored[act.label] = (act, score - 0.03 * act.cost)

        # 3) 全局巡检：熵高或一切正常时才需要
        for act in actions:
            if act.name == "monitor":
                s = 0.35 if entropy > 1.2 else (0.30 if global_mass > 0.4 else 0.08)
                scored[act.label] = (act, s)

        # 4) 兜底：没被上述规则命中的动作给一个很低的基础分，保证不至于被永久排除
        for act in actions:
            scored.setdefault(act.label, (act, 0.02))

        ranked = sorted(scored.values(), key=lambda kv: -kv[1])
        return ranked[: self.cfg.max_branching]


def _mode_of(pair_post: Dict[Tuple[str, str], float], svc: str):
    best, bp = None, -1.0
    for (r, m), p in pair_post.items():
        if r == svc and p > bp:
            best, bp = m, p
    return best


# ---------------------------------------------------------------- LLM 提议


class LLMProposal(Proposal):
    """用 LLM 做语义级候选提议。没有 backend 时静默降级为启发式提议。

    注意 LLM 在这里的职责边界：**只提议，不裁决**。
    它提出的动作若没有对应的动力学算子，planner 会给出 unknown_op_penalty；
    若它的语义判断与动力学冲突，以 rollout 出来的期望收益为准。
    """

    def __init__(self, backend=None, fallback: Optional[Proposal] = None,
                 cfg: Optional[ProposalConfig] = None):
        self.backend = backend
        self.fallback = fallback
        self.cfg = cfg or ProposalConfig()
        self.calls = 0

    def __call__(self, belief: Belief, model, actions: List[Action]) -> List[Tuple[Action, float]]:
        base = (self.fallback or HeuristicProposal())(belief, model, actions)
        if self.backend is None:
            return base
        try:
            self.calls += 1
            labels = [a.label for a in actions]
            prompt = _build_proposal_prompt(belief, labels)
            text = self.backend.complete(prompt)
            boosted = dict(base)
            for i, lab in enumerate(labels):
                if lab in text:
                    act = _find_action(actions, lab)
                    if act is not None:
                        prev = boosted.get(lab, (act, 0.02))[1]
                        boosted[lab] = (act, prev + 0.25)
            ranked = sorted(boosted.values(), key=lambda kv: -kv[1])
            return ranked[: self.cfg.max_branching]
        except Exception:
            return base


def _find_action(actions: List[Action], label: str) -> Optional[Action]:
    for a in actions:
        if a.label == label:
            return a
    return None


def _build_proposal_prompt(belief: Belief, labels: List[str]) -> str:
    top = [f"{p:.2f}: {s}" for _, p, s in belief.top(4)]
    return (
        "你是故障排查助手。当前信念熵=%.2f，最可能的几个世界状态：\n%s\n\n"
        "候选动作：\n%s\n\n"
        "请列出你最想先执行的 3 个动作（原样输出动作名即可），按优先级排序。\n"
        % (belief.entropy(), "\n".join(top), "\n".join(labels))
    )
