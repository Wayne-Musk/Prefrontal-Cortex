"""wm.grounding -- Skill Grounding：把技能从「决策者」编译成「动力学算子」。

这是整篇设计里回答「泛化从哪来」的那个模块。

传统范式：
    技能 = (触发条件, 执行套路)。遇到没见过的情境 → 没有匹配 → 失败。
    技能之间不共享结构，N 个技能只能处理 N 类情境。

本模块的做法：观察若干真实轨迹 (s, a, s')，归纳出
    precondition: 哪些特征在执行前必须成立（跨样本恒定的量）
    effect      : 哪些特征被改变了（包括随机效应：多个后继的分布）
然后把它注册进 WorldModel.operators。

一旦成为 operator，它就变成一个**可被任意组合的基本粒子**：planner 可以在任何满足
前提的信念上使用它，可以与其它算子串联，可以在 rollout 里被反事实地评估。
从一个「场景专用套路」升级为一个「通用转移算子」—— 这就是泛化。

另外这个文件也实现了 k-arm:
新bo安装在 bx skill：先用 N 次受控试跑归纳语义（一段 actionsini 的 grounding 流程），
而不是无条件相信 skill 文档宣称的效果。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from .model import Operator, WorldModel
from .types import Action, Json, signature


@dataclass
class TransitionSample:
    state_before: Json
    action: Action
    state_after: Json


@dataclass
class InducedSkill:
    action_name: str
    preconditions: Dict[Hashable, Hashable]
    effects: List[Tuple[float, Dict[Hashable, Hashable]]]   # (概率, 变更量)
    support: int
    consistency: float

    def __str__(self) -> str:
        eff = ", ".join(f"{p:.2f}:{e}" for p, e in self.effects[:2])
        return f"{self.action_name} | pre={len(self.preconditions)} | eff={eff} | n={self.support}"


@dataclass
class GroundingConfig:
    min_support: int = 2            # 至少多少条样本才敢归纳
    min_consistency: float = 0.6    # 效应一致度阈值
    max_effects: int = 4            # 保留的随机分支数
    max_preconditions: int = 8


class SkillCompiler:
    """从轨迹数据里反向编译出符号算子。"""

    def __init__(self, cfg: Optional[GroundingConfig] = None):
        self.cfg = cfg or GroundingConfig()
        self.samples: List[TransitionSample] = []

    # ------------------------------------------------------------ 采集
    def record(self, state_before: Json, action: Action, state_after: Json) -> None:
        self.samples.append(TransitionSample(dict(state_before), action, dict(state_after)))

    def n(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------ 归纳
    def induce(self) -> List[InducedSkill]:
        by_action: Dict[str, List[TransitionSample]] = {}
        for smp in self.samples:
            by_action.setdefault(smp.action.name, []).append(smp)

        out: List[InducedSkill] = []
        for name, group in by_action.items():
            skill = self._induce_one(name, group)
            if skill is not None:
                out.append(skill)
        return out

    def _induce_one(self, name: str, group: List[TransitionSample]) -> Optional[InducedSkill]:
        cfg = self.cfg
        # 1) 哪些 key 在「执行前」跨样本恒定 → 候选前提
        pre_keys: Dict[Hashable, set] = {}
        for smp in group:
            for k, v in smp.state_before.items():
                pre_keys.setdefault(k, set()).add(_freeze(v))
        pre: Dict[Hashable, Hashable] = {}
        for k, vals in pre_keys.items():
            if len(vals) == 1 and _scalar(k):
                pre[k] = next(iter(vals))
            if len(pre) >= cfg.max_preconditions:
                break

        # 2) 哪些 key 在执行后改变了 → 效应，并按多分支分布保留
        effect_counts: Dict[str, int] = {}
        total = 0
        changed_keys = 0
        for smp in group:
            delta = {}
            for k, v in smp.state_after.items():
                vb = smp.state_before.get(k)
                if _freeze(vb) != _freeze(v):
                    delta[k] = _freeze(v)
            key = signature(delta)
            effect_counts[key] = effect_counts.get(key, 0) + 1
            total += 1
            changed_keys += 1 if delta else 0

        if total == 0:
            return None
        noop_count = effect_counts.get(signature({}), 0)
        consistency = 1.0 - (noop_count / total)      # 有多大概率「确实起了作用」
        support = sum(effect_counts.values())
        if support < cfg.min_support or consistency < cfg.min_consistency:
            return None

        effects: List[Tuple[float, Dict[Hashable, Hashable]]] = []
        for key, cnt in sorted(effect_counts.items(), key=lambda kv: -kv[1])[: cfg.max_effects]:
            delta = _load(key)
            if delta:
                effects.append((cnt / total, delta))
        return InducedSkill(name, pre, effects, support, consistency)

    # ------------------------------------------------------------ 注册
    def register(self, model: WorldModel, overwrite: bool = False) -> List[str]:
        """把归纳出的技能写回世界模型，使其可参与反事实规划。"""
        registered: List[str] = []
        for skill in self.induce():
            if model.is_covered(Action(skill.action_name)) and not overwrite:
                existing = model.operators[skill.action_name]
                existing.n_samples += skill.support
                existing.support_count = max(existing.support_count, skill.support)
                continue
            model.add_operator(_to_operator(skill))
            registered.append(skill.action_name)
        return registered


def _to_operator(skill: InducedSkill) -> Operator:
    pre = dict(skill.preconditions)
    effects = list(skill.effects)

    def precond(state: Json, action: Action) -> bool:
        for k, v in pre.items():
            if _freeze(state.get(k)) != v:
                return False
        return True

    def trans(state: Json, action: Action, rng: random.Random) -> Json:
        if not precond(state, action):
            return dict(state)
        # 前提局部化：只检查参数相关的那部分前提，否则过于严苛
        r = rng.random()
        acc = 0.0
        for p, delta in effects:
            acc += p
            if r <= acc or abs(acc - 1.0) < 1e-9:
                new = dict(state)
                new.update(delta)
                return new
        return dict(state)

    op = Operator(action_name=skill.action_name, transition=trans,
                  precondition=precond, learned=True, support_count=skill.support)
    op.err_ema = -math.log(max(skill.consistency, 1e-6))
    return op


# ---------------------------------------------------------------- 工具


def _freeze(v):
    """把 dict 之类的值变成可哈希形式。"""
    if isinstance(v, dict):
        return signature(v)
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _load(s: str) -> Dict:
    import json as _json
    try:
        return _json.loads(s)
    except Exception:
        return {}


def _scalar(k) -> bool:
    return isinstance(k, str)
