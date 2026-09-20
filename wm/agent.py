"""wm.agent -- 闭环：感知 → 想象 → 决策 → 行动 → 学习。

主循环与传统 Agent 的对照：

    传统 (skill + roleplay)
        context = history
        loop: obs → LLM(context, role, skills) → action        # 决策在 token 里一次性涌现
                                                               # 没有 '\''if I do X''\'' 的评估
    World-Model Agent
        loop: obs → belief update (贝叶斯, 显式后验)
                  → plan (在模型里 rollout N×H 次, 不动真实世界)
                  → act  (只执行一步)
                  → 对比预测与现实 → surprise → 修正信念 / 修正模型 / 调整保守度

「 Thanks 只执行一步」很重要：每一步都重新规划。任何一步的偏差都会被下一次观测纠正，
不会因为一次 rollout 就一路错到底。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .grounding import SkillCompiler
from .inference import belief_update
from .llm import LLMBackend
from .model import WorldModel
from .monitor import ModelMonitor
from .planner import MCTSPlanner, PlannerConfig
from .proposal import HeuristicProposal, LLMProposal, Proposal
from .types import Action, Belief, DecisionTrace, Observation
from .value import ValueModel


@dataclass
class AgentConfig:
    n_sims: int = 48
    rollout_depth: int = 4
    prune_min: float = 5e-4
    max_hyp: int = 32
    K_propagate: int = 1
    ig_samples: int = 1
    learn_every: int = 1              # 每多少步做一次模型/算子层面的自我修正
    ground_skills: bool = True
    verbose: bool = False


class WMAgent:
    """World-Model Agent。"""

    def __init__(self,
                 model: WorldModel,
                 value: ValueModel,
                 proposal: Optional[Proposal] = None,
                 monitor: Optional[ModelMonitor] = None,
                 compiler: Optional[SkillCompiler] = None,
                 cfg: Optional[AgentConfig] = None,
                 rng: Optional[random.Random] = None,
                 llm: Optional[LLMBackend] = None):
        self.model = model
        self.value = value
        self.monitor = monitor
        self.compiler = compiler or SkillCompiler()
        self.cfg = cfg or AgentConfig()
        self.rng = rng or random.Random(0)
        if proposal is None:
            proposal = LLMProposal(llm, HeuristicProposal()) if llm else HeuristicProposal()
        self.proposal = proposal
        pcfg = PlannerConfig(
            n_sims=self.cfg.n_sims,
            rollout_depth=self.cfg.rollout_depth,
            K_propagate=self.cfg.K_propagate,
            ig_samples=self.cfg.ig_samples,
        )
        self.planner = MCTSPlanner(model, value, proposal, monitor, pcfg, self.rng)
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ 感知
    def initial_belief(self, obs: Optional[Observation] = None) -> Belief:
        b = self.model.initial_belief()
        if obs is not None:
            b.set_facts(obs.facts)
        return b

    def perceive(self, belief: Belief, action: Optional[Action],
                 obs: Observation) -> Tuple[Belief, float]:
        """用真实观测做一次 filter。返回 (新信念, surprise)。"""
        if action is None:
            from .inference import bayes_correct
            from .inference import predict_step
            b2, surprise = bayes_correct(belief, Action("noop"), obs, self.model)
            b2.renorm()
        else:
            b2, surprise = belief_update(
                belief, action, obs, self.model, self.rng,
                K=self.cfg.K_propagate,
                prune_min=self.cfg.prune_min,
                max_hyp=self.cfg.max_hyp,
            )
        b2.t = belief.t + 1
        if self.monitor is not None:
            self.monitor.observe(b2.t, action or Action("noop"), surprise, belief, b2)
        return b2, surprise

    # ------------------------------------------------------------ 决策
    def decide(self, belief: Belief) -> Tuple[Action, DecisionTrace]:
        adv = self.monitor.advice() if self.monitor else {}
        base_risk = self.value.cfg.lambda_risk
        base_info = self.value.cfg.lambda_info
        self.value.cfg.lambda_risk = base_risk * adv.get("risk_scale", 1.0)
        self.value.cfg.lambda_info = base_info * adv.get("info_bonus", 1.0)
        try:
            act, trace = self.planner.plan(belief)
        finally:
            self.value.cfg.lambda_risk = base_risk
            self.value.cfg.lambda_info = base_info
        cons = adv.get("risk_scale", 1.0) > 1.0
        trace.note = ("保守模式" if cons else "常规模式") + f" H={belief.entropy():.2f}"
        return act, trace

    # ------------------------------------------------------------ 学习
    def learn(self, belief: Belief, action: Action, belief_after: Belief,
              surprise: float) -> Dict[str, Any]:
        """一次交互后的自我修正：惊讶度 → 算子可信度 → 必要的技能再 grounding。"""
        info: Dict[str, Any] = {}
        if self.cfg.ground_skills:
            self.compiler.record(belief.map_state(), action, belief_after.map_state())
        if self.monitor is not None:
            info["n_updated_ops"] = self.monitor.apply_to_model(self.model)
            if self.monitor.in_conservative_mode(action.name):
                learned = self.compiler.register(self.model, overwrite=False)
                info["induced_skills"] = learned
        return info

    # ------------------------------------------------------------ 单步
    def step(self, belief: Belief, action: Action,
             obs: Observation) -> Tuple[Belief, float, Dict[str, Any]]:
        b2, surprise = self.perceive(belief, action, obs)
        info = self.learn(belief, action, b2, surprise)
        return b2, surprise, info

    # ------------------------------------------------------------ 诊断
    def explain(self, belief: Belief) -> str:
        lines = [f"t={belief.t}  H={belief.entropy():.3f}  |hyp|={len(belief.weights)}"]
        for sig, p, s in belief.top(3):
            lines.append(f"  p={p:.3f}  {_short(s)}")
        if self.monitor is not None:
            lines.append(self.monitor.report())
            bs = self.model.blind_spots(3)
            if bs:
                lines.append("  动力学未覆盖: " + ", ".join(f"{k}×{v}" for k, v in bs))
        return "\n".join(lines)


def _short(s, n: int = 120) -> str:
    txt = str(s)
    return txt if len(txt) <= n else txt[:n] + "…"
