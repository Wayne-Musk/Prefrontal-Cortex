"""wm.value -- 目标 / 价值模型。

传统 Agent 的「目标」写在 prompt 里，是一句祈使句，無法被搜索过程当作可微信号使用。
这里目标是一个 *势函数*：给定信念（不是给定文本），返回一个标量。

区分两类目标是本模块的核心设计：
    achievement goal  —— 终局要达成的状态（可被打折累加）
    constraint goal   —— 全程必须满足的红线（违反即大幅惩罚，不打折）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .types import Action, Belief, Json


@dataclass
class ValueConfig:
    w_goal: float = 100.0           # 达成 achievement goal 的收益
    w_constraint: float = 60.0      # 违反约束的单次惩罚
    lambda_info: float = 6.0        # 单位信息增益的价值（nats → 收益）
    lambda_cost: float = 1.0        # 资源代价权重
    lambda_risk: float = 2.0        # 副作用先验权重
    unknown_op_penalty: float = 8.0  # 动力学没有覆盖的动作：不确定的确定性惩罚
    gamma: float = 0.95


class ValueModel:
    """把信念映射成标量价值。planner 的唯一裁判。"""

    def __init__(self,
                 goal_test: Callable[[Json], bool],
                 constraint_fn: Callable[[Json], List[str]] = lambda s: [],
                 cost_fn: Callable[[Json], float] = lambda s: 0.0,
                 cfg: Optional[ValueConfig] = None):
        self.goal_test = goal_test
        self.constraint_fn = constraint_fn
        self.cost_fn = cost_fn
        self.cfg = cfg or ValueConfig()
        self.constraint_violations: Dict[str, int] = {}

    # -- 终局价值 ---------------------------------------------------
    def terminal_value(self, belief: Belief) -> float:
        if belief.mass() <= 0:
            return -self.cfg.w_goal
        p_goal = 0.0
        exp_penalty = 0.0
        for sig, p in belief.posterior().items():
            s = belief.states[sig]
            if self.goal_test(s):
                p_goal += p
            exp_penalty += p * self.cost_fn(s)
        return self.cfg.w_goal * p_goal - exp_penalty

    def violates(self, state: Json) -> bool:
        vs = self.constraint_fn(state)
        for v in vs:
            self.constraint_violations[v] = self.constraint_violations.get(v, 0) + 1
        return bool(vs)

    # -- 单步收益 ---------------------------------------------------
    def step_reward(self, action: Action, info_gain: float,
                    operator_reliability: float = 1.0,
                    is_covered: bool = True,
                    violated: bool = False) -> float:
        c = self.cfg
        r = c.lambda_info * info_gain
        r -= c.lambda_cost * action.cost
        # 模型对该动��越没把握，其副作用先验越应当被放大 —— 这是保守性的来源
        r -= c.lambda_risk * action.risk * (2.0 - operator_reliability)
        if not is_covered:
            r -= c.unknown_op_penalty
        if violated:
            r -= c.w_constraint
        return r

    def discounted(self, rewards: List[float], tail: float) -> float:
        g = self.cfg.gamma
        total = tail * (g ** len(rewards))
        for i, r in enumerate(rewards):
            total += (g ** i) * r
        return total
