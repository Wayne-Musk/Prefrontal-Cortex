"""wm.model -- 生成式世界模型：转移动力学 T(s'|s,a) + 观测模型 P(o|s',a)。

分工上的关键设计：

    Operator           —— 单个动作的局部语义（由 skill grounding 从轨迹中归纳，也可人工注入）
    EmissionModel      —— 「什么样的世界产生什么样的观测」，直接给出解析 likelihood
    WorldModel         —— 把两者组合成一个可 rollout 的生成式模拟器

与传统范式的本质差别：**技能在这里被降级了**。技能不再是「匹配到触发词就执行」的
决策主体，而是：
    (a) 向 planner 提议候选动作的先验源（proposal policy）
    (b) 向 dynamics 贡献可执行的算子显式երբ semantics
决策权从「技能自己」转移到「模型 + 搜索」。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from .types import Action, Belief, Fact, Json, Observation, signature


# ---------------------------------------------------------------- 算子


@dataclass
class Operator:
    """一个动作的动力学语义。``transition`` 允许随机（返回采样后的潜状态）。"""

    action_name: str
    transition: Callable[[Json, Action, "random.Random"], Json]
    precondition: Callable[[Json, Action], bool] = lambda s, a: True
    learned: bool = False
    n_samples: int = 0
    err_ema: float = 0.0        # 观测到的预测误差 EMA
    support_count: int = 0      # 归纳时支撑该算子的样本数

    @property
    def reliability(self) -> float:
        """0~1，算子可信度。越低，planner 越应当在该算子上保守。"""
        return math.exp(-self.err_ema)

    def fidelity(self) -> Dict[str, float]:
        return {"n": float(self.n_samples), "err_ema": self.err_ema,
                "reliability": self.reliability}


def identity_transition(state: Json, action: Action, rng) -> Json:
    return dict(state)


# ---------------------------------------------------------------- 观测模型


@dataclass
class EmissionModel:
    """truth_fn(state, action) -> {key: 真值}，再套一个均匀混淆噪声。

    概率上的约定（与 Environment 共用同一套，保证除故意的 misspecification 外模型良定）：
        P(观测 == 真值)       = 1 - noise
        P(观测 == 某个假值)   = noise / (|domain| - 1)
    """

    truth_fn: Callable[[Json, Action], Dict[Hashable, Hashable]]
    domain: Tuple[Hashable, ...] = ("ok", "degraded", "down", "normal", "high")
    noise: float = 0.10
    label_fn: Callable[[Hashable, str], str] = field(default=lambda v, k: str(v))
    # 可选：按观测 key 给出各自的取值域，避免二值信号被三值噪声稀释
    domain_fn: Optional[Callable[[Hashable, Hashable], Sequence[Hashable]]] = None

    def truth(self, state: Json, action: Action) -> Dict[Hashable, Hashable]:
        return self.truth_fn(state, action)

    def _domain_for(self, key: Hashable, tv: Hashable) -> Sequence[Hashable]:
        if self.domain_fn is not None:
            dom = self.domain_fn(key, tv)
            if dom:
                return tuple(dom)
        return self.domain

    def sample(self, state: Json, action: Action, rng: random.Random) -> Observation:
        truth = self.truth(state, action)
        facts: List[Fact] = []
        for key, val in truth.items():
            dom = self._domain_for(key, val)
            alts = [v for v in dom if v != val]
            if alts and rng.random() < self.noise:
                val = rng.choice(alts)
            facts.append(Fact(str(key), val, confidence=1.0 - self.noise, source="sensor"))
        return Observation(facts=tuple(facts), note=f"after {action.label}")

    def logprob(self, state: Json, action: Action, obs: Observation) -> float:
        truth = self.truth(state, action)
        lp = 0.0
        for f in obs.facts:
            tv = truth.get(f.key)
            if tv is None:
                # 该假设根本没有预测到这个观测维度 → 极不可能，但不是零
                lp += math.log(1e-6)
                continue
            if f.value == tv:
                lp += math.log(max(1.0 - self.noise, 1e-12))
            else:
                dom = self._domain_for(f.key, tv)
                n_alt = max(1, len([v for v in dom if v != tv]))
                lp += math.log(max(self.noise / n_alt, 1e-12))
        return lp


# ---------------------------------------------------------------- 世界模型


@dataclass
class WorldModelConfig:
    K_propagate: int = 1            # 预测步的蒙特卡洛展开数
    unknown_action_penalty: float = math.log(1e-6)
    max_prior_states: int = 512


class WorldModel:
    """可被 rollout 的生成式世界模型。

    rollout 全程只在信念空间里发生：不调工具、不写文件、不花钱、不可逆转。
    这是「零代价试错」的来源。
    """

    def __init__(self,
                 prior_states: Sequence[Json],
                 operators: Dict[str, Operator],
                 emission: EmissionModel,
                 action_space_fn: Callable[[Json, List[Json]], List[Action]],
                 cfg: Optional[WorldModelConfig] = None):
        self.prior_states = [dict(s) for s in prior_states][: (cfg or WorldModelConfig()).max_prior_states]
        self.operators = dict(operators)
        self.emission = emission
        self.action_space_fn = action_space_fn
        self.cfg = cfg or WorldModelConfig()
        self._uncovered: Dict[str, int] = {}       # 无语义覆盖的动作 → 模型盲区计数

    # -- 信念入口 ---------------------------------------------------
    def initial_belief(self) -> Belief:
        return Belief.uniform(self.prior_states)

    # -- 动力学 -----------------------------------------------------
    def sample_transition(self, state: Json, action: Action, rng: random.Random) -> Json:
        op = self.operators.get(action.name)
        if op is None:
            self._uncovered[action.name] = self._uncovered.get(action.name, 0) + 1
            return identity_transition(state, action, rng)
        return op.transition(dict(state), action, rng)

    def emission_logprob(self, state: Json, action: Action, obs: Observation) -> float:
        return self.emission.logprob(state, action, obs)

    def sample_emission(self, state: Json, action: Action, rng: random.Random) -> Observation:
        return self.emission.sample(state, action, rng)

    # -- 动作空间 ---------------------------------------------------
    def _representative_states(self, belief: Belief, k: int = 3) -> List[Json]:
        states = [s for _, _, s in belief.top(k)]
        if not states:
            # 信念退化了：退回先验，保证搜索仍能启动（比直接崩掉安全得多）
            states = [self.prior_states[0]] if self.prior_states else []
        return states

    def support(self, belief: Belief, max_actions: Optional[int] = None) -> List[Action]:
        """当前信念下可执行的动作集合（按代表性状态枚举，过滤不可能满足前提的）。"""
        seen: Dict[str, Action] = {}
        states = self._representative_states(belief)
        for s in states:
            for a in self.action_space_fn(s, states):
                if a.label in seen:
                    continue
                op = self.operators.get(a.name)
                if op is not None and not any(op.precondition(ss, a) for ss in states):
                    continue
                seen[a.label] = a
        acts = list(seen.values())
        if max_actions is not None and len(acts) > max_actions:
            acts.sort(key=lambda a: (a.cost + a.risk))
            acts = acts[:max_actions]
        return acts

    def is_covered(self, action: Action) -> bool:
        return action.name in self.operators

    def add_operator(self, op: Operator) -> None:
        self.operators[op.action_name] = op

    def blind_spots(self, top: int = 5) -> List[Tuple[str, int]]:
        return sorted(self._uncovered.items(), key=lambda kv: -kv[1])[:top]

    def stats(self) -> Dict[str, float]:
        ops = self.operators.values()
        n = len(self.operators)
        return {
            "n_operators": float(n),
            "mean_reliability": (sum(o.reliability for o in ops) / n) if n else 0.0,
            "uncovered_action_types": float(len(self._uncovered)),
        }
