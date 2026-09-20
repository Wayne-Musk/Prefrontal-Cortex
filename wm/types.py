"""wm.types -- World-Model Agent 的核心数据结构。

设计要点
--------
1. **信念 ≠ 上下文日志**。信念是「关于世界潜状态的后验分布」。传统 Agent 把历史塞进
   context window，不确定性是隐式的；这里不确定性是显式随机变量，可被搜索、被度量、
   被主动消解。
2. **动作 ≠ 技能条目**。动作是动力学模型里的算子（operator）：模型必须能回答
   「在状态 s 下执行 a，世界会变成什么样、我会看到什么」。不能回答的技能，
   在这个框架里没有资格参与决策，只能作为候选提议存在。
3. **事实带溯源与置信度**，支撑冲突检测、遗忘与记忆巩固。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

Json = Dict[str, Any]


def signature(obj: Any) -> str:
    """把任意可 JSON 化的对象压成稳定签名，用作潜状态 / 信念 / 搜索节点的 key。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- 事实层


@dataclass(frozen=True)
class Fact:
    """一条带置信度与溯源的事实。"""

    key: str
    value: Hashable
    confidence: float = 1.0
    source: str = "obs"
    t: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 必须在 [0,1]，收到 {self.confidence}")

    def conflicts_with(self, other: "Fact") -> bool:
        return self.key == other.key and self.value != other.value

    def __str__(self) -> str:
        return f"{self.key}={self.value}@p{self.confidence:.2f}"


@dataclass
class Observation:
    """环境返回的一次观测：一组事实 + 可选的自由文本备注。"""

    facts: Tuple[Fact, ...] = ()
    note: str = ""
    t: int = 0

    def get(self, key: str, default=None):
        for f in self.facts:
            if f.key == key:
                return f.value
        return default


# ---------------------------------------------------------------- 动作层


@dataclass(frozen=True)
class Action:
    """一个可执行算子。

    与「技能」的区别：技能是 *触发条件 → 执行套路* 的静态映射；
    Action 只是 *(name, params)*，它的语义完全由 WorldModel 的 operator 提供。
    语义与触发解耦， planner 才能做反事实组合。
    """

    name: str
    params: Tuple[Hashable, ...] = ()
    cost: float = 1.0
    risk: float = 0.0            # 副作用 / 不可逆性先验
    reversible: bool = True
    source: str = "skill"        # 提议来源：skill / induced / invented

    @property
    def label(self) -> str:
        if self.params:
            return f"{self.name}({','.join(str(p) for p in self.params)})"
        return self.name

    def __str__(self) -> str:
        return self.label


# ---------------------------------------------------------------- 信念层


@dataclass
class Belief:
    """关于世界潜状态 S 的后验分布 + 已落地的事实层。

    weights / states 构成一个稀疏离散分布 —— 这是 POMDP 的 belief state。
    facts 是确定性的观测沉淀（不必每次都从粒子里重算）。
    """

    weights: Dict[str, float] = field(default_factory=dict)
    states: Dict[str, Json] = field(default_factory=dict)
    facts: Dict[str, Fact] = field(default_factory=dict)
    t: int = 0

    # -- 构造 -------------------------------------------------------
    @staticmethod
    def uniform(states: Iterable[Json]) -> "Belief":
        st = [dict(s) for s in states]
        if not st:
            raise ValueError("uniform 需要至少一个潜状态")
        w = 1.0 / len(st)
        return Belief(
            weights={signature(s): w for s in st},
            states={signature(s): s for s in st},
        )

    # -- 分布运算 ---------------------------------------------------
    def posterior(self) -> Dict[str, float]:
        tot = sum(v for v in self.weights.values() if v > 0.0)
        if tot <= 0:
            return {}
        return {k: v / tot for k, v in self.weights.items() if v > 0.0}

    def mass(self) -> float:
        return sum(v for v in self.weights.values() if v > 0.0)

    def entropy(self) -> float:
        return -sum(p * math.log(max(p, 1e-12)) for p in self.posterior().values())

    def sample_state(self, rng) -> Json:
        post = self.posterior()
        r = rng.random() * sum(post.values())
        acc = 0.0
        for sig, p in post.items():
            acc += p
            if r <= acc:
                return dict(self.states[sig])
        return dict(self.states[next(iter(post))])

    def top(self, k: int = 3) -> List[Tuple[str, float, Json]]:
        post = self.posterior()
        ranked = sorted(post.items(), key=lambda kv: -kv[1])[:k]
        return [(sig, p, self.states[sig]) for sig, p in ranked]

    def map_state(self) -> Json:
        post = self.posterior()
        if not post:
            return {}
        return dict(self.states[max(post, key=post.get)])

    # -- 维护 -------------------------------------------------------
    def copy(self, t: Optional[int] = None) -> "Belief":
        return Belief(
            weights=dict(self.weights),
            states={k: dict(v) for k, v in self.states.items()},
            facts=dict(self.facts),
            t=self.t if t is None else t,
        )

    def prune(self, min_weight: float = 1e-4, max_hyp: int = 48) -> "Belief":
        """剪枝 + 重归一化：信念压缩是长期运行不至于坍塌的关键。"""
        post = self.posterior()
        if not post:
            return
        keep = sorted(post.items(), key=lambda kv: -kv[1])[:max_hyp]
        keep = [(k, v) for k, v in keep if v >= min_weight]
        if not keep:
            # 全部证据都低于阈值 ≠ 世界消失了。保留最优假设，避免信念塌缩为空集合
            keep = sorted(post.items(), key=lambda kv: -kv[1])[:1]
        self.weights = dict(keep)
        self.states = {k: self.states[k] for k in self.weights}
        self.renorm()

    def renorm(self) -> "Belief":
        tot = sum(self.weights.values())
        if tot > 0:
            self.weights = {k: v / tot for k, v in self.weights.items()}
        return self

    # -- 事实层 -----------------------------------------------------
    def set_facts(self, facts: Iterable[Fact], source: str = "obs") -> List[Tuple[Fact, Fact]]:
        """写入事实，返回 [(旧事实, 新事实)] 形式的冲突列表。"""
        conflicts: List[Tuple[Fact, Fact]] = []
        for f in facts:
            old = self.facts.get(f.key)
            if old is not None and old.conflicts_with(f):
                conflicts.append((old, f))
                # 新观测压旧观测，但保留被推翻者的置信度作为污染标记
                self.facts[f.key] = Fact(
                    f.key, f.value, max(f.confidence, old.confidence * 0.5), source, f.t
                )
            else:
                self.facts[f.key] = f
        return conflicts

    def get_fact(self, key: str, default=None):
        f = self.facts.get(key)
        return f.value if f is not None else default

    # -- 搜索去重 ---------------------------------------------------
    def key(self) -> Tuple[str, str]:
        """MCTS 节点 key。刻意不含 t：同一信念在不同路径上应当汇合复用。"""
        post = self.posterior()
        topk = sorted(post.items(), key=lambda kv: -kv[1])[:8]
        fact_key = tuple(
            sorted((f.key, str(f.value), round(f.confidence, 2)) for f in self.facts.values())
        )
        return (str(topk), str(fact_key))


# ---------------------------------------------------------------- 监控记录


@dataclass
class SurpriseRecord:
    """一次「预测 vs 现实」的落差记录 —— 元认知的基本数据单元。"""

    t: int
    action_label: str
    surprise: float              # -log P(观测 | 模型)
    entropy_before: float
    entropy_after: float
    top_before: List[Tuple[str, float]] = field(default_factory=list)
    note: str = ""

    def __str__(self) -> str:
        return f"t={self.t} {self.action_label} surprise={self.surprise:.2f}"


@dataclass
class DecisionTrace:
    """一次决策的可解释快照。"""

    chosen: Action
    root_visits: Dict[str, int] = field(default_factory=dict)
    root_values: Dict[str, float] = field(default_factory=dict)
    root_ig: Dict[str, float] = field(default_factory=dict)
    n_sims: int = 0
    entropy_before: float = 0.0
    note: str = ""

    def summary(self, top: int = 5) -> str:
        rows = sorted(self.root_values.items(), key=lambda kv: -kv[1])[:top]
        return " | ".join(f"{a}:V={v:.1f},N={self.root_visits.get(a, 0)},IG={self.root_ig.get(a, 0):.2f}"
                          for a, v in rows)
