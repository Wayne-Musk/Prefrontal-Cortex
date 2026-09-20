"""wm.planner -- 信念空间上的反事实搜索（Bayes-adaptive MCTS）。

和 ReAct / CoT 式规划的本质区别：

    ReAct:   belief(context) → LLM → a₁ → obs → LLM → a₂ …
             每一步都是 depth-1 贪心，**没有 backup**：一旦走错，错误无法回溯修正。
    MCTS:    在模型内部展开 H 步 × N 条分支，用 rollout 的回报去更新**较早节点的估值**，
             即 credit assignment 是沿着树往上走的。早先那个「先做诊断再说」的决定，
             会因为后续修复成功而被追溯地加分。

另一个关键点：**信息增益被写进了目标函数**。
rollout 里减小信念熵的动作会拿到 intrinsic reward，于是 agent 会为了「看清楚」而行动，
即使这一步对外部世界没有任何改变。这是传统范式结构上做不到的行为。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .inference import bayes_correct, expected_information_gain, predict_step
from .types import Action, Belief, DecisionTrace, Observation


@dataclass
class PlannerConfig:
    n_sims: int = 48
    rollout_depth: int = 4
    uct_c: float = 1.4
    ig_samples: int = 1
    K_propagate: int = 1
    rollout_temperature: float = 0.6
    max_nodes: int = 4096
    stop_certainty: float = 0.97      # 达成目标的信念够高就提前终止 rollout
    risk_scale: float = 1.0           # monitor 触发保守模式时上调


class _Node:
    __slots__ = ("belief", "parent", "act", "children", "N", "Q",
                 "untried", "depth", "r_in")

    def __init__(self, belief: Belief, parent: Optional["_Node"], act: Optional[Action],
                 depth: int, r_in: float = 0.0):
        self.belief = belief
        self.parent = parent
        self.act = act
        self.children: Dict[str, _Node] = {}
        self.N = 0
        self.Q = 0.0
        self.untried: List[Tuple[Action, float]] = []
        self.depth = depth
        self.r_in = r_in


class MCTSPlanner:
    """在 WorldModel 内部做想象 rollout 的规划器。全程不触碰真实环境。"""

    def __init__(self, model, value, proposal, monitor=None,
                 cfg: Optional[PlannerConfig] = None, rng: Optional[random.Random] = None):
        self.model = model
        self.value = value
        self.proposal = proposal
        self.monitor = monitor
        self.cfg = cfg or PlannerConfig()
        self.rng = rng or random.Random(0)
        self._ig_cache: Dict[Tuple, float] = {}
        self._prop_cache: Dict[Tuple, List[Tuple[Action, float]]] = {}
        self.n_nodes = 0

    # ------------------------------------------------------------ 对外接口
    def plan(self, belief: Belief) -> Tuple[Action, DecisionTrace]:
        self._ig_cache.clear()
        self._prop_cache.clear()
        self.n_nodes = 0
        root = _Node(belief.copy(), None, None, 0)
        self.n_nodes += 1

        support = self.model.support(belief)
        root.untried = self._propose(belief, support)
        if not root.untried:
            raise RuntimeError("信念下没有任何可执行动作，检查 action_space_fn")

        for _ in range(self.cfg.n_sims):
            if self.n_nodes > self.cfg.max_nodes:
                break
            node = self._select(root)
            child = self._expand(node)
            target = child if child is not None else node
            tail = self._rollout(target)
            self._backup(target, tail)

        trace = DecisionTrace(
            chosen=None,
            root_visits={c.act.label: c.N for c in root.children.values() if c.act},
            root_values={c.act.label: c.Q for c in root.children.values() if c.act},
            root_ig={},
            n_sims=self.cfg.n_sims,
            entropy_before=belief.entropy(),
        )
        # 选边：优先访问次数，Q 兜底（访问次数对小值差异更稳健）
        if root.children:
            best = max(root.children.values(), key=lambda c: (c.N, c.Q))
        else:
            (best_act, _), = sorted(root.untried, key=lambda kv: -kv[1])[:1]
            best = None
        chosen = best.act if best is not None else best_act
        trace.chosen = chosen
        for c in root.children.values():
            if c.act:
                trace.root_ig[c.act.label] = self._ig(belief, c.act)
        return chosen, trace

    # ------------------------------------------------------------ MCTS 四步
    def _select(self, root: _Node) -> _Node:
        """下降到一个「还没走完所有候选」的节点。

        停止条件必须包含 ``not node.untried``：否则一旦根节点有了第一个孩子就会被
        无视其余候选一路下降，广度上永远只探索到一条链 —— 这是 belief-tree MCTS 最常见的
        实现陷阱，症状是 planner 反复选中同一个动作。
        """
        node = root
        while not node.untried and node.children and node.depth < self.cfg.rollout_depth:
            nxt = self._uct_child(node)
            if nxt is None:
                break
            node = nxt
        return node

    def _uct_child(self, node: _Node):
        log_n = math.log(max(node.N, 1) + 1.0)
        best, best_score = None, -1e18
        for ch in node.children.values():
            if ch.N == 0:
                score = 1e9 - self.rng.random()
            else:
                score = ch.Q + self.cfg.uct_c * math.sqrt(log_n / ch.N)
            if score > best_score:
                best, best_score = ch, score
        return best

    def _expand(self, node: _Node) -> Optional[_Node]:
        """取出一个未尝试的动作，用一次采样的观测生成子节点（信念树是随机的）。"""
        if not node.untried or node.depth >= self.cfg.rollout_depth:
            return None
        act, _ = node.untried.pop(0)     # best-first：先验分高的候选先展开
        child, reward = self._step_and_reward(node.belief, act)
        ch = _Node(child, node, act, node.depth + 1, r_in=reward)
        node.children[act.label] = ch
        self.n_nodes += 1
        if ch.depth < self.cfg.rollout_depth:
            support = self.model.support(ch.belief)
            ch.untried = self._propose(ch.belief, support)
        return ch

    def _rollout(self, node: _Node) -> float:
        """Bayes-adaptive rollout：

        开局从当前信念里采一个「真实世界」h*，之后整个 rollout 在 h* 上演化和产生观测，
        但 **agent 并不知道 h*** —— 它只能靠观测去更新自己的信念副本。
        这样得到的累积奖励是真实期望收益的无偏估计，信息增益也不会被高估。
        """
        belief = node.belief.copy()
        h = belief.sample_state(self.rng)
        rewards: List[float] = []
        gamma = self.value.cfg.gamma
        steps = max(0, self.cfg.rollout_depth - node.depth)
        for _ in range(steps):
            if self.value.terminal_value(belief) >= self.cfg.stop_certainty * self.value.cfg.w_goal:
                break
            cands = self._propose(belief, self.model.support(belief))
            act = self._policy_sample(cands)
            h2 = self.model.sample_transition(h, act, self.rng)
            obs = self.model.sample_emission(h2, act, self.rng)
            bbar = predict_step(belief, act, self.model, self.rng, K=self.cfg.K_propagate)
            b2, _ = bayes_correct(bbar, act, obs, self.model)
            b2.renorm()
            b2.set_facts(obs.facts)
            b2.prune()
            ig = belief.entropy() - b2.entropy()
            rewards.append(self._reward(act, ig, bbar))
            h, belief = h2, b2
        return self.value.discounted(rewards, self.value.terminal_value(belief)) if rewards \
            else self.value.terminal_value(belief)

    def _backup(self, node: _Node, tail: float) -> None:
        gamma = self.value.cfg.gamma
        V = tail
        cur: Optional[_Node] = node
        while cur is not None:
            V = cur.r_in + gamma * V
            cur.N += 1
            cur.Q += (V - cur.Q) / cur.N
            cur = cur.parent

    # ------------------------------------------------------------ 工具
    def _step_and_reward(self, belief: Belief, act: Action) -> Tuple[Belief, float]:
        bbar = predict_step(belief, act, self.model, self.rng, K=self.cfg.K_propagate)
        bbar.renorm()
        h = belief.sample_state(self.rng)
        h2 = self.model.sample_transition(h, act, self.rng)
        obs = self.model.sample_emission(h2, act, self.rng)
        b2, _ = bayes_correct(bbar, act, obs, self.model)
        b2.renorm()
        b2.set_facts(obs.facts)
        b2.t = belief.t + 1
        b2.prune()
        ig = max(0.0, belief.entropy() - b2.entropy())
        return b2, self._reward(act, ig, bbar)

    def _reward(self, act: Action, ig: float, bbar: Belief) -> float:
        op = self.model.operators.get(act.name)
        rel = op.reliability if op is not None else 0.5
        if self.monitor is not None:
            rel = min(rel, self.monitor.reliability(act.name))
        p_viol = sum(p for sig, p in bbar.posterior().items()
                     if self.value.violates(bbar.states[sig]))
        base = self.value.step_reward(
            act, ig, operator_reliability=rel,
            is_covered=self.model.is_covered(act),
            violated=False,
        )
        return base - self.value.cfg.w_constraint * p_viol

    def _ig(self, belief: Belief, act: Action) -> float:
        k = (belief.key(), act.label)
        if k in self._ig_cache:
            return self._ig_cache[k]
        v = expected_information_gain(belief, act, self.model, self.rng,
                                      n_samples=self.cfg.ig_samples,
                                      K=self.cfg.K_propagate)
        self._ig_cache[k] = v
        return v

    def _propose(self, belief: Belief, support: List[Action]) -> List[Tuple[Action, float]]:
        k = belief.key()
        if k in self._prop_cache:
            return self._prop_cache[k]
        out = self.proposal(belief, self.model, support)
        self._prop_cache[k] = out
        return out

    def _policy_sample(self, cands: List[Tuple[Action, float]]) -> Action:
        if not cands:
            return Action("monitor", (), cost=1.0, source="fallback")
        t = max(1e-3, self.cfg.rollout_temperature)
        weights = [math.exp(s / t) for _, s in cands]
        return self.rng.choices([a for a, _ in cands], weights=weights, k=1)[0]
