"""envs.ops_world -- 一个带隐藏状态的运维 POMDP，用来验证架构。

**为什么验证必须在这种环境里做。**
在完全可观测、单步可解的任务上，World Model 打不过直接调 LLM —— 建模开销是白付的。
它真正的优势只在三类条件下显现：

    1. 部分可观测 —— 真正的「根因」看不见，只能靠带噪探针间接推断
    2. 动作有副作用 —— 乱重启会把依赖它的服务一起拖下水
    3. 「先诊断再动手」比「直接动手」便宜 —— 于是 information-seeking 行为有正价值

这三条正是真实 Agent 场景（代码库、浏览器、业务系统）的常态，而不是刻意刁难。

环境 vs 模型的分离很重要：OpsWorld 是**真实世界**，build_world_model 返回的是
agent 对它的**认知**。两者可以设置不一致（misspecify），用来验证元认知模块是否有用。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from wm import (
    Action, AgentConfig, Belief, EmissionModel, Fact, HeuristicProposal,
    Observation, Operator, ProposalConfig, HeuristicProposal as _HP, ValueConfig,
    ValueModel, WorldModel, WMAgent, ModelMonitor, SkillCompiler,
)

# ---------------------------------------------------------------- 领域定义

FIX_FOR: Dict[str, Tuple[str, ...]] = {
    "disk_full": ("clear_disk",),
    "mem_leak": ("restart", "scale_up"),
    "bad_config": ("rollback_config",),
    "bug_crash": ("patch",),
    "network_partition": ("heal_network",),
    "cert_expired": ("rotate_cert",),
}

DETECT: Dict[str, str] = {          # 探针类型 → 它能特异检测到的故障模式
    "disk": "disk_full",
    "memory": "mem_leak",
    "config": "bad_config",
    "crash": "bug_crash",
    "connectivity": "network_partition",
    "tls": "cert_expired",
}

PROBES: Tuple[str, ...] = ("latency",) + tuple(DETECT.keys())

REPAIRS: Dict[str, Tuple[float, float]] = {   # name → (cost, risk)
    "restart": (5.0, 0.6),
    "clear_disk": (3.0, 0.1),
    "rollback_config": (2.0, 0.2),
    "patch": (6.0, 0.3),
    "scale_up": (4.0, 0.2),
    "heal_network": (4.0, 0.3),
    "rotate_cert": (3.0, 0.1),
}


@dataclass
class OpsWorldConfig:
    services: Tuple[str, ...] = ("api", "db", "cache", "worker")
    deps: Dict[str, List[str]] = field(default_factory=lambda: {
        "api": ["db", "cache"], "db": [], "cache": [], "worker": ["db", "cache"],
    })
    modes: Tuple[str, ...] = ("disk_full", "mem_leak", "bad_config", "bug_crash")
    horizon: int = 12
    noise: float = 0.10
    p_cascade: float = 0.50       # 重启引发下游级联的概率（真实值）
    wrong_penalty: float = 8.0
    success_reward: float = 100.0
    timeout_penalty: float = 20.0


TRAIN = OpsWorldConfig()

OOD = OpsWorldConfig(
    services=("gateway", "api", "billing", "search", "worker", "db", "cache"),
    deps={
        "gateway": ["api"], "api": ["db", "cache"], "billing": ["db", "api"],
        "search": ["cache"], "worker": ["db", "cache"], "db": [], "cache": [],
    },
    modes=("disk_full", "mem_leak", "bad_config", "bug_crash",
           "network_partition", "cert_expired"),
    horizon=14,
    noise=0.12,
)


# ---------------------------------------------------------------- 状态工具


def rdeps_of(deps: Dict[str, List[str]], svc: str) -> List[str]:
    return [s for s, ds in deps.items() if svc in ds]


def make_state(services: Sequence[str], root: Optional[str], mode: Optional[str],
               deps: Optional[Dict[str, List[str]]] = None,
               fixed: bool = False, harm: int = 0) -> Dict:
    """构造一个潜状态。down=故障根，degraded=被级联波及的受害者，ok=正常。"""
    svc_status = {s: "ok" for s in services}
    if not fixed and root and deps:
        svc_status[root] = "down"
        for s in rdeps_of(deps, root):
            svc_status[s] = "degraded"
    return {
        "services": svc_status,
        "root": root,
        "mode": mode if mode else "none",
        "fixed": bool(fixed),
        "harm": harm,
    }


# ---------------------------------------------------------------- 真实环境


class OpsWorld:
    """地面真值环境。Agent 永远看不到它的 state。"""

    def __init__(self, cfg: OpsWorldConfig, rng: Optional[random.Random] = None,
                 root: Optional[str] = None, mode: Optional[str] = None):
        self.cfg = cfg
        self.rng = rng or random.Random(0)
        self.root = root or self.rng.choice(cfg.services)
        self.mode = mode or self.rng.choice(cfg.modes)
        self.state = make_state(cfg.services, self.root, self.mode, cfg.deps)
        self.t = 0
        self.emission = build_emission(cfg)
        self.done = False
        self.n_wrong = 0
        self.n_harmful = 0

    # -- API ------------------------------------------------------
    def reset(self, root: Optional[str] = None, mode: Optional[str] = None) -> Observation:
        self.root = root or self.rng.choice(self.cfg.services)
        self.mode = mode or self.rng.choice(self.cfg.modes)
        self.state = make_state(self.cfg.services, self.root, self.mode, self.cfg.deps)
        self.t = 0
        self.done = False
        self.n_wrong = 0
        self.n_harmful = 0
        self.emission = build_emission(self.cfg)
        return self.emission.sample(self.state, Action("monitor"), self.rng)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        cfg = self.cfg
        cost = action.cost
        reward = -cost
        if self.done:
            return self.emission.sample(self.state, action, self.rng), 0.0, True, {"repeat": True}

        if action.name in REPAIRS and action.params:
            svc = action.params[0]
            effective = (svc == self.root and action.name in FIX_FOR.get(self.mode, ()))
            if effective:
                self.state = make_state(cfg.services, None, None, cfg.deps,
                                        fixed=True, harm=self.state["harm"])
                reward += cfg.success_reward
                self.done = True
            else:
                self.n_wrong += 1
                reward -= cfg.wrong_penalty
                self.state["harm"] = self.state.get("harm", 0) + 1
                # 副作用：一次无效的修复动作会把依赖该服务的下游拖下水。
                # Agent 的模型未必知道这一项 —— 这正是 --misspecify 要检测的错配。
                for d in rdeps_of(cfg.deps, svc):
                    if self.rng.random() < cfg.p_cascade:
                        if self.state["services"].get(d) == "ok":
                            self.state["services"][d] = "degraded"
                            self.state["harm"] = self.state["harm"] + 1
                            self.n_harmful += 1

        self.t += 1
        timeout = (not self.done) and self.t >= cfg.horizon
        if timeout:
            reward -= cfg.timeout_penalty
            self.done = True
        obs = self.emission.sample(self.state, action, self.rng)
        info = {"t": self.t, "wrong": self.n_wrong, "harm": self.state.get("harm", 0),
                "done": self.done, "fixed": self.state.get("fixed", False)}
        return obs, reward, self.done, info

    # 供 recording grounding 样本使用（真实潜状态，只有「上帝视角」能拿到）
    def true_state(self) -> Dict:
        return dict(self.state)


# ---------------------------------------------------------------- 观测模型


def _domain_for_key(key: str, truth_value: Hashable) -> Sequence[Hashable]:
    if key.startswith("lat@"):
        return ("ok", "degraded", "down")
    return ("normal", "high")


def build_emission(cfg: OpsWorldConfig, noise: Optional[float] = None) -> EmissionModel:
    def truth_fn(state: Dict, action: Action) -> Dict[str, Hashable]:
        services = sorted(state.get("services", {}).keys())
        statuses = {s: state["services"].get(s, "ok") for s in services}

        def health_snapshot() -> Dict[str, Hashable]:
            return {f"lat@{s}": statuses[s] for s in services}

        if action.name == "monitor":
            return health_snapshot()
        if action.name == "probe" and len(action.params) >= 2:
            svc, kind = action.params[0], action.params[1]
            if kind == "latency":
                return {f"lat@{svc}": statuses.get(svc, "ok")}
            positive = (state.get("root") == svc and DETECT.get(kind) == state.get("mode"))
            key = f"{kind}@{svc}"
            if key not in health_snapshot() and kind != "latency":
                return {key: "high" if positive else "normal"}
            return {key: "high" if positive else "normal"}
        # 任何维修动作之后，自然会看到一次全局健康快照
        return health_snapshot()

    return EmissionModel(truth_fn=truth_fn, noise=noise if noise is not None else cfg.noise,
                         domain_fn=_domain_for_key)


# ---------------------------------------------------------------- Agent 侧世界模型


def prior_states(cfg: OpsWorldConfig,
                 modes: Optional[Sequence[str]] = None) -> List[Dict]:
    """先验假设空间。modes 可显式收窄 —— 用来模拟「模型没见过某种故障」。"""
    modes = tuple(modes) if modes is not None else tuple(cfg.modes)
    states: List[Dict] = []
    for svc in cfg.services:
        for mode in modes:
            states.append(make_state(cfg.services, svc, mode, cfg.deps))
    # 「其实没出事」的假设也必须在先验里，否则 agent 永远不敢得出「没问题」的结论
    states.append(make_state(cfg.services, None, None, cfg.deps, fixed=True))
    return states


def action_space_fn(cfg: OpsWorldConfig) -> Callable[[Dict, List[Dict]], List[Action]]:
    def fn(state: Dict, others: List[Dict]) -> List[Action]:
        services = sorted(state.get("services", {}).keys()) or sorted(cfg.services)
        acts: List[Action] = [Action("monitor", (), cost=1.0, risk=0.0)]
        for svc in services:
            for p in PROBES:
                acts.append(Action("probe", (svc, p), cost=2.0, risk=0.0))
            for name, (c, r) in REPAIRS.items():
                acts.append(Action(name, (svc,), cost=c, risk=r))
        return acts
    return fn


def build_operators(cfg: OpsWorldConfig, misspecify: bool = False) -> Dict[str, Operator]:
    """构建 TRANSITION 算子。misspecify=True 时对 restart 的级联副作用「不知道」。"""
    p_cascade_model = 0.02 if misspecify else cfg.p_cascade
    deps = cfg.deps

    def healthy(state: Dict) -> Dict:
        return make_state(sorted(state.get("services", {}).keys()) or list(cfg.services),
                          None, None, deps, fixed=True, harm=state.get("harm", 0))

    ops: Dict[str, Operator] = {}

    def identity(name: str) -> None:
        ops[name] = Operator(action_name=name,
                             transition=lambda s, a, rng: dict(s))

    identity("monitor")
    identity("probe")

    for name in REPAIRS:
        def mk(name_: str):
            def trans(state: Dict, action: Action, rng: random.Random) -> Dict:
                if state.get("fixed"):
                    return dict(state)
                svc = action.params[0] if action.params else None
                applicable = state.get("root") == svc and name_ in FIX_FOR.get(
                    state.get("mode"), ())
                if applicable:
                    return healthy(state)
                new = dict(state)
                new["services"] = dict(state.get("services", {}))
                new["harm"] = state.get("harm", 0) + 1
                for d in rdeps_of(deps, svc):
                    if rng.random() < p_cascade_model:
                        if new["services"].get(d) == "ok":
                            new["services"][d] = "degraded"
                            new["harm"] = new["harm"] + 1
                return new
            return trans
        ops[name] = Operator(action_name=name, transition=mk(name))
    return ops


def build_world_model(cfg: OpsWorldConfig, misspecify: bool = False,
                      noise: Optional[float] = None,
                      prior_modes: Optional[Sequence[str]] = None) -> WorldModel:
    return WorldModel(
        prior_states=prior_states(cfg, prior_modes),
        operators=build_operators(cfg, misspecify=misspecify),
        emission=build_emission(cfg, noise=noise),
        action_space_fn=action_space_fn(cfg),
    )


# ---------------------------------------------------------------- 组装 Agent


def build_agent(cfg: OpsWorldConfig, *, misspecify: bool = False, seed: int = 0,
                agent_cfg: Optional[AgentConfig] = None, llm=None,
                n_sims: Optional[int] = None, noise: Optional[float] = None,
                prior_modes: Optional[Sequence[str]] = None,
                lambda_info: float = 5.0,
                repair_threshold: float = 0.45) -> WMAgent:
    """prior_modes 收窄先验时，可以模拟「世界的某部分是模型完全没见过的」。"""
    model = build_world_model(cfg, misspecify=misspecify, noise=noise,
                              prior_modes=prior_modes)

    def goal_test(s: Dict) -> bool:
        return bool(s.get("fixed"))

    def cost_fn(s: Dict) -> float:
        return 8.0 * float(s.get("harm", 0))

    value = ValueModel(goal_test=goal_test, cost_fn=cost_fn,
                       cfg=ValueConfig(w_goal=100.0, lambda_info=lambda_info,
                                       lambda_cost=1.0, lambda_risk=2.0, gamma=0.95))
    proposal = (_HP if False else HeuristicProposal)(
        cfg=ProposalConfig(max_branching=12, repair_mass_threshold=repair_threshold,
                               top_services=3),
        probes=PROBES, repairs=tuple(REPAIRS.keys()),
        fix_map={m: v[0] for m, v in FIX_FOR.items()},
        detect_map=DETECT,
    )
    acfg = agent_cfg or AgentConfig()
    if n_sims is not None:
        acfg.n_sims = n_sims
    return WMAgent(model=model, value=value, proposal=proposal,
                   monitor=ModelMonitor(), compiler=SkillCompiler(),
                   cfg=acfg, rng=random.Random(seed), llm=llm)
