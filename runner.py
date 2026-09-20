"""World Model Agent 的回合驱动器（bench 与 demo 共用）。

整个 agent 的本质浓缩在五行里：

    obs → 贝叶斯修正信念 → 在模型内部 rollout N×H 次 → 只走出最优的一步 → 与现实对账

「只走出最优的一步」是关键：每步都重规划，任何偏差都会被下一次观测拉回来。"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from wm.types import Action, Belief, Observation

from envs.ops_world import OpsWorld, OpsWorldConfig


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    total_reward: float
    n_wrong: int
    n_harmful: int
    n_diagnostic: int
    n_probe: int
    mean_surprise: float
    final_entropy: float
    final_confidence: float
    root_mode: Tuple[str, str] = ("-", "-")
    trace: List[str] = field(default_factory=list)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- WM Agent 回合


def run_wm_episode(world: OpsWorld, agent, *, root: str, mode: str,
                   seed: int, verbose: bool = False,
                   extra_agent=None) -> EpisodeResult:
    obs = world.reset(root=root, mode=mode)
    belief = agent.initial_belief()
    # 首次观测本身等价于执行了一次 monitor
    belief, s0 = agent.perceive(belief, Action("monitor", (), cost=1.0), obs)

    surprises: List[float] = [s0]
    total = 0.0
    steps = 0
    n_diag = 1
    n_probe = 0
    trace: List[str] = []
    done = False

    while not done and steps < world.cfg.horizon + 2:
        act, dtrace = agent.decide(belief)
        obs, reward, done, info = world.step(act)
        total += reward
        steps += 1
        if act.name in ("monitor", "probe"):
            n_diag += 1
            if act.name == "probe":
                n_probe += 1
        if verbose:
            h_after = belief.entropy()
            trace.append(f"t{steps:02d} H={h_after:5.2f} → {act.label:28s} "
                         f"(reward {reward:7.1f}) {dtrace.note}")
        belief, s, _ = agent.step(belief, act, obs)
        surprises.append(s)
        if verbose:
            trace[-1] += f"  surprise={s:5.2f}"

    conf = max(belief.posterior().values()) if belief.posterior() else 0.0
    return EpisodeResult(
        success=bool(info.get("fixed")),
        steps=steps,
        total_reward=total,
        n_wrong=world.n_wrong,
        n_harmful=world.n_harmful,
        n_diagnostic=n_diag,
        n_probe=n_probe,
        mean_surprise=_mean(surprises),
        final_entropy=belief.entropy(),
        final_confidence=conf,
        root_mode=(root, mode),
        trace=trace,
    )


# ---------------------------------------------------------------- 对照组回合


def run_baseline_episode(world: OpsWorld, agent, *, root: str, mode: str,
                         verbose: bool = False) -> EpisodeResult:
    obs = world.reset(root=root, mode=mode)
    agent.reset()
    total = 0.0
    steps = 0
    n_diag = 1
    n_probe = 0
    trace: List[str] = []
    done = False
    info: Dict[str, Any] = {}

    while not done and steps < world.cfg.horizon + 2:
        act = agent.decide(obs)
        obs, reward, done, info = world.step(act)
        total += reward
        steps += 1
        if act.name in ("monitor", "probe"):
            n_diag += 1
            if act.name == "probe":
                n_probe += 1
        if verbose:
            dec = agent.decisions[-1] if getattr(agent, "decisions", None) else act.label
            trace.append(f"t{steps:02d} → {act.label:28s} ({reward:7.1f})  [{dec}]")

    return EpisodeResult(
        success=bool(info.get("fixed")),
        steps=steps,
        total_reward=total,
        n_wrong=world.n_wrong,
        n_harmful=world.n_harmful,
        n_diagnostic=n_diag,
        n_probe=n_probe,
        mean_surprise=0.0,          # 对照组结构上没有这个量
        final_entropy=float("nan"),
        final_confidence=float("nan"),
        root_mode=(root, mode),
        trace=trace,
    )


# ---------------------------------------------------------------- 批量统计


def cases(cfg: OpsWorldConfig, limit: Optional[int] = None,
          shuffle: int = 0) -> List[Tuple[str, str]]:
    pairs = [(s, m) for s in cfg.services for m in cfg.modes]
    if shuffle:
        random.Random(shuffle).shuffle(pairs)
    return pairs[:limit] if limit else pairs


def aggregate(results: List[EpisodeResult]) -> Dict[str, float]:
    n = len(results) or 1
    return {
        "success": sum(r.success for r in results) / n,
        "reward": sum(r.total_reward for r in results) / n,
        "steps": sum(r.steps for r in results) / n,
        "wrong": sum(r.n_wrong for r in results) / n,
        "harmful": sum(r.n_harmful for r in results) / n,
        "diagnostic": sum(r.n_diagnostic for r in results) / n,
        "probe": sum(r.n_probe for r in results) / n,
        "surprise": _mean([r.mean_surprise for r in results]),
        "confidence": _mean([r.final_confidence for r in results]),
    }
