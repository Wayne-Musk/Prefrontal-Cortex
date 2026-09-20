"""把 World-Model Agent 的伪代码展开成可运行的 Python 闭环。

这不是另一个 planner，而是把 ``Others/Code-loop.py`` 中的抽象步骤
接到仓库已有的 WMAgent、OpsWorld 和模型监控器上：每次只提交一步，
下一次观测回来后重新规划。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from envs.ops_world import OpsWorld
from wm.types import Action


@dataclass
class CodeLoopTask:
    """一次任务的环境参数和运行预算。"""

    root: str
    mode: str
    horizon: Optional[int] = None
    seed: int = 0
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeLoopResult:
    """闭环的可审计结果。"""

    success: bool
    state: Dict[str, Any]
    steps: int
    total_reward: float
    mean_surprise: float
    actions: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


def run(task: CodeLoopTask, world: OpsWorld, agent, *,
        verbose: bool = False) -> CodeLoopResult:
    """执行 ``observe -> imagine -> simulate -> evaluate -> act -> learn``。

    ``agent.decide`` 内部执行候选提议、MCTS 搜索和价值评估；这里保留
    伪代码的外层控制流，并把真实环境提交和预测-现实对账显式写出来。
    """
    obs = world.reset(root=task.root, mode=task.mode)
    belief = agent.initial_belief()
    belief, surprise = agent.perceive(
        belief, Action("monitor", (), cost=1.0), obs
    )
    surprises = [surprise]
    actions: List[str] = []
    trace: List[str] = []
    total_reward = 0.0
    done = False
    max_steps = task.horizon or world.cfg.horizon

    while not done and len(actions) < max_steps:
        # imagine.generate + simulator.rollout + evaluator.score + search.select
        # 由 agent.decide -> MCTSPlanner.plan 在模型内部完成，不触碰 world。
        action, decision = agent.decide(belief)
        predicted_state = belief.map_state()

        # planner.refine + executor.commit：只把当前最优的一步交给现实环境。
        obs, reward, done, _ = world.step(action)
        actual_state = world.true_state()
        total_reward += reward
        actions.append(action.label)

        # compare(best.predicted_state, actual.state)
        prediction_error = _state_difference(predicted_state, actual_state)
        belief, surprise, learning = agent.step(belief, action, obs)
        surprises.append(surprise)
        if verbose:
            trace.append(
                f"t{len(actions):02d} {action.label:28s} "
                f"reward={reward:6.1f} surprise={surprise:5.2f} "
                f"error={prediction_error:.2f} {decision.note}"
            )

        # learning 是 learner.update 的结果，保留在这里便于调试器观察。
        _ = learning

    return CodeLoopResult(
        success=bool(world.true_state().get("fixed")),
        state=world.true_state(),
        steps=len(actions),
        total_reward=total_reward,
        mean_surprise=sum(surprises) / len(surprises),
        actions=actions,
        trace=trace,
    )


def _state_difference(predicted: Dict[str, Any], actual: Dict[str, Any]) -> float:
    """返回一个轻量的结构差异分数，供日志和外部监控使用。"""
    keys = set(predicted) | set(actual)
    if not keys:
        return 0.0
    different = sum(predicted.get(key) != actual.get(key) for key in keys)
    return different / len(keys)


if __name__ == "__main__":
    from envs.ops_world import TRAIN, build_agent

    demo_task = CodeLoopTask(root="api", mode="disk_full", seed=7)
    demo_world = OpsWorld(TRAIN)
    demo_agent = build_agent(TRAIN, seed=demo_task.seed, n_sims=32)
    demo_result = run(demo_task, demo_world, demo_agent, verbose=True)
    print(f"success={demo_result.success} steps={demo_result.steps} "
          f"reward={demo_result.total_reward:.1f}")
    for line in demo_result.trace:
        print(line)
