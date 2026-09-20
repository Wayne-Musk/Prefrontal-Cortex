"""评测入口：World Model Agent vs skill+roleplay 对照组。

用法：
    python bench.py                       # 训练分布（已见过的拓扑 / 故障）
    python bench.py --ood                 # 分布外：更大拓扑 + 两种没见过的故障
    python bench.py --misspecify          # 模型与现实不一致时，元认知是否救得回来
    python bench.py --sims 24 --limit 8   # 快速跑

公平性说明（很重要，否则结果不可信）：
    * 所有 agent 在同一组 (root, mode) 案例上逐对比较，每案例同 seed
    * 对照组拿到了环境的完整领域知识（哪些修复对应哪些故障），只缺「信念」
    * Oracle 版对照组甚至拿到了 OOD 新故障的正确修复技能
    * World Model Agent 没有被告知任何新故障；它靠已有的转移算子 + 信息增益搜索来应对
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

from envs.ops_world import OOD, TRAIN, OpsWorld, OpsWorldConfig, build_agent
from baselines import RandomSkillAgent, SkillRoleplayAgent, SkillRoleplayAgentOracle
from runner import EpisodeResult, aggregate, cases, run_baseline_episode, run_wm_episode


TABLE_COLS = ["success", "reward", "steps", "wrong", "harmful", "probe", "surprise", "confidence"]


def _fmt_row(name: str, agg: Dict[str, float]) -> str:
    cells = [
        f"{agg['success']*100:6.1f}%",
        f"{agg['reward']:8.1f}",
        f"{agg['steps']:6.2f}",
        f"{agg['wrong']:6.2f}",
        f"{agg['harmful']:6.2f}",
        f"{agg['probe']:6.2f}",
        f"{agg['surprise']:7.2f}",
        f"{agg['confidence']:6.2f}",
    ]
    return f"{name:<28s}" + "".join(f"{c:>10s}" for c in cells) if False else \
        f"{name:<26s} " + " ".join(cells)


def print_table(rows: List[Tuple[str, Dict[str, float]]], title: str) -> None:
    header = f"{'agent':<26s} {'成功率':>7s} {'总回报':>8s} {'步数':>6s} {'误修':>6s} {'连带损伤':>8s} {'探针':>6s} {'惊讶度':>7s} {'终局置信':>8s}"
    print("\n" + "=" * len(header))
    print(title)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, agg in rows:
        print(_fmt_row(name, agg))
    print("=" * len(header))


def evaluate_wm(cfg: OpsWorldConfig, label: str, *, pairs, sims: int,
                misspecify: bool = False, base_seed: int = 1000,
                verbose_first: bool = False,
                prior_modes=None) -> Tuple[str, Dict[str, float], List[EpisodeResult]]:
    results: List[EpisodeResult] = []
    t0 = time.time()
    for i, (root, mode) in enumerate(pairs):
        seed = base_seed + i
        world = OpsWorld(cfg, rng=random.Random(seed))
        agent = build_agent(cfg, misspecify=misspecify, seed=seed, n_sims=sims,
                            prior_modes=prior_modes)
        results.append(run_wm_episode(world, agent, root=root, mode=mode, seed=seed,
                                      verbose=(verbose_first and i == 0)))
    agg = aggregate(results)
    dt = time.time() - t0
    print(f"  [{label}] {len(pairs)} 个案例 / {dt:.1f}s ({dt/max(1,len(pairs)):.2f}s 每例)")
    if verbose_first and results[0].trace:
        print("\n  ---- 首回合逐帧轨迹 ----")
        for line in results[0].trace:
            print("   " + line)
        print("")
    return label, agg, results


def evaluate_baseline(cfg: OpsWorldConfig, label: str, factory: Callable,
                      pairs, base_seed: int = 1000) -> Tuple[str, Dict[str, float]]:
    results: List[EpisodeResult] = []
    for i, (root, mode) in enumerate(pairs):
        seed = base_seed + i
        world = OpsWorld(cfg, rng=random.Random(seed))
        agent = factory(cfg, seed=seed)
        results.append(run_baseline_episode(world, agent, root=root, mode=mode))
    return label, aggregate(results)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", action="store_true", help="分布外评测")
    ap.add_argument("--misspecify", action="store_true", help="让 agent 的世界模型与现实不一致")
    ap.add_argument("--meta", action="store_true", help="跑元认知专项实验（未见故障）")
    ap.add_argument("--sims", type=int, default=48, help="每次决策的 MCTS 模拟数")
    ap.add_argument("--limit", type=int, default=0, help="限制案例数")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args(argv)

    cfg = OOD if args.ood else TRAIN
    pairs = cases(cfg, limit=args.limit or None)
    tag = "OOD（7 服务 / 6 故障，含 2 种未见过的模式）" if args.ood \
        else "训练分布（4 服务 / 4 故障）"
    print(f"\n场景: {tag}   案例数: {len(pairs)}   MCTS sims: {args.sims}")

    rows: List[Tuple[str, Dict[str, float]]] = []

    if not args.baseline_only:
        label = "World-Model Agent"
        if args.misspecify:
            label += " (模型错配)"
        _, agg, _ = evaluate_wm(cfg, label, pairs=pairs, sims=args.sims,
                                misspecify=args.misspecify, verbose_first=args.verbose)
        rows.append((label, agg))

    _, agg_sr = evaluate_baseline(cfg, "skill+roleplay", lambda c, seed: SkillRoleplayAgent(c, seed=seed), pairs)
    rows.append(("skill+roleplay", agg_sr))
    _, agg_or = evaluate_baseline(cfg, "skill+roleplay(oracle)", lambda c, seed: SkillRoleplayAgentOracle(c, seed=seed), pairs)
    rows.append(("skill+roleplay(oracle)", agg_or))
    _, agg_rd = evaluate_baseline(cfg, "random", lambda c, seed: RandomSkillAgent(c, seed=seed), pairs)
    rows.append(("random", agg_rd))

    print_table(rows, f"结果 · {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
