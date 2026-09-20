"""在**训练集**上做超参网格搜索（不允许用测试集调参，所以这里只跑 TRAIN）。"""
from __future__ import annotations

import argparse
import itertools
import random
import sys

from envs.ops_world import TRAIN, OpsWorld, build_agent
from runner import aggregate, cases, run_wm_episode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--grid", type=str, default="5,8,11", help="lambda_info 取值")
    ap.add_argument("--thrs", type=str, default="0.20,0.35,0.45")
    args = ap.parse_args()

    pairs = cases(TRAIN)
    print(f"{'lam_info':>9s} {'thr':>5s} {'成功率':>8s} {'总回报':>8s} {'步数':>6s} {'误修':>6s} {'探针':>6s} {'耗时':>6s}")
    best = None
    for li, thr in itertools.product([float(x) for x in args.grid.split(",")],
                                     [float(x) for x in args.thrs.split(",")]):
        res = []
        for i, (root, mode) in enumerate(pairs):
            seed = 1000 + i
            world = OpsWorld(TRAIN, rng=random.Random(seed))
            agent = build_agent(TRAIN, seed=seed, n_sims=args.sims,
                                lambda_info=li, repair_threshold=thr)
            res.append(run_wm_episode(world, agent, root=root, mode=mode, seed=seed))
        ag = aggregate(res)
        print(f"{li:9.1f} {thr:5.2f} {ag['success']*100:7.1f}% {ag['reward']:8.1f} "
              f"{ag['steps']:6.2f} {ag['wrong']:6.2f} {ag['probe']:6.2f}")
        sys.stdout.flush()
        if best is None or ag["reward"] > best[0]:
            best = (ag["reward"], li, thr)
    print("best =", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
