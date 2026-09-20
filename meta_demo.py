"""元认知专项实验：当世界出现模型从未假设过的东西时会发生什么。

这一场比的不是「谁修得好」（双方都修不好），而是：

    A. 系统能不能发出「我这里不理解」的信号？
    B. 在完全没把握时，它会不会转而选择低成本、可逆的观察动作，而不是硬着头皮乱修？

对照组的答案在结构上是「不能」——它没有可被推翻的预测，因此永远不会有 surprise。
"""
from __future__ import annotations

import argparse
import random
from statistics import mean
from typing import Dict, List

from baselines import SkillRoleplayAgent
from envs.ops_world import OOD, TRAIN, OpsWorld, build_agent
from runner import run_baseline_episode, run_wm_episode

UNSEEN = ("network_partition", "cert_expired")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=48)
    args = ap.parse_args()

    cases = [(s, m) for s in OOD.services for m in UNSEEN]
    print(f"\n【场景】OOD 拓扑（{len(OOD.services)} 服务），故障类型为模型先验里不存在的 "
          f"{UNSEEN}，共 {len(cases)} 例\n")

    wm_rows, base_rows = [], []
    conservative_flags = 0
    surprise_all: List[float] = []

    for i, (root, mode) in enumerate(cases):
        seed = 2000 + i
        # --- World Model Agent：先验假设空间被显式收窄 ---
        world = OpsWorld(OOD, rng=random.Random(seed))
        agent = build_agent(OOD, seed=seed, n_sims=args.sims, prior_modes=TRAIN.modes)
        r = run_wm_episode(world, agent, root=root, mode=mode, seed=seed)
        wm_rows.append(r)
        surprise_all.append(r.mean_surprise)
        if agent.monitor.in_conservative_mode():
            conservative_flags += 1

        # --- 对照组：技能库里也没有这类故障（与 WM 的先验收窄对等） ---
        world2 = OpsWorld(OOD, rng=random.Random(seed))
        b = SkillRoleplayAgent(OOD, seed=seed, skill_modes=TRAIN.modes)
        base_rows.append(run_baseline_episode(world2, b, root=root, mode=mode))

    def agg(rows, key):
        return mean([getattr(r, key) for r in rows])

    print(f"{'指标':<24s}{'World-Model Agent':>20s}{'skill+roleplay':>20s}")
    print("-" * 64)
    rows = [
        ("修复成功率", agg(wm_rows, "success"), agg(base_rows, "success")),
        ("平均误修次数", agg(wm_rows, "n_wrong"), agg(base_rows, "n_wrong")),
        ("平均连带损伤", agg(wm_rows, "n_harmful"), agg(base_rows, "n_harmful")),
        ("平均诊断动作数", agg(wm_rows, "n_diagnostic"), agg(base_rows, "n_diagnostic")),
        ("平均总回报", agg(wm_rows, "total_reward"), agg(base_rows, "total_reward")),
    ]
    for name, w, b in rows:
        print(f"{name:<24s}{w:>20.2f}{b:>20.2f}")
    print("-" * 64)
    print(f"{'平均 surprise (nats)':<24s}{mean(surprise_all):>20.2f}{'无此量':>20s}")
    print(f"{'触发保守模式的回合数':<24s}{conservative_flags}/{len(cases)}{'无此机制':>20s}")

    print("\n结论：")
    print("  双方都修不好 —— 这符合预期，世界模型不会凭空发明它没见过的疗法。")
    print("  差别在于 World Model Agent 全程带着一个可读的不确定性指标（surprise），")
    print("  并据此降低对该类动作的信任；对照组在同样的处境下只会继续按技能的固定顺序")
    print("  一次次执行无效且带副作用的修复，且对外表现与胸有成竹时毫无区别。")

    # 抽一个例子做逐帧展示
    print("\n【样本回合逐帧】Markov 信念熵 H 与 surprise：")
    w = OpsWorld(OOD, rng=random.Random(999))
    a = build_agent(OOD, seed=999, n_sims=args.sims, prior_modes=TRAIN.modes)
    r = run_wm_episode(w, a, root=OOD.services[0], mode=UNSEEN[0], seed=999, verbose=True)
    for line in r.trace[:12]:
        print("   " + line)
    print(f"\n   该回合是否进入保守模式：{a.monitor.in_conservative_mode()}")
    print("   模型不可靠区域（按 surprise 排序）：")
    for name, s, streak in a.monitor.blind_spots(4):
        print(f"     {name:<12s} surprise_ema={s:5.2f}  reliability={a.monitor.reliability(name):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
