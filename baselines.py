"""baselines -- 对照组：skill + role-playing 范式。

对照组的选取原则：**不能搭稻草人**。
如果一个柿子捏软了的 baseline 赢了，说明不了任何事。所以这里实现的是

    A. SkillRoleplayAgent       —— 只装备「见过的故障」对应的技能（真实部署情形）
    B. SkillRoleplayAgentOracle —— 装备**全部**技能，包括 OOD 新故障的正确修复动作

B 是技能范式的**上界**：给了它完美的技能库，它缺的只剩「信念」。
如果 B 依然输，那么输的原因就确凿地落在范式本身（无信念、无前瞻、无预测误差修正），
而不是「技能没写全」。

两者的共同结构，也是传统范式的三个结构性缺陷：
    1. 无潜变量信念：症状直接映射到动作假设，选错就重试
    2. 无前瞻：depth-1 贪心，不会为了「看清楚」而付出一步代价
    3. 无对账：不会把「我猜错了」这件事沉淀成对自身模型的修正
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from wm.types import Action, Observation

from envs.ops_world import (DETECT, FIX_FOR, PROBES, REPAIRS, OpsWorldConfig,
                            OpsWorld, build_emission)

# 猜测顺序：按「常见故障」的先验频次排（资深工程师的技能顺序大致就是这样）
MODE_PRIORITY = ("mem_leak", "bad_config", "disk_full", "bug_crash",
                 "network_partition", "cert_expired")

# 探针 → 能确认哪种模式
PROBE_OF_MODE = {v: k for k, v in DETECT.items()}


@dataclass
class EpisodeMemory:
    """对照组也有「记忆」，只是它是不可查询的线性日志，不是可推断的信念状态。"""
    tried: List[Tuple[str, str]] = field(default_factory=list)     # (service, action)
    observations: List[str] = field(default_factory=list)
    n_steps: int = 0


class SkillRoleplayAgent:
    """传统 skill + role-playing agent。"""

    name = "skill+roleplay"

    def __init__(self, cfg: OpsWorldConfig, seed: int = 0,
                 full_skills: bool = False,
                 skill_modes: Optional[Tuple[str, ...]] = None,
                 probe_before_repair: int = 1,
                 avoid_restart: bool = True):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.full_skills = full_skills
        self.probe_before_repair = probe_before_repair
        self.avoid_restart = avoid_restart
        self.memory = EpisodeMemory()
        # 角色设定 —— 在传统范式里它只能作为行为先验，不能作为世界模型被查询
        self.role = "你是一名资深 SRE，遇到告警先定位根因再动手。"
        # 技能槽：symptom → repair skill。skill_modes 用来限定「见过哪些故障」
        if skill_modes is not None:
            modes = tuple(skill_modes)
        else:
            modes = tuple(FIX_FOR.keys()) if full_skills else tuple(cfg.modes)
        self.skills = {m: FIX_FOR[m][0] for m in modes if m in FIX_FOR}
        self.decisions: List[str] = []

    # -- 每回合开始重置（但注意：它不会积累跨回合的信念，只清掉线性日志）
    def reset(self) -> None:
        self.memory = EpisodeMemory()
        self.decisions = []

    # -- 从日志里提炼「症状」，这是它能做的全部推理
    def _suspects(self, obs: Observation) -> List[str]:
        down, degraded, ok = [], [], []
        for f in obs.facts:
            key = f.key
            if not key.startswith("lat@"):
                continue
            svc = key.split("@", 1)[1]
            {"down": down, "degraded": degraded}.get(f.value, ok).append(svc)
        return down + degraded + ok

    def _probe_results(self) -> Dict[str, str]:
        """日志里的探针读数。对照组能「读到」它们，但无法把它们转成后验。"""
        out: Dict[str, str] = {}
        for line in self.memory.observations:
            for token in line.split(","):
                if "=" in token and not token.startswith("lat@"):
                    k, v = token.split("=", 1)
                    out[k] = v
        return out

    def decide(self, obs: Observation) -> Action:
        self.memory.n_steps += 1
        self.memory.observations.append(
            ",".join(f"{f.key}={f.value}" for f in obs.facts))

        suspects = self._suspects(obs)
        if not suspects:
            suspects = list(self.cfg.services)

        # 技能触发:N步之内先按技能做「基础巡检」
        if self.memory.n_steps <= self.probe_before_repair:
            return Action("monitor", (), cost=1.0)

        # 选定一个疑似根因服务 + 一种故障猜测 → 直接动手。
        # 注意这里缺的那一步：没有任何「先花一步把可能性压到足够窄再动手」的机制。
        for svc in suspects:
            for mode in MODE_PRIORITY:
                if mode not in self.skills:
                    continue
                repair = self.skills[mode]
                if repair == "restart" and self.avoid_restart:
                    repair = "scale_up" if "mem_leak" in self.skills else repair
                if (svc, repair) in self.memory.tried:
                    continue
                self.memory.tried.append((svc, repair))
                cost, risk = REPAIRS[repair]
                self.decisions.append(f"skill[{mode}]→{repair}({svc})")
                return Action(repair, (svc,), cost=cost, risk=risk)

        # 全都试过了 → 兜底
        svc = self.rng.choice(suspects)
        self.decisions.append(f"fallback→restart({svc})")
        cost, risk = REPAIRS["restart"]
        return Action("restart", (svc,), cost=cost, risk=risk)


class SkillRoleplayAgentOracle(SkillRoleplayAgent):
    """拥有完整技能库的对照组上界。"""

    name = "skill+roleplay(oracle"

    def __init__(self, cfg: OpsWorldConfig, seed: int = 0):
        super().__init__(cfg, seed=seed, full_skills=True)


class RandomSkillAgent:
    """下界：随机执行技能。用来确认环境本身不是平凡的。"""

    name = "random"

    def __init__(self, cfg: OpsWorldConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)

    def reset(self) -> None:
        return None

    def decide(self, obs: Observation) -> Action:
        svc = self.rng.choice(list(self.cfg.services))
        if self.rng.random() < 0.5:
            return Action("monitor", (), cost=1.0)
        if self.rng.random() < 0.3:
            return Action("probe", (svc, self.rng.choice(PROBES)), cost=2.0)
        name = self.rng.choice(list(REPAIRS.keys()))
        c, r = REPAIRS[name]
        return Action(name, (svc,), cost=c, risk=r)
