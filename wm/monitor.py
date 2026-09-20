"""wm.monitor -- 元认知：预测误差监控与模型自我修正。

**为什么这是必备模块而不是锦上添花。**

一个纯 LLM 的世界模型会「幻觉出一个自洽但错误的世界」：它对任意观测都能事后解释。
唯一能对抗这件事的机制，是在行动之前先把预测写下来，然后跟现实对账。

    surprise = -log P(o | model, belief, action)

这是贝叶斯意义上的惊讶度。它有三个消费者：

    1. 信念层的贝叶斯修正（inference 模块，正常运行时用）
    2. **模型层**：同一类动作反复高 surprise ⇒ 这一带的动力学是错的 ⇒ 下调该算子可信度
    3. **策略层**：进入保守模式 ⇒ planner 放大副作用惩罚、提高信息增益权重

第 2、3 条是传统 skill + roleplay 范式完全缺失的能力：它会把同一个错的修复套路
重试到底，因为它没有「我的世界模型在这里不成立」这个概念。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Action, Belief, SurpriseRecord


@dataclass
class MonitorConfig:
    alpha: float = 0.35                  # surprise EMA 的学习率
    surprise_threshold: float = 2.0      # nats；超过即视为一次显著预测失败
    alert_streak: int = 2                # 连续多少次进入保守模式
    reliability_floor: float = 0.20
    conservative_risk_scale: float = 2.0
    conservative_info_bonus: float = 1.5
    max_records: int = 4000


class ModelMonitor:
    """模型的自我审视模块。"""

    def __init__(self, cfg: Optional[MonitorConfig] = None):
        self.cfg = cfg or MonitorConfig()
        self.records: List[SurpriseRecord] = []
        self._ema: Dict[str, float] = {}          # action_name → surprise EMA
        self._streak: Dict[str, int] = {}
        self.total_surprise = 0.0
        self.n_updates = 0

    # ------------------------------------------------------------ 记录
    def observe(self, t: int, action: Action, surprise: float,
                belief_before: Optional[Belief] = None,
                belief_after: Optional[Belief] = None,
                note: str = "") -> SurpriseRecord:
        rec = SurpriseRecord(
            t=t,
            action_label=action.label,
            surprise=surprise,
            entropy_before=belief_before.entropy() if belief_before else 0.0,
            entropy_after=belief_after.entropy() if belief_after else 0.0,
            top_before=[(str(s.get("mode")), p) for _, p, s in (belief_before.top(2) if belief_before else [])],
            note=note,
        )
        self.records.append(rec)
        if len(self.records) > self.cfg.max_records:
            self.records = self.records[-self.cfg.max_records:]

        a = self.cfg.alpha
        prev = self._ema.get(action.name, 0.0)
        self._ema[action.name] = (1 - a) * prev + a * surprise
        hit = surprise > self.cfg.surprise_threshold
        self._streak[action.name] = (self._streak.get(action.name, 0) + 1) if hit else 0
        self.total_surprise += surprise
        self.n_updates += 1
        return rec

    # ------------------------------------------------------------ 查询
    def surprise_ema(self, action_name: str) -> float:
        return self._ema.get(action_name, 0.0)

    def reliability(self, action_name: str) -> float:
        """把 EMA 惊讶度压缩成 0~1 的可信度。供 planner 给旧动作打折扣。"""
        r = math.exp(-self.surprise_ema(action_name))
        return max(self.cfg.reliability_floor, r)

    def mean_surprise(self) -> float:
        return self.total_surprise / self.n_updates if self.n_updates else 0.0

    def in_conservative_mode(self, action_name: Optional[str] = None) -> bool:
        keys = [action_name] if action_name else list(self._streak.keys())
        return any(self._streak.get(k, 0) >= self.cfg.alert_streak for k in keys)

    def advice(self, action_name: Optional[str] = None) -> Dict[str, float]:
        """给 planner 的策略建议。这是「元认知影响行动」的通道。"""
        cons = self.in_conservative_mode(action_name)
        return {
            "risk_scale": self.cfg.conservative_risk_scale if cons else 1.0,
            "info_bonus": self.cfg.conservative_info_bonus if cons else 1.0,
        }

    def blind_spots(self, top: int = 5) -> List[Tuple[str, float, int]]:
        """模型最不可靠的区域。这直接指明「下一步该去学什么 / 该去问谁」。"""
        rows = [(k, v, self._streak.get(k, 0)) for k, v in self._ema.items()]
        return sorted(rows, key=lambda kv: -kv[1])[:top]

    # ------------------------------------------------------------ 写回
    def apply_to_model(self, model) -> int:
        """把可信度写回算子的 err_ema，闭环到 planner 的 step_reward。"""
        n = 0
        for name, op in model.operators.items():
            if name in self._ema:
                op.err_ema = -math.log(max(self.reliability(name), 1e-6))
                op.n_samples = max(op.n_samples, 1)
                n += 1
        return n

    def report(self) -> str:
        if not self.records:
            return "monitor: 暂无记录"
        lines = [f"均值 surprise={self.mean_surprise():.2f} nats, 记录数={len(self.records)}"]
        for name, s, streak in self.blind_spots(3):
            flag = " [保守]" if streak >= self.cfg.alert_streak else ""
            lines.append(f"  {name}: surprise_ema={s:.2f} reliability={self.reliability(name):.2f}{flag}")
        return "\n".join(lines)
