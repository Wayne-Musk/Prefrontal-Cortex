"""wm -- Agent-Level World Model 核心库。

分层：

    types      信念 / 动作 / 事实 / 追踪记录的数据结构
    inference  信念空间推理：predict → correct → 惊讶度 → 信息增益
    model      生成式世界模型：T(s'|s,a) + P(o|s',a)
    value      目标与约束的势函数
    proposal   候选提议（技能在这里降级为生成器）
    planner    信念空间上的 Bayes-adaptive MCTS
    monitor    元认知：预测误差监控 → 保守度 / 算子可信度
    grounding  Skill Grounding：技能 → 动力学算子
    agent      闭环主循环
"""
from .types import Action, Belief, DecisionTrace, Fact, Json, Observation, SurpriseRecord
from .model import EmissionModel, Operator, WorldModel, WorldModelConfig
from .value import ValueConfig, ValueModel
from .proposal import HeuristicProposal, LLMProposal, Proposal, ProposalConfig
from .planner import MCTSPlanner, PlannerConfig
from .monitor import ModelMonitor, MonitorConfig
from .grounding import GroundingConfig, SkillCompiler, TransitionSample
from .agent import AgentConfig, WMAgent
from .inference import belief_update, expected_information_gain, marginal, predict_step

__all__ = [
    "Action", "Belief", "Fact", "Observation", "Json", "DecisionTrace", "SurpriseRecord",
    "EmissionModel", "Operator", "WorldModel", "WorldModelConfig",
    "ValueConfig", "ValueModel",
    "Proposal", "ProposalConfig", "HeuristicProposal", "LLMProposal",
    "MCTSPlanner", "PlannerConfig",
    "ModelMonitor", "MonitorConfig",
    "SkillCompiler", "GroundingConfig", "TransitionSample",
    "WMAgent", "AgentConfig",
    "belief_update", "predict_step", "expected_information_gain", "marginal",
]

__version__ = "0.1.0"
