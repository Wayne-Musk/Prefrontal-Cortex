"""wm.inference -- 信念空间推理：预测步、贝叶斯修正、信息增益。

这是整套架构的地基。关键点是三条恒等式之外的东西：

   传统 Agent:history → LLM → action           （不确定性隐含在 token 里）
   本架构  :P(s'|s,a) 预测 + P(o|s',a) 修正 → 后验 belief → 搜索
                                                 ↑ surprise = -log P(o)
``surprise`` 不是副产品，它是 *模型对自身无知的唯一诚实指标*，被 monitor 消费。
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from .types import Action, Belief, Fact, Json, Observation, signature


# ---------------------------------------------------------------- 预测步


def _propagate(posterior: Dict[str, float], states: Dict[str, Json],
               action: Action, model, rng: random.Random,
               K: int = 1) -> Tuple[Dict[str, float], Dict[str, Json]]:
    """稀疏转移核：b̄(s') = Σ_s T(s'|s, a) b(s)。K>1 时对后继做蒙特卡洛展开。"""
    new_w: Dict[str, float] = {}
    new_s: Dict[str, Json] = {}
    for sig, w in posterior.items():
        s = states[sig]
        share = w / K
        for _ in range(K):
            s2 = model.sample_transition(s, action, rng)
            sig2 = signature(s2)
            if sig2 not in new_s:
                new_s[sig2] = s2
            new_w[sig2] = new_w.get(sig2, 0.0) + share
    return new_w, new_s


def predict_step(belief: Belief, action: Action, model,
                 rng: random.Random, K: int = 1) -> Belief:
    """只做时间推进，不掺观测。纯想象，不产生任何环境副作用。"""
    post = belief.posterior()
    w, st = _propagate(post, belief.states, action, model, rng, K=K)
    return Belief(weights=w, states=st, facts=dict(belief.facts), t=belief.t + 1)


# ---------------------------------------------------------------- 修正步


def bayes_correct(prior: Belief, action: Action, obs: Observation,
                  model) -> Tuple[Belief, float]:
    """b(s') ∝ P(o | s', a) · b̄(s')，并返回 surprise = -log P(o | model)。

    surprise 是贝叶斯的证据下界型指标：
      - ≈0        → 观测完全在模型预期内
      - 突然变大  → 模型对世界的理解在这一带失效（blind spot）
    """
    if not prior.weights:
        return prior.copy(), 0.0
    weights: Dict[str, float] = {}
    for sig, pw in prior.weights.items():
        logp = model.emission_logprob(prior.states[sig], action, obs)
        if logp > 0.0:
            logp = 0.0
        if logp < -60.0:
            logp = -60.0          # 数值下溢保护
        weights[sig] = pw * math.exp(logp)
    evidence = sum(weights.values())
    surprise = -math.log(max(evidence, 1e-300))
    b = Belief(weights=weights, states=dict(prior.states),
               facts=dict(prior.facts), t=prior.t)
    return b, surprise


def belief_update(belief: Belief, action: Action, obs: Observation, model,
                  rng: random.Random, K: int = 1,
                  prune_min: float = 1e-4,
                  max_hyp: int = 48) -> Tuple[Belief, float]:
    """完整的 filter：predict → correct → 事实层合并 → 剪枝。返回 (新信念, surprise)。"""
    bbar = predict_step(belief, action, model, rng, K=K)
    b, surprise = bayes_correct(bbar, action, obs, model)
    b.renorm()
    b.set_facts(obs.facts, source="obs")
    b.t = belief.t + 1
    b.prune(prune_min, max_hyp)
    return b, surprise


# ---------------------------------------------------------------- 信息增益


def expected_information_gain(belief: Belief, action: Action, model,
                              rng: random.Random,
                              n_samples: int = 4, K: int = 1) -> float:
    """EPIG ≈ H(b) - E_{o ~ P(o|a,b)} [H(b')]。

    这一个量是「诊断性动作」存在的理由。纯 flight Do skill+roleplay 范式没有对应的
    机制：它无法为「什么都不做，只为看清楚」这类动作赋予价值，于是只能在没看清楚时
    就盲动手。
    """
    if belief.mass() <= 0:
        return 0.0
    bbar = predict_step(belief, action, model, rng, K=K)
    bbar.renorm()
    if bbar.mass() <= 0:
        return 0.0
    h0 = belief.entropy()
    acc = 0.0
    n = max(1, n_samples)
    for _ in range(n):
        sig = _sample_index(bbar.posterior(), rng)
        o = model.sample_emission(bbar.states[sig], action, rng)
        b2, _ = bayes_correct(bbar, action, o, model)
        b2.renorm()
        acc += b2.entropy()
    return max(0.0, h0 - acc / n)


def _sample_index(post: Dict[str, float], rng) -> str:
    r = rng.random() * sum(post.values())
    acc = 0.0
    for sig, p in post.items():
        acc += p
        if r <= acc:
            return sig
    return next(iter(post))


def marginal(belief: Belief, project) -> Dict[Hashable, float]:
    """把后验边缘化到任意特征上：project(state) -> hashable, 返回该特征的分布。"""
    out: Dict[Hashable, float] = {}
    for sig, p in belief.posterior().items():
        k = project(belief.states[sig])
        out[k] = out.get(k, 0.0) + p
    return out
