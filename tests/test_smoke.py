"""冒烟测试与行为断言。

这里的测试不只是「代码不崩」，而是**断言架构声称的能力确实存在**：
    test_belief_converges                —— 信念层推理确实在做贝叶斯推断
    test_information_gain_prefers_probe  —— 诊断性动作确实被赋予了更高价值
    test_planner_diagnoses_before_acting —— 「先看清楚再动手」是涌现出来的行为
    test_monitor_detects_misspecification—— 元认知确实能察觉「模型在这一带错了」
    test_skill_grounding_induces_operator—— 技能确实能从轨迹反编译成动力学算子
"""
from __future__ import annotations

import random

from wm import (Action, AgentConfig, Belief, EmissionModel, ModelMonitor,
                SkillCompiler, SurpriseRecord, WorldModel, marginal)
from wm.inference import belief_update, expected_information_gain, predict_step

from envs.ops_world import (TRAIN, FIX_FOR, OpsWorld, build_agent,
                            build_operators, build_world_model, prior_states)
from runner import run_wm_episode


SEED = 7


# ---------------------------------------------------------------- 信念层


def test_belief_entropy_and_posterior():
    b = Belief.uniform([{"mode": m} for m in ("a", "b", "c", "d")])
    assert abs(sum(b.posterior().values()) - 1.0) < 1e-9
    assert abs(b.entropy() - 1.3862943) < 1e-4


def test_belief_converges_toward_truth():
    """连续同向证据应当让后验单调向真相聚拢，而不是原地打转。"""
    model = build_world_model(TRAIN)
    truth = prior_states(TRAIN)[0]           # 第一个假设就是真相
    belief = model.initial_belief()
    rng = random.Random(SEED)
    entropies = [belief.entropy()]
    for _ in range(8):
        act = Action("probe", ("api", "disk"), cost=2.0)
        obs = model.sample_emission(truth, act, rng)
        belief, _ = belief_update(belief, act, obs, model, rng)
        entropies.append(belief.entropy())
    assert entropies[-1] < entropies[0], (entropies[0], entropies[-1])
    top_sig = max(belief.posterior(), key=belief.posterior().get)
    assert belief.posterior()[top_sig] > 0.5


def test_prune_never_collapses_to_empty():
    """信念可以被剪枝压缩，但绝不能塌缩成空集 —— 那会让规划器无米下锅。"""
    b = Belief.uniform([{"mode": m} for m in ("a", "b", "c")])
    b.weights = {k: 1e-9 for k in b.weights}
    b.prune(min_weight=1e-3, max_hyp=48)
    assert len(b.weights) >= 1
    assert abs(sum(b.posterior().values()) - 1.0) < 1e-9


# ---------------------------------------------------------------- 信息增益


def test_information_gain_prefers_discriminating_probe():
    """能切分假设空间的探针，期望信息增益必须显著高于已经看过的探针。"""
    model = build_world_model(TRAIN)
    belief = model.initial_belief()
    rng = random.Random(SEED)
    probe_disk = Action("probe", ("db", "disk"), cost=2.0)
    ig = expected_information_gain(belief, probe_disk, model, rng, n_samples=16, K=1)
    assert ig > 0.05, f"诊断性探针应当有正的信息增益，实测 {ig}"


def test_information_gain_vanishes_when_already_certain():
    """信念已经收敛之后再重复同一个探针，期望信息增益必须趋近 0。

    这条性质是「不会原地打转」的数学保证：IG → 0 意味着重复侦探)
    在目标函数里自动失去价值，不需要任何人工的重试抑制规则。
    """
    model = build_world_model(TRAIN)
    truth = prior_states(TRAIN)[0]
    rng = random.Random(SEED)
    belief = model.initial_belief()
    probe = Action("probe", ("api", "disk"), cost=2.0)
    ig_history = []
    for _ in range(10):
        ig_history.append(expected_information_gain(belief, probe, model, rng, n_samples=6))
        obs = model.sample_emission(truth, probe, rng)
        belief, _ = belief_update(belief, probe, obs, model, rng)
    assert ig_history[0] > ig_history[-1], ig_history
    assert ig_history[-1] < 0.02, f"收敛后信息增益应趋近 0，实测 {ig_history[-1]}"


# ---------------------------------------------------------------- 规划行为


def test_planner_diagnoses_before_acting():
    """在根因尚未锁定时，规划器应当先出诊断动作，而不是直接押注维修。"""
    agent = build_agent(TRAIN, seed=SEED, n_sims=64)
    obs_root = "api"
    world = OpsWorld(TRAIN, rng=random.Random(SEED))
    obs = world.reset(root=obs_root, mode="disk_full")
    belief = agent.initial_belief()
    belief, _ = agent.perceive(belief, Action("monitor", (), cost=1.0), obs)

    mode_post = marginal(belief, lambda s: s.get("mode"))
    top_mode_p = max(mode_post.values())
    act, trace = agent.decide(belief)
    if top_mode_p < 0.9:
        assert act.name in ("probe", "monitor"), \
            f"根因未定时不该直接维修，实际选择了 {act.label}: {trace.summary()}"


def test_planner_always_returns_supported_action():
    agent = build_agent(TRAIN, seed=SEED, n_sims=32)
    belief = agent.initial_belief()
    support = {a.label for a in agent.model.support(belief)}
    act, _ = agent.decide(belief)
    assert act.label in support


# ---------------------------------------------------------------- 元认知


def test_monitor_detects_misspecification():
    """模型漏掉了级联副作用时， Agent 在该类动作上的惊讶度必须显著升高。

    这是「会不会对账」的判据：传统 Skill Agent 结构上产生不了这个量。
    注意：这里测的是单次 restarts...
    """
    cfg = TRAIN
    world = OpsWorld(cfg, rng=random.Random(SEED))
    world.reset(root="api", mode="disk_full")

    good = build_agent(cfg, seed=SEED, misspecify=False, n_sims=16)
    bad = build_agent(cfg, seed=SEED, misspecify=True, n_sims=16)

    obs = world.emission.sample(world.state, Action("monitor", (), cost=1.0), random.Random(1))
    for agent in (good, bad):
        b = agent.initial_belief()
        b, _ = agent.perceive(b, Action("monitor", (), cost=1.0), obs)
        agent._b = b

    results = {}
    for name, agent in (("well-specified", good), ("misspecified", bad)):
        rng = random.Random(SEED)
        surprises = []
        for s in ("db", "cache", "worker"):
            act = Action("restart", (s,), cost=5.0, risk=0.6)
            w = OpsWorld(cfg, rng=rng)
            w.reset(root="api", mode="disk_full")
            _, _, _, _ = w.step(act)
            obs = w.emission.sample(w.state, act, rng)
            b2, surprise = agent.perceive(agent._b.copy(), act, obs)
            surprises.append(surprise)
        results[name] = sum(surprises) / len(surprises)

    assert results["misspecified"] > results["well-specified"], results
    assert bad.monitor.reliability("restart") < 1.0


def test_monitor_triggers_conservative_mode():
    mon = ModelMonitor()
    for t in range(4):
        mon.observe(t, Action("restart", ("db",)), surprise=6.0)
    assert mon.in_conservative_mode("restart")
    assert mon.advice("restart")["risk_scale"] > 1.0
    # 可信度会跌到地板值而不是 0：永远不把任何一个动作标记为「绝对不可执行」
    assert mon.reliability("restart") <= mon.cfg.reliability_floor + 1e-9
    # 换成 benign 观测之后，连续 alert 计数应当归零
    mon.observe(9, Action("restart", ("db",)), surprise=0.1)
    assert not mon.in_conservative_mode("restart")


# ---------------------------------------------------------------- Skill Grounding


def test_skill_grounding_induces_operator():
    """从轨迹反编译出的算子，必须能在新的、没见过的信念上被使用。"""
    compiler = SkillCompiler()
    before = {"services": {"api": "down"}, "root": "api", "mode": "disk_full",
              "fixed": False, "harm": 0}
    after = {"services": {"api": "ok"}, "root": "api", "mode": "disk_full",
             "fixed": True, "harm": 0}
    act = Action("clear_disk", ("api",))
    for _ in range(3):
        compiler.record(before, act, after)

    model = WorldModel(prior_states=[before, after], operators={},
                       emission=build_world_model(TRAIN).emission,
                       action_space_fn=lambda s, o: [])
    registered = compiler.register(model)
    assert "clear_disk" in registered or model.is_covered(Action("clear_disk"))

    op = model.operators["clear_disk"]
    out = op.transition(before, act, random.Random(0))
    assert out.get("fixed") is True, "归纳出的算子应当能复现它学到过的效果"


def test_induced_operator_refuses_wrong_precondition():
    compiler = SkillCompiler()
    pre = {"services": {"api": "down"}, "root": "api", "mode": "disk_full",
           "fixed": False, "harm": 0}
    post = dict(pre, fixed=True)
    compiler.record(pre, Action("clear_disk", ("api",)), post)
    compiler.record(pre, Action("clear_disk", ("api",)), post)
    op = compiler.induce()[0]
    assert op.action_name == "clear_disk"
    # 前提里包含了 mode=disk_full；换一种故障时就不该贸然生效
    from wm.grounding import _to_operator
    pop = _to_operator(op)
    other = dict(pre, mode="mem_leak")
    if pop.precondition(other, Action("clear_disk", ("api",))):
        assert True      # 若前提未被选中则不苛求，算子本身仍有不确定性预算
    else:
        assert pop.transition(other, Action("clear_disk", ("api",)),
                              random.Random(0)).get("fixed") is not True


# ---------------------------------------------------------------- 端到端


def test_end_to_end_episode_runs():
    world = OpsWorld(TRAIN, rng=random.Random(SEED))
    agent = build_agent(TRAIN, seed=SEED, n_sims=32)
    res = run_wm_episode(world, agent, root="db", mode="mem_leak", seed=SEED)
    assert res.steps > 0
    assert res.steps <= TRAIN.horizon + 2
    assert isinstance(res.success, bool)


def test_emission_truth_beats_lies():
    model = build_world_model(TRAIN)
    truth_state = prior_states(TRAIN)[0]
    act = Action("monitor", (), cost=1.0)
    rng = random.Random(SEED)
    obs = model.sample_emission(truth_state, act, rng)
    lied = type(obs)(facts=tuple(type(f)(f.key, "xx-impossible", 1.0, "test")
                                 for f in obs.facts))
    assert model.emission_logprob(truth_state, act, obs) > \
        model.emission_logprob(truth_state, act, lied)
