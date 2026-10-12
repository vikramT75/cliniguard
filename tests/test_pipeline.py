import asyncio

from guardrail_proxy.config import Action, FailMode, GuardPolicy
from guardrail_proxy.guards.base import Guard, GuardContext, GuardVerdict, Stage
from guardrail_proxy.pipeline import GuardPipeline


class StubGuard(Guard):
    stage = Stage.INPUT

    def __init__(self, name, score=0.0, delay=0.0, error=None, **policy):
        super().__init__(GuardPolicy(**policy))
        self.name = name
        self.score, self.delay, self.error = score, delay, error
        self.finished = False

    async def check(self, ctx):
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        self.finished = True
        return GuardVerdict(
            score=self.score,
            reason="stub",
            messages=[{"role": "user", "content": f"rewritten by {self.name}"}],
        )


def ctx():
    return GuardContext(request_id="t", messages=[{"role": "user", "content": "hi"}])


async def test_block_short_circuits_slow_guards():
    slow = StubGuard("slow", delay=5)
    fast_block = StubGuard("blocker", score=0.9, action=Action.BLOCK)
    outcome = await GuardPipeline([slow, fast_block], []).run_input(ctx())
    assert outcome.blocked_by.guard == "blocker"
    assert not slow.finished
    assert [r.guard for r in outcome.results] == ["slow", "blocker"]  # policy order


async def test_timeout_fail_closed_blocks():
    g = StubGuard("phi", delay=1, timeout_s=0.05, fail_mode=FailMode.CLOSED)
    outcome = await GuardPipeline([g], []).run_input(ctx())
    assert outcome.blocked_by is not None
    assert outcome.results[0].error == "timeout"


async def test_error_fail_open_flags_and_continues():
    g = StubGuard("judge", error=RuntimeError("judge down"), fail_mode=FailMode.OPEN)
    outcome = await GuardPipeline([g], []).run_input(ctx())
    assert outcome.blocked_by is None
    assert outcome.results[0].action is Action.FLAG
    assert "judge down" in outcome.results[0].error


async def test_redact_rewrite_applied_only_when_triggered():
    quiet = StubGuard("quiet", score=0.1, action=Action.REDACT)
    loud = StubGuard("loud", score=0.9, action=Action.REDACT)
    outcome = await GuardPipeline([quiet, loud], []).run_input(ctx())
    assert outcome.messages == [{"role": "user", "content": "rewritten by loud"}]


async def test_guards_run_concurrently():
    guards = [StubGuard(f"g{i}", delay=0.2) for i in range(4)]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await GuardPipeline(guards, []).run_input(ctx())
    assert loop.time() - t0 < 0.5
