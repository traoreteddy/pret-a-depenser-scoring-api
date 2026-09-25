import uuid

from scoring_api.storage.sampling import deterministic_unit, should_sample


def test_deterministic() -> None:
    rid = str(uuid.uuid4())
    assert deterministic_unit(rid) == deterministic_unit(rid)
    assert 0 <= deterministic_unit(rid) < 1


def test_rate_is_respected_on_average() -> None:
    ids = [str(uuid.uuid4()) for _ in range(20_000)]
    kept = sum(should_sample(i, 0.2, threshold=0.48, rate=0.25, grey_zone=0.0) for i in ids)
    assert 0.23 < kept / len(ids) < 0.27


def test_rate_zero_and_one() -> None:
    rid = str(uuid.uuid4())
    assert not should_sample(rid, 0.1, threshold=0.48, rate=0.0, grey_zone=0.0)
    assert should_sample(rid, 0.1, threshold=0.48, rate=1.0, grey_zone=0.0)


def test_grey_zone_forces_sampling() -> None:
    rid = "fixed-id"
    assert should_sample(rid, 0.47, threshold=0.48, rate=0.0, grey_zone=0.05)
    assert not should_sample(rid, 0.10, threshold=0.48, rate=0.0, grey_zone=0.05)


def test_forced() -> None:
    assert should_sample("x", None, threshold=0.48, rate=0.0, grey_zone=0.0, forced=True)
