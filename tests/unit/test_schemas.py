"""Validation stricte des entrées client."""

import pytest
from pydantic import ValidationError

from scoring_api.api.schemas import EXAMPLE_CLIENT, ClientInput


def _errors(payload: dict) -> list[dict]:  # type: ignore[type-arg]
    with pytest.raises(ValidationError) as exc:
        ClientInput.model_validate(payload)
    return exc.value.errors()


def test_example_is_valid() -> None:
    ClientInput.model_validate(EXAMPLE_CLIENT)


@pytest.mark.parametrize("field", ["AMT_CREDIT", "DAYS_BIRTH", "FLAG_DOCUMENT_3"])
def test_missing_required_field(field: str) -> None:
    payload = {k: v for k, v in EXAMPLE_CLIENT.items() if k != field}
    errs = _errors(payload)
    assert errs[0]["loc"] == (field,) and errs[0]["type"] == "missing"


@pytest.mark.parametrize(
    ("field", "value", "err_type"),
    [
        ("DAYS_BIRTH", 1826, "less_than_equal"),  # âge de -5 ans
        ("DAYS_BIRTH", -60_000, "greater_than_equal"),  # 164 ans
        ("AMT_CREDIT", 0, "greater_than"),  # crédit nul
        ("AMT_ANNUITY", 0.0, "greater_than"),
        ("EXT_SOURCE_2", 1.5, "less_than_equal"),
        ("EXT_SOURCE_2", -0.1, "greater_than_equal"),
        ("OWN_CAR_AGE", 150.0, "less_than_equal"),
        ("DAYS_ID_PUBLISH", 10, "less_than_equal"),
        ("PREV_COUNT", -1.0, "greater_than_equal"),
    ],
)
def test_out_of_range(field: str, value: object, err_type: str) -> None:
    errs = _errors({**EXAMPLE_CLIENT, field: value})
    assert errs[0]["loc"] == (field,) and errs[0]["type"] == err_type


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AMT_CREDIT", "abc"),
        ("AMT_CREDIT", "30000"),  # chaîne numérique refusée en mode strict
        ("DAYS_BIRTH", -14000.5),
        ("DAYS_BIRTH", "-14000"),
        ("FLAG_DOCUMENT_3", 2),
        ("FLAG_DOCUMENT_3", "1"),
        ("EXT_SOURCE_1", [0.5]),
    ],
)
def test_wrong_types(field: str, value: object) -> None:
    errs = _errors({**EXAMPLE_CLIENT, field: value})
    assert errs[0]["loc"] == (field,)


def test_unknown_field_rejected() -> None:
    errs = _errors({**EXAMPLE_CLIENT, "AMT_INCOME_TOTAL": 100_000.0})
    assert errs[0]["type"] == "extra_forbidden"


def test_null_allowed_for_optional_only() -> None:
    ClientInput.model_validate({**EXAMPLE_CLIENT, "EXT_SOURCE_2": None})
    errs = _errors({**EXAMPLE_CLIENT, "AMT_CREDIT": None})
    assert errs[0]["loc"] == ("AMT_CREDIT",)


def test_int_accepted_for_float_fields() -> None:
    m = ClientInput.model_validate({**EXAMPLE_CLIENT, "AMT_CREDIT": 500000})
    assert m.AMT_CREDIT == 500000.0


def test_cross_field_coherence() -> None:
    errs = _errors({**EXAMPLE_CLIENT, "PREV_COUNT": 2.0, "PREV_REFUSED_COUNT": 3.0})
    assert "PREV_REFUSED_COUNT" in errs[0]["msg"]
    errs = _errors(
        {**EXAMPLE_CLIENT, "BUREAU_DAYS_CREDIT_MIN": -100.0, "BUREAU_DAYS_CREDIT_MAX": -900.0}
    )
    assert "BUREAU_DAYS_CREDIT_MIN" in errs[0]["msg"]


def test_bool_rejected_for_flag() -> None:
    errs = _errors({**EXAMPLE_CLIENT, "FLAG_EMP_PHONE": True})
    assert errs[0]["loc"] == ("FLAG_EMP_PHONE",)
