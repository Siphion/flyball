"""Model profiles: matched by model id, read from JSON, one operating point per level."""

from __future__ import annotations

import json

import pytest

from flyball.governor import GovernorConfig
from flyball.profiles import (
    BONSAI_2,
    LEGACY,
    QWEN_THINKING,
    all_profiles,
    load_profiles,
    resolve,
)

BONSAI_PATH = "F:\\ai\\models\\bonsai2-27b\\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"


@pytest.mark.parametrize(
    "names, expected",
    [
        ([BONSAI_PATH], BONSAI_2),
        (["bonsai2-27b"], BONSAI_2),
        (["", "Qwen3-30B-A3B-Thinking"], QWEN_THINKING),
        (["gpt-oss-20b"], LEGACY),
        ([], LEGACY),
    ],
)
def test_profiles_are_matched_on_the_model_id(names, expected) -> None:
    assert resolve(names, all_profiles()) is expected


def test_a_named_profile_wins_over_matching() -> None:
    assert resolve(["Qwen3-8B"], all_profiles(), forced="bonsai-2") is BONSAI_2
    assert resolve(["Qwen3-8B"], all_profiles(), forced="no-such-profile") is LEGACY


@pytest.mark.parametrize(
    "body, default, level",
    [
        ({}, "", "xhigh"),  # the template's own default
        ({}, "medium", "medium"),  # a server started with its own default
        ({"reasoning_effort": "low"}, "medium", "low"),
        ({"reasoning_effort": "none"}, "", "none"),
        ({"chat_template_kwargs": {"enable_thinking": False}}, "medium", "none"),
        ({"reasoning_effort": "high"}, "", "high"),  # unknown to the profile: kept by name
    ],
)
def test_the_level_a_request_asks_for(body, default, level) -> None:
    assert BONSAI_2.requested_level(body, default) == level


def test_operating_points_are_per_level() -> None:
    assert BONSAI_2.operating_point("xhigh") is not None
    assert BONSAI_2.operating_point("medium") is not None
    assert BONSAI_2.operating_point("low") is None  # never measured: not governed
    assert LEGACY.operating_point("anything") == GovernorConfig()


def test_a_json_profile_extends_a_built_in_and_replaces_it(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": [{
        "id": "bonsai-2",
        "extends": "bonsai-2",
        "governor": {"medium": {"first_probe": 1200, "cap": None}, "xhigh": None},
    }]}), encoding="utf-8")
    profiles = all_profiles(load_profiles(path))
    bonsai = resolve([BONSAI_PATH], profiles)
    assert bonsai is not BONSAI_2 and bonsai.levels == BONSAI_2.levels
    assert bonsai.operating_point("medium") == GovernorConfig(first_probe=1200, cap=None)
    assert bonsai.operating_point("xhigh") is None


def test_a_new_profile_from_json(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps([{
        "id": "gpt-oss",
        "match": ["*gpt-oss*"],
        "levels": [{"name": "low", "effort": "low"}, {"name": "high", "effort": "high"}],
        "default_level": "low",
        "probe": None,
    }]), encoding="utf-8")
    profile = resolve(["gpt-oss-20b"], all_profiles(load_profiles(path)))
    assert profile.id == "gpt-oss" and profile.level_names() == ["low", "high"]
    assert profile.probe is None and profile.operating_point("low") is None


def test_extending_an_unknown_profile_is_an_error(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps([{"id": "x", "extends": "nope"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profiles(path)
