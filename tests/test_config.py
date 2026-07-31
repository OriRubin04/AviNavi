"""Configuration loading, overrides and validation."""

from __future__ import annotations

import os

import pytest
import yaml

from avinavi.config import (
    Config,
    ConfigError,
    apply_overrides,
    load_config,
    validate_config,
)


def write_config(tmp_path, data) -> str:
    path = str(tmp_path / "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


def test_the_shipped_default_config_loads_and_validates():
    """config/default.yaml must always be a valid configuration."""
    config = load_config()
    assert config.mode == "offline"
    assert config.mavlink.device == "/dev/ttyAMA0"
    # Transmission is off by default; the safety progression depends on it.
    assert config.output.send_to_autopilot is False


def test_the_default_yaml_covers_every_dataclass_field():
    """A field with no YAML entry is a field nobody knows they can tune."""
    from avinavi.config import DEFAULT_CONFIG_PATH

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    from dataclasses import fields

    missing = [f.name for f in fields(Config) if f.name not in raw]
    assert missing == []


def test_a_partial_config_falls_back_to_defaults(tmp_path):
    path = write_config(tmp_path, {"mavlink": {"baudrate": 57600}})
    config = load_config(path)
    assert config.mavlink.baudrate == 57600
    assert config.mavlink.device == "/dev/ttyAMA0"


def test_an_unknown_key_is_rejected(tmp_path):
    """A typo must fail loudly rather than being silently ignored."""
    path = write_config(tmp_path, {"mavlink": {"buadrate": 57600}})
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(path)


def test_a_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "absent.yaml"))


def test_nested_overrides_are_applied():
    config = load_config(overrides=["mavlink.device=/dev/ttyACM0", "rates.camera_fps=60"])
    assert config.mavlink.device == "/dev/ttyACM0"
    assert config.rates.camera_fps == 60.0


def test_overrides_parse_yaml_scalars():
    config = load_config(
        overrides=["output.send_to_autopilot=true", "initial_position.manual_lat_deg=32.5"]
    )
    assert config.output.send_to_autopilot is True
    assert config.initial_position.manual_lat_deg == 32.5


def test_an_override_can_create_a_missing_section():
    data = apply_overrides({}, ["camera_extrinsics.rotation.yaw_deg=90"])
    assert data["camera_extrinsics"]["rotation"]["yaw_deg"] == 90


def test_a_malformed_override_is_rejected():
    with pytest.raises(ConfigError, match="key.path=value"):
        apply_overrides({}, ["nonsense"])


def test_deeply_nested_extrinsics_load(tmp_path):
    path = write_config(
        tmp_path,
        {
            "camera_extrinsics": {
                "rotation": {"yaw_deg": 90.0},
                "translation": {"x_m": 0.25, "z_m": -0.1},
            }
        },
    )
    config = load_config(path)
    assert config.camera_extrinsics.rotation.yaw_deg == 90.0
    assert config.camera_extrinsics.translation.x_m == 0.25
    assert config.camera_extrinsics.translation.z_m == -0.1
    assert config.camera_extrinsics.translation.y_m == 0.0


def test_a_list_field_loads(tmp_path):
    path = write_config(tmp_path, {"camera": {"distortion": [0.1, -0.02, 0.0, 0.0, 0.0]}})
    config = load_config(path)
    assert config.camera.distortion == pytest.approx([0.1, -0.02, 0.0, 0.0, 0.0])


def test_optional_fields_accept_null(tmp_path):
    path = write_config(tmp_path, {"camera": {"fx": None}})
    assert load_config(path).camera.fx is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_an_unknown_mode_is_rejected():
    config = Config()
    config.mode = "hover"
    with pytest.raises(ConfigError, match="mode must be"):
        validate_config(config)


def test_an_unknown_output_message_is_rejected():
    config = Config()
    config.output.message = "telepathy"
    with pytest.raises(ConfigError, match="output.message"):
        validate_config(config)


def test_odometry_slower_than_localization_is_rejected():
    """Publishing below the localization rate would waste every other fix."""
    config = Config()
    config.rates.visual_localization_hz = 10.0
    config.rates.mavlink_odometry_hz = 5.0
    with pytest.raises(ConfigError, match="must be >="):
        validate_config(config)


def test_manual_initial_position_requires_coordinates():
    config = Config()
    config.initial_position.source = "manual"
    with pytest.raises(ConfigError, match="manual_lat_deg"):
        validate_config(config)


def test_manual_initial_position_with_coordinates_validates():
    config = Config()
    config.initial_position.source = "manual"
    config.initial_position.manual_lat_deg = 32.0
    config.initial_position.manual_lon_deg = 34.8
    validate_config(config)


def test_an_out_of_range_confidence_threshold_is_rejected():
    config = Config()
    config.confidence.min_confidence = 1.5
    with pytest.raises(ConfigError, match="min_confidence"):
        validate_config(config)


def test_a_max_search_radius_below_the_nominal_one_is_rejected():
    config = Config()
    config.tile_search.search_radius_m = 500.0
    config.tile_search.max_search_radius_m = 100.0
    with pytest.raises(ConfigError, match="max_search_radius_m"):
        validate_config(config)


def test_a_non_positive_propagation_age_is_rejected():
    config = Config()
    config.propagation.max_age_s = 0.0
    with pytest.raises(ConfigError, match="max_age_s"):
        validate_config(config)


def test_an_unknown_connection_type_is_rejected():
    config = Config()
    config.mavlink.connection_type = "carrier_pigeon"
    with pytest.raises(ConfigError, match="connection_type"):
        validate_config(config)
