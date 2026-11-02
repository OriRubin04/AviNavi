"""Manual re-anchoring: the operator clicks the map to say where to search.

Two things make this worth testing carefully.

First, a re-anchor must only ever move the *search window*.  It is not a
position fix, and the next frame still has to earn its match, which is what
makes a click safe to accept in flight.  A channel that leaked straight into
the published position would be a remote-control input to the navigation
solution.

Second, the same command number carries **degE7 integers** in ``COMMAND_INT``
and **plain float degrees** in ``COMMAND_LONG``.  Reading one as the other is a
factor of ten million: the point does not land slightly wrong, it vanishes off
the planet.  Both decodings are asserted, and so is the rejection of a sender
that confuses them.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

from avinavi.config import ReanchorConfig
from avinavi.mavlink.reanchor import (
    MAV_CMD_DO_SET_ROI,
    MAV_CMD_SPATIAL_USER_1,
    MAV_RESULT_ACCEPTED,
    MAV_RESULT_DENIED,
    ReanchorListener,
)
from avinavi.types import GeoPoint

from tests.fakes import FakeConnection

LAT, LON = 32.0853, 34.7818


class Message:
    """A stand-in for a decoded MAVLink message.

    The listener only ever reads attributes and the source ids, so a namespace
    is enough and keeps each test's intent visible in one place.
    """

    def __init__(self, message_type: str, src_system: int = 255,
                 src_component: int = 190, **fields) -> None:
        self._type = message_type
        self._src_system = src_system
        self._src_component = src_component
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self) -> str:
        return self._type

    def get_srcSystem(self) -> int:  # noqa: N802 - pymavlink's own spelling
        return self._src_system

    def get_srcComponent(self) -> int:  # noqa: N802
        return self._src_component


def command_int(**fields) -> Message:
    defaults = dict(
        target_system=1, target_component=197, command=MAV_CMD_SPATIAL_USER_1,
        param1=0.0, param2=0.0, param3=0.0, param4=0.0,
        x=int(LAT * 1e7), y=int(LON * 1e7), z=0.0, frame=0,
    )
    defaults.update(fields)
    return Message("COMMAND_INT", **defaults)


def command_long(**fields) -> Message:
    defaults = dict(
        target_system=1, target_component=197, command=MAV_CMD_SPATIAL_USER_1,
        param1=0.0, param2=0.0, param3=0.0, param4=0.0,
        param5=LAT, param6=LON, param7=0.0,
    )
    defaults.update(fields)
    return Message("COMMAND_LONG", **defaults)


def listener(**overrides) -> ReanchorListener:
    config = ReanchorConfig()
    config.control_socket_host = ""
    for key, value in overrides.items():
        setattr(config, key, value)
    connection = FakeConnection()
    connection.system_id = 1
    connection.component_id = 197
    return ReanchorListener.from_config(config, connection)


# ---------------------------------------------------------------------------
# Handing the request over
# ---------------------------------------------------------------------------
def test_a_request_is_handed_over_exactly_once():
    """Otherwise the engine would re-anchor on every cycle forever."""
    reanchor = listener()
    reanchor.submit(GeoPoint(LAT, LON, 0.0), source="test")
    assert reanchor.take() is not None
    assert reanchor.take() is None


def test_the_latest_request_wins():
    """A second click supersedes the first; it does not queue behind it."""
    reanchor = listener()
    reanchor.submit(GeoPoint(LAT, LON, 0.0), source="first")
    reanchor.submit(GeoPoint(LAT + 0.01, LON, 0.0), source="second")
    request = reanchor.take()
    assert request.source == "second"
    assert request.geo.lat_deg == pytest.approx(LAT + 0.01)


def test_a_request_carries_the_configured_radius_by_default():
    reanchor = listener(radius_m=1234.0)
    reanchor.submit(GeoPoint(LAT, LON, 0.0), source="test")
    assert reanchor.take().radius_m == 1234.0


def test_an_explicit_radius_overrides_the_default():
    reanchor = listener(radius_m=1234.0)
    reanchor.submit(GeoPoint(LAT, LON, 0.0), source="test", radius_m=500.0)
    assert reanchor.take().radius_m == 500.0


def test_the_callback_fires_so_the_runner_can_react_immediately():
    seen = []
    reanchor = listener()
    reanchor.on_request = seen.append
    reanchor.submit(GeoPoint(LAT, LON, 0.0), source="test")
    assert len(seen) == 1
    assert seen[0].source == "test"


def test_a_failing_callback_does_not_lose_the_request():
    def explode(_request):
        raise RuntimeError("boom")

    reanchor = listener()
    reanchor.on_request = explode
    reanchor.submit(GeoPoint(LAT, LON, 0.0), source="test")
    assert reanchor.take() is not None


# ---------------------------------------------------------------------------
# COMMAND_INT / MAV_CMD_SPATIAL_USER_1 — the recommended channel
# ---------------------------------------------------------------------------
def test_a_spatial_user_command_int_decodes_dege7():
    reanchor = listener()
    reanchor._on_command_int(command_int())
    request = reanchor.take()
    assert request is not None
    assert request.geo.lat_deg == pytest.approx(LAT, abs=1e-7)
    assert request.geo.lon_deg == pytest.approx(LON, abs=1e-7)
    assert request.source == "spatial_user_1"


def test_a_spatial_user_command_int_is_acknowledged():
    """The ACK is what turns a click into visible confirmation for the operator.

    None of the eavesdropping channels can offer this, which is the main reason
    this one is recommended.
    """
    reanchor = listener()
    reanchor._on_command_int(command_int())
    ack = reanchor.connection.last("COMMAND_ACK")
    assert ack is not None
    assert ack.command == MAV_CMD_SPATIAL_USER_1
    assert ack.result == MAV_RESULT_ACCEPTED


def test_an_undecodable_spatial_user_command_is_denied_not_ignored():
    """Silence would leave the operator believing the click landed."""
    reanchor = listener()
    reanchor._on_command_int(command_int(x=0, y=0))
    assert reanchor.take() is None
    assert reanchor.connection.last("COMMAND_ACK").result == MAV_RESULT_DENIED


def test_param1_carries_an_optional_search_radius():
    reanchor = listener()
    reanchor._on_command_int(command_int(param1=800.0))
    assert reanchor.take().radius_m == 800.0


def test_a_command_addressed_to_another_component_is_ignored():
    """ArduPilot's router would not deliver it to us anyway; do not act on it."""
    reanchor = listener()
    reanchor._on_command_int(command_int(target_component=1))
    assert reanchor.take() is None


def test_a_broadcast_command_is_accepted():
    reanchor = listener()
    reanchor._on_command_int(command_int(target_system=0, target_component=0))
    assert reanchor.take() is not None


def test_the_spatial_user_channel_can_be_turned_off():
    reanchor = listener(accept_spatial_user=False)
    reanchor._on_command_int(command_int())
    assert reanchor.take() is None


# ---------------------------------------------------------------------------
# COMMAND_LONG — float degrees, not degE7
# ---------------------------------------------------------------------------
def test_a_spatial_user_command_long_decodes_float_degrees():
    reanchor = listener()
    reanchor._on_command_long(command_long())
    request = reanchor.take()
    assert request is not None
    assert request.geo.lat_deg == pytest.approx(LAT)
    assert request.source == "spatial_user_1_long"


def test_a_sender_that_packs_dege7_into_command_long_is_rejected():
    """The factor-of-ten-million trap, caught rather than flown.

    A naive sender reusing its COMMAND_INT code would put 320853000 into
    ``param5``.  Accepting that would move the search to a latitude that does
    not exist, and the aircraft would then search nowhere at all.
    """
    reanchor = listener()
    reanchor._on_command_long(command_long(param5=LAT * 1e7, param6=LON * 1e7))
    assert reanchor.take() is None


def test_a_zero_coordinate_command_long_is_ignored():
    reanchor = listener()
    reanchor._on_command_long(command_long(param5=0.0, param6=0.0))
    assert reanchor.take() is None


# ---------------------------------------------------------------------------
# DO_SET_ROI — eavesdropping, so no ACK
# ---------------------------------------------------------------------------
def test_a_roi_command_re_anchors():
    reanchor = listener()
    reanchor._on_command_int(command_int(command=MAV_CMD_DO_SET_ROI, target_component=1))
    request = reanchor.take()
    assert request is not None
    assert request.source == "roi_command_int"


def test_a_roi_command_is_not_acknowledged():
    """We are overhearing traffic meant for the autopilot.

    An ACK from us would collide with the autopilot's own reply to the same
    command and could leave the GCS believing the wrong thing happened.
    """
    reanchor = listener()
    reanchor._on_command_int(command_int(command=MAV_CMD_DO_SET_ROI, target_component=1))
    assert reanchor.connection.of_type("COMMAND_ACK") == []


def test_the_roi_channel_can_be_turned_off():
    reanchor = listener(accept_roi=False)
    reanchor._on_command_int(command_int(command=MAV_CMD_DO_SET_ROI, target_component=1))
    assert reanchor.take() is None


# ---------------------------------------------------------------------------
# SET_GPS_GLOBAL_ORIGIN — "Set EKF Origin Here", the zero-code path
# ---------------------------------------------------------------------------
def origin_message(lat=LAT, lon=LON, alt_mm=40_000) -> Message:
    return Message(
        "SET_GPS_GLOBAL_ORIGIN",
        target_system=1,
        latitude=int(lat * 1e7),
        longitude=int(lon * 1e7),
        altitude=alt_mm,
    )


def test_set_gps_global_origin_re_anchors():
    reanchor = listener()
    reanchor._on_set_gps_global_origin(origin_message())
    request = reanchor.take()
    assert request is not None
    assert request.source == "set_gps_global_origin"
    assert request.geo.lat_deg == pytest.approx(LAT, abs=1e-7)


def test_the_origin_altitude_is_millimetres_not_centimetres():
    """ArduPilot uses centimetres internally; this message does not.

    Carrying the internal unit over would be a factor of ten in altitude.
    """
    reanchor = listener()
    reanchor._on_set_gps_global_origin(origin_message(alt_mm=123_000))
    assert reanchor.take().geo.alt_m == pytest.approx(123.0)


def test_an_identical_repeated_origin_is_debounced():
    reanchor = listener()
    reanchor._on_set_gps_global_origin(origin_message())
    assert reanchor.take() is not None
    reanchor._on_set_gps_global_origin(origin_message())
    assert reanchor.take() is None


def test_a_genuinely_moved_origin_is_a_second_re_anchor():
    """Debouncing must not swallow a deliberate second click elsewhere."""
    reanchor = listener()
    reanchor._on_set_gps_global_origin(origin_message())
    reanchor.take()
    reanchor._on_set_gps_global_origin(origin_message(lat=LAT + 0.01))
    assert reanchor.take() is not None


def test_the_origin_channel_can_be_turned_off():
    reanchor = listener(accept_set_origin=False)
    reanchor.start()
    # Disabled at start() time, so the handler is never subscribed; calling it
    # directly would bypass the switch, so assert on the subscription instead.
    assert "SET_GPS_GLOBAL_ORIGIN" not in reanchor.connection.handlers


# ---------------------------------------------------------------------------
# HOME_POSITION — reliable but intrusive, so off by default
# ---------------------------------------------------------------------------
def home_message(lat=LAT, lon=LON) -> Message:
    return Message(
        "HOME_POSITION",
        latitude=int(lat * 1e7),
        longitude=int(lon * 1e7),
        altitude=40_000,
    )


def test_home_position_is_off_by_default():
    """Accepting it would mean the operator had moved the RTL point."""
    assert ReanchorConfig().accept_home_position is False


def test_the_first_home_position_is_a_baseline_not_a_request():
    """It is just where home already was, broadcast at boot."""
    reanchor = listener(accept_home_position=True)
    reanchor._on_home_position(home_message())
    assert reanchor.take() is None


def test_a_moved_home_position_re_anchors():
    reanchor = listener(accept_home_position=True)
    reanchor._on_home_position(home_message())
    reanchor._on_home_position(home_message(lat=LAT + 0.01))
    request = reanchor.take()
    assert request is not None
    assert request.source == "home_position"


def test_a_home_re_broadcast_at_the_same_place_is_ignored():
    """ArduPilot re-sends home on arming; that is not operator intent."""
    reanchor = listener(accept_home_position=True, min_move_m=25.0)
    reanchor._on_home_position(home_message())
    reanchor._on_home_position(home_message(lat=LAT + 0.00005))  # ~5 m
    assert reanchor.take() is None


# ---------------------------------------------------------------------------
# Coordinate sanity
# ---------------------------------------------------------------------------
def test_the_zero_zero_sentinel_is_not_a_position():
    """ArduPilot's own "use current location" marker, not the Gulf of Guinea."""
    assert ReanchorListener._decode_dege7(0, 0, 0.0, "test") is None


def test_coordinates_too_small_to_be_dege7_are_rejected():
    """A sender packing float degrees into an integer field looks like this."""
    assert ReanchorListener._decode_dege7(32, 34, 0.0, "test") is None


def test_out_of_range_coordinates_are_rejected():
    assert ReanchorListener._decode_dege7(int(120.0 * 1e7), int(34.0 * 1e7), 0.0, "t") is None
    assert ReanchorListener._decode_dege7(int(32.0 * 1e7), int(200.0 * 1e7), 0.0, "t") is None


def test_a_valid_dege7_pair_decodes():
    geo = ReanchorListener._decode_dege7(int(LAT * 1e7), int(LON * 1e7), 12.5, "test")
    assert geo.lat_deg == pytest.approx(LAT, abs=1e-7)
    assert geo.alt_m == 12.5


# ---------------------------------------------------------------------------
# The UDP control socket
# ---------------------------------------------------------------------------
def test_the_control_socket_accepts_a_json_point():
    reanchor = listener(control_socket_host="127.0.0.1", control_socket_port=14699)
    reanchor.start()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(
            json.dumps({"lat": LAT, "lon": LON, "radius_m": 900.0}).encode(),
            ("127.0.0.1", 14699),
        )
        deadline = time.monotonic() + 3.0
        while reanchor.pending is None and time.monotonic() < deadline:
            time.sleep(0.02)
        request = reanchor.take()
        assert request is not None
        assert request.radius_m == 900.0
    finally:
        reanchor.stop()


def test_the_control_socket_ignores_malformed_datagrams():
    reanchor = listener(control_socket_host="127.0.0.1", control_socket_port=14700)
    reanchor.start()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b"not json at all", ("127.0.0.1", 14700))
        sender.sendto(json.dumps({"lat": 999.0, "lon": 0.0}).encode(), ("127.0.0.1", 14700))
        time.sleep(0.4)
        assert reanchor.pending is None
    finally:
        reanchor.stop()


def test_the_socket_is_not_opened_when_no_host_is_configured():
    reanchor = listener(control_socket_host="")
    reanchor.start()
    try:
        assert reanchor._socket is None
    finally:
        reanchor.stop()
