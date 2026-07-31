"""The gate between the visual solution and the flight controller.

Section 26 of the project specification requires that the visual position be
treated as monitoring data first, and only later as a flight-control input.
That progression is enforced in two places:

* ``output.send_to_autopilot`` is the master switch and defaults to ``false``,
  so a fresh checkout computes and logs everything and transmits nothing;
* this class, which withholds transmission until the solution has *earned* it
  and withdraws it the moment it stops looking healthy.

The checks are deliberately conservative.  Suppressing a good solution costs a
few hundred milliseconds of external navigation; transmitting a bad one can put
the aircraft somewhere it is not.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from ..geo import haversine_distance_m
from ..types import GeoPoint, VisualFix

LOG = logging.getLogger(__name__)


class SafetyGate:
    """Decides whether a fix may reach the autopilot."""

    def __init__(
        self,
        required_consecutive_good_fixes: int = 5,
        max_consecutive_failures: int = 10,
        max_position_jump_m: float = 100.0,
        require_heartbeat: bool = True,
    ) -> None:
        self.required_consecutive_good_fixes = int(required_consecutive_good_fixes)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.max_position_jump_m = float(max_position_jump_m)
        self.require_heartbeat = bool(require_heartbeat)

        self._good_streak = 0
        self._failure_streak = 0
        self._armed = False
        self._last_geo: Optional[GeoPoint] = None
        self._last_reason: str = "waiting for the first good fix"

    @classmethod
    def from_config(cls, config) -> "SafetyGate":
        return cls(
            required_consecutive_good_fixes=config.required_consecutive_good_fixes,
            max_consecutive_failures=config.max_consecutive_failures,
            max_position_jump_m=config.max_position_jump_m,
            require_heartbeat=config.require_heartbeat,
        )

    @property
    def armed(self) -> bool:
        """True once enough consecutive good fixes have accumulated."""
        return self._armed

    @property
    def reason(self) -> str:
        return self._last_reason

    @property
    def good_streak(self) -> int:
        return self._good_streak

    def observe(self, fix: VisualFix) -> Tuple[bool, str]:
        """Feed a fix in; returns ``(accepted, reason)``.

        A rejected fix is still logged and still shown in the diagnostics - it
        just does not advance the arming streak and, for a large jump, is not
        handed to the propagator either.
        """
        if not fix.valid:
            self._failure_streak += 1
            self._good_streak = 0
            if self._armed and self._failure_streak >= self.max_consecutive_failures:
                self._armed = False
                self._last_reason = (
                    "%d consecutive invalid fixes; external navigation disarmed"
                    % self._failure_streak
                )
                LOG.warning("SAFETY: %s", self._last_reason)
            return (False, "fix is invalid: %s" % fix.reason.value)

        if fix.geo is not None and self._last_geo is not None:
            jump = haversine_distance_m(self._last_geo, fix.geo)
            if jump > self.max_position_jump_m:
                # A jump this large is either a mis-localization against the
                # wrong tile or a genuine re-acquisition after a long dropout.
                # Either way it must not be fused as a continuous measurement.
                self._good_streak = 0
                self._armed = False
                self._last_geo = fix.geo
                self._last_reason = (
                    "position jumped %.0f m (limit %.0f m); re-arming from scratch"
                    % (jump, self.max_position_jump_m)
                )
                LOG.warning("SAFETY: %s", self._last_reason)
                return (False, self._last_reason)

        self._failure_streak = 0
        self._good_streak += 1
        self._last_geo = fix.geo

        if not self._armed and self._good_streak >= self.required_consecutive_good_fixes:
            self._armed = True
            self._last_reason = (
                "%d consecutive good fixes; external navigation armed" % self._good_streak
            )
            LOG.info("SAFETY: %s", self._last_reason)
        elif not self._armed:
            self._last_reason = "arming: %d/%d good fixes" % (
                self._good_streak,
                self.required_consecutive_good_fixes,
            )
        else:
            self._last_reason = "armed"

        return (True, self._last_reason)

    def may_transmit(self, link_ok: bool) -> Tuple[bool, str]:
        """Final check immediately before a message goes out."""
        if self.require_heartbeat and not link_ok:
            return (False, "no MAVLink heartbeat")
        if not self._armed:
            return (False, self._last_reason)
        return (True, "armed")

    def reset(self) -> None:
        self._good_streak = 0
        self._failure_streak = 0
        self._armed = False
        self._last_geo = None
        self._last_reason = "reset"
