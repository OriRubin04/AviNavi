"""Forward propagation of the last visual fix between localization cycles.

Visual localization runs at 2-5 Hz on a Raspberry Pi 5; ArduPilot's EKF wants
external navigation at 10-20 Hz.  Rather than retransmitting the same pose -
which lies to the filter about how recent the measurement is and can drive an
oscillation - this class propagates the last fix forward with the velocity
estimated from the two most recent fixes, and inflates the reported position
uncertainty with age.

The propagator is also where the "do not send stale data" rule is enforced.
Past ``max_age_s`` it returns ``None``, the output stage stops transmitting, and
ArduPilot's own external-navigation timeout takes over.  It never invents a
position it cannot justify.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from ..frames import ned_velocity_to_body_frd
from ..types import (
    Attitude,
    NavigationSolution,
    NEDPosition,
    NEDVelocity,
    VisualFix,
)

LOG = logging.getLogger(__name__)


class PosePropagator:
    """Holds the latest visual fix and produces solutions on demand."""

    def __init__(
        self,
        enabled: bool = True,
        max_age_s: float = 1.0,
        use_velocity: bool = True,
        max_velocity_dt_s: float = 2.0,
        max_speed_mps: float = 40.0,
        sigma_growth_mps: float = 2.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_age_s = float(max_age_s)
        self.use_velocity = bool(use_velocity)
        self.max_velocity_dt_s = float(max_velocity_dt_s)
        self.max_speed_mps = float(max_speed_mps)
        self.sigma_growth_mps = float(sigma_growth_mps)

        self._last: Optional[VisualFix] = None
        self._previous: Optional[VisualFix] = None
        self._velocity: Optional[NEDVelocity] = None
        self._reset_counter: int = 0

    @classmethod
    def from_config(cls, config) -> "PosePropagator":
        return cls(
            enabled=config.enabled,
            max_age_s=config.max_age_s,
            use_velocity=config.use_velocity,
            max_velocity_dt_s=config.max_velocity_dt_s,
            max_speed_mps=config.max_speed_mps,
            sigma_growth_mps=config.sigma_growth_mps,
        )

    # -- state --------------------------------------------------------------
    @property
    def last_fix(self) -> Optional[VisualFix]:
        return self._last

    @property
    def velocity(self) -> Optional[NEDVelocity]:
        return self._velocity

    @property
    def reset_counter(self) -> int:
        """Incremented whenever the estimate jumps, per the MAVLink field's contract."""
        return self._reset_counter

    def note_reset(self) -> None:
        """Record a discontinuity so ``ODOMETRY.reset_counter`` advances.

        Called when localization recovers after a dropout or the fix jumps far
        enough that the EKF should treat it as a new track rather than a
        continuous measurement.
        """
        self._reset_counter = (self._reset_counter + 1) % 256

    def clear(self) -> None:
        """Forget everything - used when the link or the camera drops."""
        self._last = None
        self._previous = None
        self._velocity = None

    # -- updates ------------------------------------------------------------
    def update(self, fix: VisualFix) -> None:
        """Accept a new *valid* visual fix."""
        if not fix.valid or fix.position_ned is None:
            return
        self._previous = self._last
        self._last = fix
        self._velocity = self._estimate_velocity(self._previous, fix)

    def _estimate_velocity(
        self, previous: Optional[VisualFix], current: VisualFix
    ) -> Optional[NEDVelocity]:
        if not self.use_velocity or previous is None or previous.position_ned is None:
            return None
        dt = current.timestamp_monotonic - previous.timestamp_monotonic
        if dt <= 1e-3 or dt > self.max_velocity_dt_s:
            return None
        a = previous.position_ned
        b = current.position_ned
        velocity = NEDVelocity(
            (b.north_m - a.north_m) / dt,
            (b.east_m - a.east_m) / dt,
            (b.down_m - a.down_m) / dt,
        )
        speed = math.sqrt(
            velocity.north_mps**2 + velocity.east_mps**2 + velocity.down_mps**2
        )
        if speed > self.max_speed_mps:
            LOG.warning(
                "discarding implausible visual velocity of %.1f m/s (limit %.1f)",
                speed,
                self.max_speed_mps,
            )
            return None
        return velocity

    # -- output -------------------------------------------------------------
    def solution_at(self, now_monotonic: float) -> Optional[NavigationSolution]:
        """Best available navigation solution at ``now_monotonic``.

        Returns ``None`` when there is nothing valid to send, which the caller
        must treat as "transmit nothing", not as "resend the last message".
        """
        fix = self._last
        if fix is None or fix.position_ned is None or fix.attitude is None:
            return None

        age = now_monotonic - fix.timestamp_monotonic
        if age < 0.0:
            age = 0.0
        if age > self.max_age_s:
            return None

        position = fix.position_ned
        velocity = self._velocity if self.enabled else None
        if velocity is not None and age > 0.0:
            position = NEDPosition(
                position.north_m + velocity.north_mps * age,
                position.east_m + velocity.east_mps * age,
                position.down_m + velocity.down_mps * age,
            )

        sigma = fix.position_sigma_m
        if sigma != sigma:  # NaN
            sigma = 1.0
        sigma += self.sigma_growth_mps * age

        yaw_sigma = fix.yaw_sigma_rad
        if yaw_sigma != yaw_sigma:
            yaw_sigma = 0.2

        velocity_ned = velocity or NEDVelocity(0.0, 0.0, 0.0)
        attitude = fix.attitude or Attitude()
        body_velocity = ned_velocity_to_body_frd(
            velocity_ned.as_tuple(), attitude.as_tuple()
        )

        return NavigationSolution(
            timestamp_monotonic=now_monotonic,
            timestamp_unix=fix.timestamp_unix + age,
            position_ned=position,
            attitude=attitude,
            velocity_body_frd=body_velocity,
            position_sigma_m=sigma,
            # Velocity from differenced fixes is far noisier than the positions
            # themselves; scale accordingly rather than pretending otherwise.
            velocity_sigma_mps=max(0.2, 2.0 * sigma / max(self.max_velocity_dt_s, 0.1)),
            yaw_sigma_rad=yaw_sigma,
            confidence=fix.confidence,
            age_s=age,
            propagated=age > 1e-3,
            reset_counter=self._reset_counter,
        )

    def has_velocity(self) -> bool:
        return self._velocity is not None
