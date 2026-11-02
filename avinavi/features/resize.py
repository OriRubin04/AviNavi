"""The "use the configured resize" sentinel, shared by both extractors.

It lives in its own module so neither feature backend has to import the other.

Why a sentinel rather than ``None``
-----------------------------------
The obvious spelling is wrong in a way that is silent and costly::

    def extract(self, image, resize_max_side: Optional[int] = None):
        max_side = resize_max_side if resize_max_side is not None else self.resize_max_side

With that, a caller passing ``None`` to mean *do not resize this image* gets the
extractor's configured resize instead — the two meanings collapse. And a caller
did exactly that: :func:`avinavi.sim.evaluate.build_tile_database` passes
``None`` precisely because a 1024 px map tile should be described at **full
resolution** rather than at the live camera's smaller working size, and its
tiles were quietly being downscaled anyway.

Nothing failed. No warning appeared. The tile features were simply computed at a
different scale from the one intended, which costs match quality in a way that
looks like ordinary difficulty.

With the sentinel the two meanings are distinct:

* argument omitted  → use the extractor's configured ``resize_max_side``
* ``resize_max_side=None`` → do not resize at all
* ``resize_max_side=512`` → resize so the longest side is 512
"""

from __future__ import annotations


class _UseConfiguredResize:
    """A distinct type so the sentinel is recognisable in a traceback."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<use the extractor's configured resize_max_side>"


#: Default for every ``resize_max_side`` parameter.  See the module docstring.
USE_CONFIGURED_RESIZE = _UseConfiguredResize()
