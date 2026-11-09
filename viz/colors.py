"""Per-face colour schemes (pure NumPy, uint8 RGB).

Coverage is drawn as a tint over the natural rock colour, so the shape and the sunlit /
dark sides stay readable.  The live scanner footprint is painted on top:
bright = imaged right now, faint = inside the field of view but not counted (too far for
the resolution limit, too oblique, unlit or hidden).
"""
from __future__ import annotations

import numpy as np

# Spacecraft palette - chosen to stay distinct from the green/orange coverage colours.
SC_COLORS_HEX = ["#4da6ff", "#ff5c8a", "#b18cff", "#00e0e0", "#ffe14d", "#ff6b4d", "#e8e8e8"]


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


SC_COLORS = np.array([hex_to_rgb(h) for h in SC_COLORS_HEX])


def sc_color(i: int) -> np.ndarray:
    return SC_COLORS[i % len(SC_COLORS)]


def sc_color_hex(i: int) -> str:
    return SC_COLORS_HEX[i % len(SC_COLORS_HEX)]


ROCK = np.array([156, 146, 130.0])               # Eros: a fairly uniform grey-tan regolith
IMAGED = np.array([40, 225, 95.0])
NEVER_LIT = np.array([60, 95, 200.0])            # never sunlit during the mission: not part of A*

# sequential palette for "number of spacecraft that saw the facet" (1 .. 7+)
HEAT = np.array([[0, 0, 0], [40, 60, 160], [30, 140, 190], [60, 190, 120],
                 [200, 210, 60], [250, 150, 40], [230, 70, 50], [180, 20, 90]], dtype=float)

COLOR_MODES = ["Coverage", "By spacecraft", "Redundancy heat", "Natural"]

LEGENDS = {
    "Coverage": [("imaged so far", IMAGED), ("not imaged yet", ROCK), ("never sunlit this mission", NEVER_LIT)],
    "Redundancy heat": [(f"{i} spacecraft" if i < 7 else "7+ spacecraft", HEAT[i]) for i in range(1, 8)],
}
FOOTPRINT_LEGEND = [("bright = imaged right now", np.array([235, 240, 255.0])),
                    ("faint = in view, not counted", np.array([120, 125, 150.0]))]


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def natural(n_faces: int, seed: int = 7) -> np.ndarray:
    """Rock colour with a little per-face variation (float RGB)."""
    rng = np.random.default_rng(seed)
    return ROCK[None, :] * rng.uniform(0.86, 1.06, (n_faces, 1))


def _mix(base: np.ndarray, col, w: float, mask: np.ndarray) -> None:
    base[mask] = (1 - w) * base[mask] + w * np.asarray(col, float)


def paint(mode: str, k: np.ndarray | None, seen: np.ndarray | None, base: np.ndarray,
          never_lit: np.ndarray | None = None) -> np.ndarray:
    """Float RGB per face.  k: spacecraft count per face, seen: (N, F) bool, base: natural rock,
    never_lit: faces that are not sunlit at any time of the mission (outside A*)."""
    out = base.copy()
    if mode == "Natural" or k is None:
        return out
    if mode == "Coverage":
        if never_lit is not None:
            _mix(out, NEVER_LIT, 0.75, never_lit)
        _mix(out, IMAGED, 0.85, k >= 1)
    elif mode == "By spacecraft":
        cnt = seen.sum(axis=0)
        cols = np.array([sc_color(i) for i in range(seen.shape[0])])
        mean = (seen.T.astype(float) @ cols) / np.maximum(cnt, 1)[:, None]
        m = cnt > 0
        out[m] = 0.4 * out[m] + 0.6 * mean[m]
    elif mode == "Redundancy heat":
        m = k >= 1
        out[m] = 0.35 * out[m] + 0.65 * HEAT[np.clip(k[m], 0, len(HEAT) - 1)]
    else:
        raise ValueError(mode)
    return out


def footprint(rgb: np.ndarray, imaged: np.ndarray, in_view: np.ndarray) -> np.ndarray:
    """Overlay the live scanner footprints: imaged (N, F) bright, in_view-only (N, F) faint."""
    out = rgb.copy()
    for i in range(imaged.shape[0]):
        faint = np.flatnonzero(in_view[i] & ~imaged[i])          # footprints are small: index, don't mask
        out[faint] = 0.65 * out[faint] + 0.35 * sc_color(i)
        bright = np.flatnonzero(imaged[i])
        out[bright] = 0.15 * out[bright] + 0.85 * (0.55 * sc_color(i) + 0.45 * 255)
    return _u8(out)
