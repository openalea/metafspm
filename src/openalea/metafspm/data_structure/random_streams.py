"""
Reproducible random draws, one stream per entity.

A draw is a pure function of (seed, stream name, step, entity id): a counter-based generator (SplitMix64 mixing of
the four keys), so draws do not depend on the order in which entities are visited, nor on which other entities exist.
A model that drew from a global generator in visiting order (rhizodep's re-seeding with `random_choice * vid`) gets the
same distributions, reproducibly, but not the same draws.

    u = uniform(vids, seed=plant_seed, stream="lateral_emergence", step=iteration)
"""
import zlib

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1, _MIX2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser (uint64, wrapping arithmetic)."""
    x = (x ^ (x >> np.uint64(30))) * _MIX1
    x = (x ^ (x >> np.uint64(27))) * _MIX2
    return x ^ (x >> np.uint64(31))


def _key(text: str) -> np.uint64:
    return np.uint64(zlib.crc32(text.encode("utf-8")))


def bits(ids, seed: int = 0, stream: str = "", step: int = 0, draw: int = 0) -> np.ndarray:
    """64 random bits per id, a pure function of (seed, stream, step, draw, id)."""
    ids = np.asarray(ids, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        state = _mix(np.uint64(seed) * _GOLDEN + _key(stream))
        state = _mix(state ^ (np.uint64(step) * _GOLDEN + np.uint64(draw)))
        return _mix(state ^ (ids * _GOLDEN + _GOLDEN))


def uniform(ids, low=0., high=1., seed: int = 0, stream: str = "", step: int = 0, draw: int = 0) -> np.ndarray:
    """Uniform draws in [low, high), one per id (53-bit resolution)."""
    u = (bits(ids, seed, stream, step, draw) >> np.uint64(11)).astype(np.float64) * 2. ** -53
    return low + (high - low) * u


def normal(ids, loc=0., scale=1., seed: int = 0, stream: str = "", step: int = 0) -> np.ndarray:
    """Normal draws (Box-Muller on two independent uniform draws)."""
    u1 = uniform(ids, seed=seed, stream=stream, step=step, draw=0)
    u2 = uniform(ids, seed=seed, stream=stream, step=step, draw=1)
    return loc + scale * np.sqrt(-2. * np.log1p(-u1)) * np.cos(2. * np.pi * u2)


def exponential(ids, scale=1., seed: int = 0, stream: str = "", step: int = 0) -> np.ndarray:
    return -scale * np.log1p(-uniform(ids, seed=seed, stream=stream, step=step))


def integers(ids, low: int, high: int, seed: int = 0, stream: str = "", step: int = 0) -> np.ndarray:
    """Integers in [low, high)."""
    return np.floor(uniform(ids, low, high, seed=seed, stream=stream, step=step)).astype(np.int64)


DISTRIBUTIONS = {"uniform": uniform, "normal": normal, "exponential": exponential, "integers": integers}


def draw(distribution: str, ids, **options) -> np.ndarray:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution '{distribution}', expected one of {list(DISTRIBUTIONS)}")
    return DISTRIBUTIONS[distribution](ids, **options)
