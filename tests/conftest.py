"""Shared fixtures for gw_remnant test suite."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def synthetic_waveform():
    """Minimal chirp-like (2,2) mode on a short time array for fast unit tests.

    Returns (time, h_dict, q) where h_dict contains a single (2,2) mode.
    """
    time = np.linspace(-500, 50, 5501)
    freq = 0.02 + 0.0001 * (time - time[0])
    phase = np.cumsum(freq) * np.gradient(time)
    amp = 0.1 * np.exp(-((time - 0) ** 2) / (2 * 50**2))
    h22 = amp * np.exp(-1j * 2 * phase)
    h_dict = {(2, 2): h22}
    q = 2.0
    return time, h_dict, q


@pytest.fixture
def q8_nr_data():
    """Load q=8 non-spinning NR data from tutorials/q8_NR.npy.

    Returns (time, h_dict, q).
    """
    data_path = Path(__file__).resolve().parent.parent / "tutorials" / "q8_NR.npy"
    if not data_path.exists():
        pytest.skip(f"NR data file not found: {data_path}")
    data = np.load(data_path, allow_pickle=True).item()
    time = data["t"]
    h_dict = {k: v for k, v in data.items() if k != "t"}
    return time, h_dict, 8
