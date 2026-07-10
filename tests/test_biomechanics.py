import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.biomechanics import (
    compute_smoothness,
    compute_spectral_arc_length,
    compute_symmetry,
    compute_periodicity,
    compute_range_of_motion
)

def create_synthetic_data(duration_ms=2000, num_joints=4, func=np.sin):
    """Creates a synthetic DataFrame with 'tick' and 'q'."""
    ticks = np.arange(0, duration_ms, 10)  # 10ms step (100Hz)
    t_sec = ticks / 1000.0
    
    q_data = []
    for t in t_sec:
        # Create a predictable pattern
        row = [func(2 * np.pi * t + (i * 0.1)) for i in range(num_joints)]
        q_data.append(row)
        
    return pd.DataFrame({
        'tick': ticks,
        'q': q_data
    })

def test_compute_smoothness():
    # A perfect sine wave is very smooth
    df = create_synthetic_data(func=np.sin)
    res = compute_smoothness(df)
    
    assert "mean_ldlj" in res
    assert res["mean_ldlj"] < 0 # LDLJ is negative
    
def test_compute_spectral_arc_length():
    df = create_synthetic_data(func=np.sin)
    res = compute_spectral_arc_length(df)
    
    assert "mean_sparc" in res
    assert res["mean_sparc"] <= 0 # SPARC is negative or 0

def test_compute_symmetry():
    # 4 joints, left half (0,1) and right half (2,3)
    # The amplitudes are identical (sin wave), just slight phase shift.
    # ROM should be identical.
    df = create_synthetic_data(func=np.sin)
    res = compute_symmetry(df)
    
    assert "mean_symmetry_index" in res
    # Should be close to 0 symmetry index since ROM is identical
    assert abs(res["mean_symmetry_index"]) < 1e-1 

def test_compute_periodicity():
    df = create_synthetic_data(func=np.sin)
    res = compute_periodicity(df)
    
    assert "regularity_score" in res
    # A sine wave is highly periodic
    assert res["regularity_score"] > 0.5 

def test_compute_range_of_motion():
    df = create_synthetic_data(func=np.sin)
    res = compute_range_of_motion(df)
    
    assert "mean_rom" in res
    # Sine wave goes from -1 to 1, so ROM is ~2.0
    assert 1.9 < res["mean_rom"] < 2.1

def test_insufficient_data():
    df = pd.DataFrame()
    assert compute_smoothness(df)["mean_ldlj"] == 0.0
    assert compute_symmetry(df)["mean_symmetry_index"] is None
    assert compute_periodicity(df)["regularity_score"] == 0.0
    assert compute_range_of_motion(df)["mean_rom"] == 0.0
