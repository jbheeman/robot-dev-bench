import pytest
import numpy as np
import pandas as pd
from processing.filter import TelemetryFilter
from processing.synchronizer import DataSynchronizer

def test_butterworth_filter():
    """Test that the zero-phase filter successfully attenuates high-frequency noise."""
    # Create a clean 1Hz sine wave sampled at 100Hz
    t = np.linspace(0, 1.0, 100, endpoint=False)
    clean_signal = np.sin(2 * np.pi * 1.0 * t)
    
    # Add high-frequency noise (30Hz)
    noise = 0.5 * np.sin(2 * np.pi * 30.0 * t)
    noisy_signal = clean_signal + noise
    
    # Filter with a 15Hz cutoff
    t_filter = TelemetryFilter(sample_rate=100.0, cutoff_freq=15.0)
    filtered_signal = t_filter.filter_array(noisy_signal)
    
    # The filtered signal should be very close to the original clean signal
    mse_noisy = np.mean((noisy_signal - clean_signal)**2)
    mse_filtered = np.mean((filtered_signal - clean_signal)**2)
    
    assert mse_filtered < mse_noisy
    assert mse_filtered < 0.05

def test_dataframe_filtering():
    """Test filtering applied directly to DataFrame columns containing lists/arrays."""
    t = np.linspace(0, 1.0, 100)
    noisy_joints = [np.random.normal(0, 1, 12).tolist() for _ in range(100)]
    df = pd.DataFrame({"tick": range(100), "q": noisy_joints})
    
    t_filter = TelemetryFilter(sample_rate=100.0, cutoff_freq=10.0)
    filtered_df = t_filter.filter_dataframe_columns(df, columns=["q"])
    
    assert len(filtered_df) == 100
    assert "q" in filtered_df.columns
    assert len(filtered_df.iloc[0]["q"]) == 12

def test_synchronization():
    """Test pandas merge_asof on different frequency DataFrames."""
    # LowState at 100Hz (ticks 0, 10, 20, ...)
    low_data = {"tick": [0, 10, 20, 30, 40], "q": [[0]]*5}
    low_df = pd.DataFrame(low_data)
    
    # HighState at 50Hz (ticks 0, 20, 40)
    # Notice it is slightly misaligned (21 instead of 20)
    high_data = {"tick": [0, 21, 40], "odom": [[1]]*3}
    high_df = pd.DataFrame(high_data)
    
    # Sync with nearest neighbor, tolerance of 5 ticks
    sync_df = DataSynchronizer.sync_low_high_states(low_df, high_df, tolerance=5)
    
    assert len(sync_df) == 5
    # Tick 20 should match with HighState tick 21 (nearest)
    assert not pd.isna(sync_df.loc[sync_df['tick'] == 20, 'odom'].values[0])
    
    # Tick 10 is too far from 0 and 21 (tolerance is 5), so it should be NaN
    assert pd.isna(sync_df.loc[sync_df['tick'] == 10, 'odom'].values[0])
