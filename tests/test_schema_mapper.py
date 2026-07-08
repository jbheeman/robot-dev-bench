import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.schema_mapper import SchemaMapper

def test_format_a_passthrough():
    """Test that Format A (our native schema) passes through untouched."""
    df = pd.DataFrame({
        'tick': [0, 10, 20],
        'q': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
        'q_cmd': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
        'tau': [[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]]
    })
    
    out_df = SchemaMapper.normalise(df)
    
    assert 'q' in out_df.columns
    assert 'tick' in out_df.columns
    assert 'tau' in out_df.columns
    assert out_df['tick'].iloc[1] == 10

def test_format_b_lerobot_single_body():
    """Test mapping of LeRobot 'observation.state' and 'action' arrays."""
    df = pd.DataFrame({
        'timestamp': [0.0, 0.01, 0.02],
        'observation.state': [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        'action': [[1.5, 2.5], [1.6, 2.6], [1.7, 2.7]],
        'frame_index': [0, 1, 2]
    })
    
    out_df = SchemaMapper.normalise(df)
    
    # Must have the canonical columns added
    assert 'q' in out_df.columns
    assert 'q_cmd' in out_df.columns
    assert 'tick' in out_df.columns
    
    # Timestamp (s) -> tick (ms)
    assert out_df['tick'].iloc[1] == 10.0
    assert out_df['tick'].iloc[2] == 20.0
    
    # Check values mapped correctly
    assert out_df['q'].iloc[0] == [1.0, 2.0]
    assert out_df['q_cmd'].iloc[0] == [1.5, 2.5]
    
    # Original columns should be preserved
    assert 'observation.state' in out_df.columns

def test_format_c_lerobot_segmented():
    """Test mapping and concatenation of segmented LeRobot joints."""
    df = pd.DataFrame({
        'timestamp': [0.0, 0.01],
        'observation.left_arm': [[1.0, 1.1], [1.2, 1.3]],
        'observation.right_arm': [[2.0, 2.1], [2.2, 2.3]],
        'observation.gripper': [3.0, 3.1],  # Scalar
        'action.left_arm': [[4.0, 4.1], [4.2, 4.3]],
        'action.right_arm': [[5.0, 5.1], [5.2, 5.3]],
        'action.gripper': [6.0, 6.1]       # Scalar
    })
    
    out_df = SchemaMapper.normalise(df)
    
    assert 'q' in out_df.columns
    assert 'q_cmd' in out_df.columns
    assert 'tick' in out_df.columns
    
    # Suffixes sorted alphabetically: 'gripper', 'left_arm', 'right_arm'
    # So concatenation order for q should be:
    # gripper, left_arm[0], left_arm[1], right_arm[0], right_arm[1]
    expected_q_row0 = [3.0, 1.0, 1.1, 2.0, 2.1]
    expected_q_cmd_row0 = [6.0, 4.0, 4.1, 5.0, 5.1]
    
    assert out_df['q'].iloc[0] == expected_q_row0
    assert out_df['q_cmd'].iloc[0] == expected_q_cmd_row0
    
    # Second row
    assert out_df['q'].iloc[1] == [3.1, 1.2, 1.3, 2.2, 2.3]

def test_empty_dataframe():
    """Test empty dataframe returns unchanged."""
    df = pd.DataFrame()
    out_df = SchemaMapper.normalise(df)
    assert out_df.empty

def test_unknown_schema():
    """Test that a completely foreign schema is just passed through."""
    df = pd.DataFrame({
        'weird_col1': [1, 2],
        'weird_col2': [3, 4]
    })
    out_df = SchemaMapper.normalise(df)
    assert 'weird_col1' in out_df.columns
    assert 'q' not in out_df.columns
