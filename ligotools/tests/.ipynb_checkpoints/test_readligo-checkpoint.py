import pytest
import os
import numpy as np


from ligotools import readligo 

TEST_FILE_H1 = 'data/H-H1_LOSC_4_V2-1126259446-32.hdf5' 

def check_data_file_exists(filepath):
    """Helper to check if a required external data file exists."""
    if not os.path.exists(filepath):
        pytest.skip(f"Required test data file not found: {filepath}. Cannot run test.")

def test_loaddata_output_types():
    """Tests that loaddata returns strain, time, and a dict, and they are not None."""
    check_data_file_exists(TEST_FILE_H1)
    
    strain, time, chan_dict = readligo.loaddata(TEST_FILE_H1, 'H1')
    
    assert isinstance(strain, np.ndarray), "Strain should be a NumPy array."
    assert isinstance(time, np.ndarray), "Time should be a NumPy array."
    assert isinstance(chan_dict, dict), "Channel dictionary should be a dictionary."
    
    assert strain is not None
    assert time is not None
    assert chan_dict is not None


def test_loaddata_data_integrity():
    """Tests the length and integrity of the loaded data arrays."""
    check_data_file_exists(TEST_FILE_H1)
    
    strain, time, _ = readligo.loaddata(TEST_FILE_H1, 'H1')
    
    expected_length = 131072
    assert len(strain) == expected_length, f"Expected {expected_length} samples, got {len(strain)}."
    
    assert len(strain) == len(time), "Strain and Time arrays must have the same length."
