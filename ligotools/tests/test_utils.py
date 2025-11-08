import numpy as np
from ligotools import utils


def test_whiten_output_scaling_simplified():
    """
    Test that whiten returns same-length array and roughly unit variance (std dev near 1.0)
    using simple, synthetic data.
    """
    
    
    fs = 4096 
    noise_floor = fs 
    
    dt = 1.0 / fs
    N = 4096 
    strain = np.random.normal(0, 1.0, N)
    
    #strain = np.random.normal(0, 1e-21, N) 

    freqs = np.fft.rfftfreq(N, dt)
    #noise_floor = 1e-42

    psd_vals = noise_floor * np.ones_like(freqs)
    whitened_strain = utils.whiten(strain, psd_vals, dt)

    assert isinstance(whitened_strain, np.ndarray)
    assert len(whitened_strain) == len(strain)
    assert 0.1 < np.std(whitened_strain) < 2.0


def test_write_wavfile(tmp_path):
    """Test that write_wavfile writes a file of correct length."""
    filename = tmp_path / "test.wav"
    rate = 4096
    data = np.random.randn(rate).astype(np.float32)

    utils.write_wavfile(str(filename), rate, data)

    assert filename.exists()
    # optional: read file back in to verify
    import wave
    with wave.open(str(filename), 'rb') as f:
        assert f.getnframes() == rate


def test_reqshift_changes_frequency():
    """Test that reqshift shifts a sine wave frequency."""
    fs = 4096
    dt = 1.0 / fs
    t = np.linspace(0, 1, fs)
    freq = 50  # Hz
    data = np.sin(2 * np.pi * freq * t)

    shifted = utils.reqshift(data, fshift=10, sample_rate=fs)

    assert isinstance(shifted, np.ndarray)
    assert len(shifted) == len(data)
    assert not np.allclose(data, shifted)
