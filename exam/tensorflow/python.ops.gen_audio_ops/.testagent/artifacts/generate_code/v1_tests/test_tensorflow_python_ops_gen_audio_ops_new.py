"""Tests for tensorflow.python.ops.gen_audio_ops module."""

import math
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== BLOCK:HEADER START ====
# Test class and helper functions
class TestGenAudioOps:
    """Test class for gen_audio_ops module."""
    
    @staticmethod
    def generate_sine_wave(frequency, sample_rate, duration, channels=1):
        """Generate a sine wave audio signal."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)
        if channels > 1:
            signal = np.stack([signal] * channels, axis=-1)
        return signal.astype(np.float32)
    
    @staticmethod
    def generate_random_audio(shape, min_val=-1.0, max_val=1.0):
        """Generate random audio data within valid range."""
        audio = np.random.uniform(min_val, max_val, shape).astype(np.float32)
        return audio
    
    @staticmethod
    def compute_spectrogram_manual(audio, window_size, stride, magnitude_squared=False):
        """Manual computation of spectrogram for validation."""
        # Simplified manual computation for basic validation
        # In practice, this would use proper FFT
        num_samples = audio.shape[0]
        num_channels = audio.shape[1] if len(audio.shape) > 1 else 1
        
        if num_channels > 1:
            audio = audio.reshape(num_samples, num_channels)
        
        # Calculate number of windows
        num_windows = max(0, (num_samples - window_size) // stride + 1)
        
        # For basic validation, return a simple shape
        if magnitude_squared:
            return np.ones((num_channels, num_windows, window_size // 2 + 1), dtype=np.float32)
        else:
            return np.ones((num_channels, num_windows, window_size // 2 + 1), dtype=np.float32)
    
    @staticmethod
    def create_wav_data(audio, sample_rate):
        """Create WAV encoded data from audio."""
        # Use TensorFlow's encode_wav function
        audio_tensor = tf.constant(audio, dtype=tf.float32)
        sample_rate_tensor = tf.constant(sample_rate, dtype=tf.int32)
        wav_data = gen_audio_ops.encode_wav(audio_tensor, sample_rate_tensor)
        return wav_data.numpy()
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize("input_shape,window_size,stride,magnitude_squared,dtype", [
        ([16000, 1], 512, 256, False, "float32"),
        ([32000, 2], 1024, 512, True, "float32"),  # param_extension: 多通道音频和平方幅度测试
        ([8000, 1], 256, 128, False, "float32"),   # param_extension: 非2的幂窗口大小测试
    ])
    def test_audio_spectrogram_basic(self, input_shape, window_size, stride, magnitude_squared, dtype):
        """Test basic functionality of audio_spectrogram."""
        # Generate test audio data
        num_samples, num_channels = input_shape
        audio = self.generate_sine_wave(
            frequency=440.0,  # A4 note
            sample_rate=16000,
            duration=num_samples / 16000,
            channels=num_channels
        )
        
        # Convert to tensor
        audio_tensor = tf.constant(audio, dtype=getattr(tf, dtype))
        
        # Call the function
        spectrogram = gen_audio_ops.audio_spectrogram(
            input=audio_tensor,
            window_size=window_size,
            stride=stride,
            magnitude_squared=magnitude_squared
        )
        
        # Weak assertions (round 1)
        # 1. Shape assertion
        num_windows = max(0, (num_samples - window_size) // stride + 1)
        expected_channels = num_channels
        expected_freq_bins = window_size // 2 + 1
        
        assert spectrogram.shape == (expected_channels, num_windows, expected_freq_bins), \
            f"Expected shape ({expected_channels}, {num_windows}, {expected_freq_bins}), got {spectrogram.shape}"
        
        # 2. Dtype assertion
        assert spectrogram.dtype == tf.float32, \
            f"Expected dtype float32, got {spectrogram.dtype}"
        
        # 3. Finite values assertion
        spectrogram_np = spectrogram.numpy()
        assert np.all(np.isfinite(spectrogram_np)), \
            "Spectrogram contains non-finite values"
        
        # 4. Basic property assertions
        assert np.all(spectrogram_np >= 0), \
            "Spectrogram values should be non-negative"
        
        # For magnitude_squared=True, values should be squared
        if magnitude_squared:
            # Values should be positive (squared magnitudes)
            assert np.all(spectrogram_np >= 0), \
                "Squared magnitude spectrogram should have non-negative values"
        else:
            # Regular magnitude - still non-negative
            assert np.all(spectrogram_np >= 0), \
                "Magnitude spectrogram should have non-negative values"
        
        # Additional basic checks
        assert spectrogram_np.size > 0, "Spectrogram should not be empty"
        
        # Check that output has reasonable values (not all zeros unless input is zero)
        if np.any(audio != 0):
            assert np.any(spectrogram_np > 0), \
                "Spectrogram should have positive values for non-zero input"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
    @pytest.mark.parametrize("audio_shape,sample_rate,desired_channels,desired_samples,dtype", [
        ([44100, 2], 44100, -1, -1, "float32"),
        ([22050, 1], 22050, 2, 11025, "float32"),  # param_extension: 指定通道数和样本数测试
    ])
    def test_wav_encode_decode_roundtrip(self, audio_shape, sample_rate, desired_channels, desired_samples, dtype):
        """Test roundtrip consistency of encode_wav and decode_wav."""
        # Generate test audio data
        num_samples, num_channels = audio_shape
        audio = self.generate_random_audio(audio_shape)
        
        # Encode to WAV
        audio_tensor = tf.constant(audio, dtype=getattr(tf, dtype))
        sample_rate_tensor = tf.constant(sample_rate, dtype=tf.int32)
        
        wav_data = gen_audio_ops.encode_wav(audio_tensor, sample_rate_tensor)
        
        # Decode from WAV
        decoded = gen_audio_ops.decode_wav(
            contents=wav_data,
            desired_channels=desired_channels,
            desired_samples=desired_samples
        )
        
        decoded_audio, decoded_sample_rate = decoded
        
        # Weak assertions (round 1)
        # 1. Shape assertion
        if desired_samples == -1:
            expected_samples = num_samples
        else:
            expected_samples = min(desired_samples, num_samples)
        
        if desired_channels == -1:
            expected_channels = num_channels
        else:
            expected_channels = desired_channels
        
        assert decoded_audio.shape == (expected_samples, expected_channels), \
            f"Expected shape ({expected_samples}, {expected_channels}), got {decoded_audio.shape}"
        
        # 2. Dtype assertion
        assert decoded_audio.dtype == tf.float32, \
            f"Expected dtype float32, got {decoded_audio.dtype}"
        
        # 3. Sample rate assertion
        assert decoded_sample_rate == sample_rate, \
            f"Expected sample rate {sample_rate}, got {decoded_sample_rate}"
        
        # 4. Finite values assertion
        decoded_audio_np = decoded_audio.numpy()
        assert np.all(np.isfinite(decoded_audio_np)), \
            "Decoded audio contains non-finite values"
        
        # 5. Basic property assertions
        # Audio values should be in valid range [-1, 1] after decode
        assert np.all(decoded_audio_np >= -1.0) and np.all(decoded_audio_np <= 1.0), \
            f"Decoded audio values out of range [-1, 1]. Min: {np.min(decoded_audio_np)}, Max: {np.max(decoded_audio_np)}"
        
        # Check that we didn't lose all information
        assert decoded_audio_np.size > 0, "Decoded audio should not be empty"
        
        # For the case where desired_channels/desired_samples don't truncate,
        # we can check basic consistency
        if desired_channels == -1 and desired_samples == -1:
            # The audio should be similar (allowing for WAV encoding/decoding precision loss)
            # We'll do a weak check here - just ensure it's not completely different
            original_flat = audio.flatten()
            decoded_flat = decoded_audio_np.flatten()
            
            # Simple correlation check (weak)
            if len(original_flat) == len(decoded_flat):
                # Check that they're not completely uncorrelated
                correlation = np.corrcoef(original_flat, decoded_flat)[0, 1]
                assert not np.isnan(correlation), "Correlation is NaN"
                # Weak assertion: correlation should not be exactly 0 for non-zero input
                if np.any(original_flat != 0):
                    assert abs(correlation) > 0.01, \
                        f"Audio lost too much information in roundtrip. Correlation: {correlation}"
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize("spectrogram_shape,sample_rate,upper_frequency_limit,lower_frequency_limit,filterbank_channel_count,dct_coefficient_count,dtype", [
        ([100, 257], 16000, 4000.0, 20.0, 40, 13, "float32"),
        ([50, 129], 8000, 2000.0, 100.0, 20, 20, "float32"),  # param_extension: 不同MFCC参数配置测试
    ])
    def test_mfcc_feature_extraction(self, spectrogram_shape, sample_rate, upper_frequency_limit, 
                                    lower_frequency_limit, filterbank_channel_count, 
                                    dct_coefficient_count, dtype):
        """Test MFCC feature extraction correctness."""
        # Generate random spectrogram data (magnitude squared as required)
        num_frames, num_freq_bins = spectrogram_shape
        spectrogram = np.random.uniform(0.1, 10.0, spectrogram_shape).astype(np.float32)
        
        # Convert to tensor
        spectrogram_tensor = tf.constant(spectrogram, dtype=getattr(tf, dtype))
        sample_rate_tensor = tf.constant(sample_rate, dtype=tf.int32)
        
        # Call the MFCC function
        mfcc_features = gen_audio_ops.mfcc(
            spectrogram=spectrogram_tensor,
            sample_rate=sample_rate_tensor,
            upper_frequency_limit=upper_frequency_limit,
            lower_frequency_limit=lower_frequency_limit,
            filterbank_channel_count=filterbank_channel_count,
            dct_coefficient_count=dct_coefficient_count
        )
        
        # Weak assertions (round 1)
        # 1. Shape assertion
        expected_shape = (num_frames, dct_coefficient_count)
        assert mfcc_features.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {mfcc_features.shape}"
        
        # 2. Dtype assertion
        assert mfcc_features.dtype == tf.float32, \
            f"Expected dtype float32, got {mfcc_features.dtype}"
        
        # 3. Finite values assertion
        mfcc_np = mfcc_features.numpy()
        assert np.all(np.isfinite(mfcc_np)), \
            "MFCC features contain non-finite values"
        
        # 4. Basic property assertions
        # MFCC features should have reasonable range
        # They can be positive or negative, but shouldn't be extreme
        mfcc_abs = np.abs(mfcc_np)
        assert np.all(mfcc_abs < 1e6), \
            f"MFCC values too large. Max absolute value: {np.max(mfcc_abs)}"
        
        # Check that output is not all zeros (unless input is pathological)
        assert np.any(mfcc_np != 0), \
            "MFCC features should not be all zeros for non-zero input"
        
        # Check that dimensions match expectations
        assert mfcc_np.shape[0] == num_frames, \
            f"Number of frames mismatch: expected {num_frames}, got {mfcc_np.shape[0]}"
        
        assert mfcc_np.shape[1] == dct_coefficient_count, \
            f"DCT coefficient count mismatch: expected {dct_coefficient_count}, got {mfcc_np.shape[1]}"
        
        # Basic statistical properties
        mfcc_mean = np.mean(mfcc_np)
        mfcc_std = np.std(mfcc_np)
        
        # MFCC features often have mean around 0 (after mean normalization in some implementations)
        # This is a weak check - just ensure they're not extremely biased
        assert abs(mfcc_mean) < 100.0, \
            f"MFCC mean too large: {mfcc_mean}"
        
        # Standard deviation should be reasonable
        assert 0.1 < mfcc_std < 100.0, \
            f"MFCC standard deviation out of reasonable range: {mfcc_std}"
        
        # Check that different frames produce different features (not all identical)
        if num_frames > 1:
            first_frame = mfcc_np[0, :]
            second_frame = mfcc_np[1, :]
            frames_different = not np.allclose(first_frame, second_frame, rtol=1e-5, atol=1e-5)
            # This is a weak assertion - just log if they're identical
            if not frames_different:
                print(f"Warning: First two MFCC frames are identical for shape {spectrogram_shape}")
        
        # Parameter validation checks
        assert upper_frequency_limit > lower_frequency_limit, \
            f"Upper frequency limit ({upper_frequency_limit}) should be greater than lower limit ({lower_frequency_limit})"
        
        assert upper_frequency_limit <= sample_rate / 2, \
            f"Upper frequency limit ({upper_frequency_limit}) should be <= Nyquist frequency ({sample_rate / 2})"
        
        assert filterbank_channel_count > 0, \
            f"Filterbank channel count should be positive, got {filterbank_channel_count}"
        
        assert dct_coefficient_count > 0, \
            f"DCT coefficient count should be positive, got {dct_coefficient_count}"
        
        assert dct_coefficient_count <= filterbank_channel_count, \
            f"DCT coefficient count ({dct_coefficient_count}) should be <= filterbank channels ({filterbank_channel_count})"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# Placeholder for 音频数据边界值处理 (deferred)
# This block will be replaced in later rounds
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# Placeholder for 数据类型和形状错误处理 (deferred)
# This block will be replaced in later rounds
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
# Additional test functions and cleanup
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====