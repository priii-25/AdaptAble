import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, lfilter
import noisereduce as nr
import os
import warnings

WORKING_DIR = '/working'
os.chdir(WORKING_DIR)

audio_file = '/input/monkey/Monkey Noises SFX.wav'
y, sr = librosa.load(audio_file)

def normalize_audio(audio, target_dBFS=-20.0):
    current_dBFS = 20 * np.log10(np.max(np.abs(audio)))
    dB_difference = target_dBFS - current_dBFS
    normalization_factor = 10 ** (dB_difference / 20)
    normalized_audio = audio * normalization_factor
    return normalized_audio

normalized_audio = normalize_audio(y)

eq_bands = {
    'low_rumble': (20, 100),
    'bass': (100, 200),
    'midrange': (200, 2000),
    'treble': (2000, 5000),
    'high_treble': (5000, 20000)
}

def equalize_audio(audio, sr, eq_bands):
    eq_audio = np.zeros_like(audio, dtype=np.float64)
    max_gain = 0.0

    for band_name, (low, high) in eq_bands.items():
        nyq = 0.5 * sr
        low_normalized = low / nyq
        high_normalized = min(high / nyq, 0.99)
        b, a = butter(4, [low_normalized, high_normalized], btype='band')
        band_audio = lfilter(b, a, audio)

        if band_name == 'low_rumble':
            gain = 0.2
        elif band_name == 'bass':
            gain = 0.3
        elif band_name == 'midrange':
            gain = 0.4
        elif band_name == 'treble':
            gain = 0.3
        elif band_name == 'high_treble':
            gain = 0.2

        max_gain = max(max_gain, gain)
        band_audio *= gain
        eq_audio += band_audio

    eq_audio /= max_gain
    eq_audio = np.clip(eq_audio, -1.0, 1.0)

    return eq_audio

equalized_audio = equalize_audio(normalized_audio, sr, eq_bands)

def calculate_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

TARGET_RMS = -20.0
TARGET_RMS_LINEAR = np.power(10, TARGET_RMS / 20)
current_rms = calculate_rms(equalized_audio)
normalized_rms_audio = equalized_audio * (TARGET_RMS_LINEAR / current_rms)
normalized_rms_audio = np.clip(normalized_rms_audio, -1.0, 1.0)

warnings.simplefilter(action='ignore', category=RuntimeWarning)

def patched_nonstationary_reduce_noise(y, sr, *args, **kwargs):
    from noisereduce.spectralgate.nonstationary import _non_stationary_noise_reduction
    def patched_division(x, y):
        return np.divide(x, y, out=np.zeros_like(x), where=y!=0, casting='unsafe')
    _non_stationary_noise_reduction.patched_division = patched_division
    return _non_stationary_noise_reduction(y, sr, *args, **kwargs)

try:
    reduced_noise_audio = patched_nonstationary_reduce_noise(y=normalized_rms_audio, sr=sr)
except Exception as e:
    print(f"Error during noise reduction: {str(e)}")
    reduced_noise_audio = normalized_rms_audio

onset_frames = librosa.onset.onset_detect(y=reduced_noise_audio, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
print("Onset times (trigger points):", onset_times)

y_min = min(np.min(y), np.min(reduced_noise_audio))
y_max = max(np.max(y), np.max(reduced_noise_audio))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Audio Waveform')
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim(y_min, y_max)

plt.subplot(1, 2, 2)
plt.title('Reduced Audio Waveform')
librosa.display.waveshow(reduced_noise_audio, sr=sr, alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()

sf.write('./normalized_audio_final.wav', normalized_audio, sr)
print("Audio files saved successfully.")