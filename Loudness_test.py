import tensorflow as tf
import numpy as np
import librosa
from CREPE_test import *

DB_RANGE = 80.0


def nan_to_num(x, value=0.0):
    """Replace NaNs with value."""
    return tf.where(tf.math.is_nan(x), value * tf.ones_like(x), x)


def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = tf.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = tf.where(x <= 0.0, eps, x)
    return tf.math.log(safe_x)


def logb(x, base=2.0, eps=1e-5):
    """Logarithm with base as an argument."""
    return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)


def log10(x, eps=1e-5):
    """Logarithm with base 10."""
    return logb(x, base=10, eps=eps)


def power_to_db(power, ref_db=0.0, range_db=DB_RANGE, use_tf=True):
    """Converts power from linear scale to decibels."""
    # Choose library.
    maximum = tf.maximum if use_tf else np.maximum
    log_base10 = log10 if use_tf else np.log10

    # Convert to decibels.
    pmin = 10 ** -(range_db / 10.0)
    power = maximum(pmin, power)
    db = 10.0 * log_base10(power)

    # Set dynamic range.
    db -= ref_db
    db = maximum(db, -range_db)
    return db


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Non-differentiable stft using librosa, one example at a time."""
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    is_2d = (len(audio.shape) == 2)

    if pad_end:
        audio = pad(audio, frame_size, hop_size, 'same', axis=is_2d).numpy()

    def stft_fn(y):
        return librosa.stft(
            y=y, n_fft=int(frame_size), hop_length=hop_size, center=False).T

    s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
    return s


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft in tensorflow, computed in batch."""
    assert frame_size * overlap % 2.0 == 0.0

    # Remove channel dim if present.
    audio = tf_float32(audio)
    if len(audio.shape) == 3:
        audio = tf.squeeze(audio, axis=-1)

    s = tf.signal.stft(
        signals=audio,
        frame_length=int(frame_size),
        frame_step=int(frame_size * (1.0 - overlap)),
        fft_length=int(frame_size),
        pad_end=pad_end)
    return s


def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=512,
                     range_db=DB_RANGE,
                     ref_db=0.0,
                     use_tf=True,
                     padding='center'):
    """Perceptual loudness (weighted power) in dB.
  Function is differentiable if use_tf=True.
  Args:
    audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
      [audio_length,].
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The old (<v2.0.0)
      default value corresponded to white noise with amplitude=1.0 and
      n_fft=2048. With v2.0.0 it was set to 0.0 to be more consistent with power
      calculations that have a natural scale for 0 dB being amplitude=1.0.
    use_tf: Make function differentiable by using tensorflow.
    padding: 'same', 'valid', or 'center'.
  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
    # Pick tensorflow or numpy.
    lib = tf if use_tf else np
    reduce_mean = tf.reduce_mean if use_tf else np.mean
    stft_fn = stft if use_tf else stft_np

    # Make inputs tensors for tensorflow.
    frame_size = n_fft
    hop_size = sample_rate // frame_rate
    audio = pad(audio, frame_size, hop_size, padding=padding)
    audio = audio if use_tf else np.array(audio)

    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    audio = audio[lib.newaxis, :] if is_1d else audio

    # Take STFT.
    overlap = 1 - hop_size / frame_size
    s = stft_fn(audio, frame_size=frame_size, overlap=overlap, pad_end=False)

    # Compute power.
    amplitude = lib.abs(s)
    power = amplitude ** 2

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10 ** (a_weighting / 10)
    power = power * weighting

    # Average over frequencies (weighted power per a bin).
    avg_power = reduce_mean(power, axis=-1)
    loudness = power_to_db(avg_power,
                           ref_db=ref_db,
                           range_db=range_db,
                           use_tf=use_tf)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    return loudness


loudness = compute_loudness(audio,
                            sample_rate=sr,
                            frame_rate=250,
                            n_fft=512,
                            range_db=DB_RANGE,
                            ref_db=0.0,
                            use_tf=True,
                            padding='center')


plt.scatter(x=range(len(loudness)), y=np.array(loudness))
plt.tight_layout()
plt.show()

print('debug')
