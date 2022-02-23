import crepe
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------------------------------------------------
# VARIABLES
# ---------------------------------------------------------------------------------------------------------------------
CREPE_SAMPLE_RATE = 16000
CREPE_FRAME_SIZE = 1024

# ---------------------------------------------------------------------------------------------------------------------
# LOAD A WAV FILE AND PERFORM CREPE PREDICTION
# ---------------------------------------------------------------------------------------------------------------------
audio, sr = librosa.load('AUDIO_TEST/scala.wav', res_type='kaiser_fast', mono=True)
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

# ---------------------------------------------------------------------------------------------------------------------
# PLOT PREDICTION
# ---------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.scatter(time, frequency)
plt.show()




def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)


def get_framed_lengths(input_length, frame_size, hop_size, padding='center'):
  """Give a strided framing, such as tf.signal.frame, gives output lengths.
  Args:
    input_length: Original length along the dimension to be framed.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.
  Returns:
    n_frames: Number of frames left after striding.
    padded_length: Length of the padded signal before striding.
  """
  # Use numpy since this function isn't used dynamically.
  def get_n_frames(length):
    return int(np.floor((length - frame_size) / hop_size)) + 1

  if padding == 'valid':
    padded_length = input_length
    n_frames = get_n_frames(input_length)

  elif padding == 'center':
    padded_length = input_length + frame_size
    n_frames = get_n_frames(padded_length)

  elif padding == 'same':
    n_frames = int(np.ceil(input_length / hop_size))
    padded_length = (n_frames - 1) * hop_size + frame_size

  return n_frames, padded_length


def pad(x, frame_size, hop_size, padding='center',
        axis=1, mode='CONSTANT', constant_values=0):
  """Pad a tensor for strided framing such as tf.signal.frame.
  Args:
    x: Tensor to pad, any shape.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.
    axis: Axis along which to pad `x`.
    mode: Padding mode for tf.pad(). One of "CONSTANT", "REFLECT", or
      "SYMMETRIC" (case-insensitive).
    constant_values: Passthrough kwarg for tf.pad().
  Returns:
    A padded version of `x` along axis. Output sizes can be computed separately
      with strided_lengths.
  """
  x = tf_float32(x)

  if padding == 'valid':
    return x

  if hop_size > frame_size:
    raise ValueError(f'During padding, frame_size ({frame_size})'
                     f' must be greater than hop_size ({hop_size}).')

  if len(x.shape) <= 1:
    axis = 0

  n_t = x.shape[axis]
  _, n_t_padded = get_framed_lengths(n_t, frame_size, hop_size, padding)
  pads = [[0, 0] for _ in range(len(x.shape))]

  if padding == 'same':
    pad_amount = int(n_t_padded - n_t)
    pads[axis] = [0, pad_amount]

  elif padding == 'center':
    pad_amount = int(frame_size // 2)  # Symmetric even padding like librosa.
    pads[axis] = [pad_amount, pad_amount]

  else:
    raise ValueError('`padding` must be one of [\'center\', \'same\''
                     f'\'valid\'], received ({padding}).')

  return tf.pad(x, pads, mode=mode, constant_values=constant_values)


def compute_f0(audio, frame_rate, viterbi=True, padding='center'):
    """Fundamental frequency (f0) estimate using CREPE.
  This function is non-differentiable and takes input as a numpy array.
  Args:
    audio: Numpy ndarray of single audio (16kHz) example. Shape [audio_length,].
    frame_rate: Rate of f0 frames in Hz.
    viterbi: Use Viterbi decoding to estimate f0.
    padding: Apply zero-padding for centered frames.
      'same', 'valid', or 'center'.
  Returns:
    f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
    f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
  """
    sample_rate = CREPE_SAMPLE_RATE
    crepe_step_size = 1000 / frame_rate  # milliseconds
    hop_size = sample_rate // frame_rate

    audio = pad(audio, CREPE_FRAME_SIZE, hop_size, padding)
    audio = np.asarray(audio)

    # Compute f0 with crepe.
    _, f0_hz, f0_confidence, _ = crepe.predict(
        audio,
        sr=sample_rate,
        viterbi=viterbi,
        step_size=crepe_step_size,
        center=False,
        verbose=0)

    # Postprocessing.
    f0_hz = f0_hz.astype(np.float32)
    f0_confidence = f0_confidence.astype(np.float32)
    f0_confidence = np.nan_to_num(f0_confidence)  # Set nans to 0 in confidence

    return f0_hz, f0_confidence


f0_hz, f0_confidence = compute_f0(audio, frame_rate=250, viterbi=True, padding='center')


print('debug')
