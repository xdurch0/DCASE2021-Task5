import librosa
import numpy as np
import tensorflow as tf

tfkl = tf.keras.layers


def inverse_softplus(x):
    return tf.math.log(tf.exp(x) - 1.)


class LogMel(tfkl.Layer):
    """Compute Mel spectrograms and apply logarithmic compression."""

    def __init__(self,
                 n_mels: int,
                 n_fft: int,
                 hop_len: int,
                 sr: int,
                 pad: bool = True,
                 compression: float = 1e-8,
                 **kwargs):
        """Prepare variables for conversion.

        Parameters:
            n_mels: Number of mel frequency bands.
            n_fft: Size of FFT window.
            hop_len: Hop size between FFT applications.
            sr: Sampling rate of audio.
            pad: Whether to pad first/last FFT windows. This means frames will
                 be "centered" around time instead of "left-aligned".
            compression: Additive offset for log compression.
            kwargs: Arguments for tfkl.Layer.

        """
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.pad = pad

        to_mel = librosa.filters.mel(sr, n_fft, n_mels=n_mels).T

        self.mel_matrix = tf.Variable(initial_value=to_mel,
                                      trainable=self.trainable,
                                      dtype=tf.float32,
                                      name=self.name + "_weights")
        self.compression = tf.Variable(
            initial_value=tf.ones(n_mels) * inverse_softplus(compression),
            trainable=self.trainable,
            dtype=tf.float32,
            name=self.name + "_compression")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply the layer.

        Parameters:
            inputs: Audio. Note that we assume a channel axis (size 1) even
                    though it is not used. Compatibility reasons.
            kwargs: Other arguments to Layer.call; ignored.

        Returns:
            Log-power mel spectrogram.

        """
        if self.pad:
            inputs = tf.pad(
                inputs, ((0, 0), (self.n_fft // 2, self.n_fft // 2), (0, 0)),
                mode="reflect")

        spectros = tf.signal.stft(inputs[:, :, 0], self.n_fft, self.hop_len)
        power = tf.abs(spectros) ** 2

        mel = tf.matmul(power, self.mel_matrix)
        logmel = tf.math.log(mel + tf.nn.softplus(self.compression))

        return logmel


class SincConv(tfkl.Layer):
    """SincNet layer."""

    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides: int,
                 padding: str,
                 normalize: bool = True,
                 mel_init: bool = True,
                 window: bool = True,
                 **kwargs):
        """Set up parameters for the sinc layer.

        Parameters:
            filters: How many filters to create. Must be even if mel_init is
                     used.
            kernel_size: Size of the filters in samples. Must be odd.
            strides: Hop size between filter applications.
            padding: Padding used for convolution.
            normalize: If True, normalize filters such that their maximum
                       absolute value is 1.
            mel_init: If True, initialize frequencies according to the mel
                      scale. Otherwise, frequencies are sampled uniformly
                      between 0 and the Nyquist frequency.
            window: If True, apply window to the Sinc filters.
            kwargs: Arguments for tfkl.Layer.

        """
        if not kernel_size % 2:
            raise ValueError("Kernel size must be odd.")

        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.normalize = normalize
        self.window = window
        if window:
            self.window_fn = 0.54 - 0.46*tf.cos(
                2*np.pi*tf.range(kernel_size, dtype=tf.float32) / kernel_size)

        if mel_init:
            if filters % 2:
                raise NotImplementedError("Please use even number of filters.")
            # Note: mel_frequencies has an SR of 22050 as default, which we
            # simply use here. The SR does not "make it out of this function"
            # since afterwards we deal in fractions of Nyquist frequency.
            edge_frequencies = librosa.mel_frequencies(2 * filters - 1)
            lower = np.concatenate([[0.], edge_frequencies[1::2]])
            upper = np.concatenate([lower[1:], [11025.]])
            assert len(lower) == filters
            assert len(upper) == filters

            self.f1 = tf.Variable(lower / 22050, name="freqs_lower",
                                  dtype=tf.float32)
            self.f2 = tf.Variable(upper / 22050, name="freqs_upper",
                                  dtype=tf.float32)
        else:
            self.f1 = self.add_weight(
                name="freqs_lower",
                initializer=tf.keras.initializers.RandomUniform(0., 0.5),
                shape=(filters,))
            self.f2 = self.add_weight(
                name="freqs_upper",
                initializer=tf.keras.initializers.RandomUniform(0., 0.5),
                shape=(filters,))

        # sinc(0) will return nan; this is used to fix that.
        # there is probably a better way.
        nan_killer = np.zeros(kernel_size, dtype=np.float32)
        nan_killer[kernel_size // 2] = 1.
        self.nan_killer = tf.convert_to_tensor(nan_killer[:, None])

        self.filter_space = tf.range(-(self.kernel_size // 2),
                                     self.kernel_size // 2 + 1,
                                     dtype=tf.float32)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply the layer.

        Parameters:
            inputs: Audio input. Assumed to have a channel axis (size 1).
            kwargs: Other arguments to Layer.call; ignored.

        Returns:
            Result of convolving sinc kernels with input.

        """
        kernels = self.get_kernels()[:, None, :]
        return tf.nn.conv1d(inputs, kernels, stride=self.strides,
                            padding=self.padding.upper())

    def get_kernels(self) -> tf.Tensor:
        """Get bandpass sinc kernels for convolution.

        Returns:
            kernel_size x filters matrix of bandpass filters.

        """
        def sinc(x: tf.Tensor) -> tf.Tensor:
            with_nans = tf.math.sin(x) / x
            where_nans = tf.math.is_nan(with_nans)
            return (tf.cast(where_nans, tf.float32) * self.nan_killer +
                    tf.math.multiply_no_nan(
                        with_nans, tf.cast(tf.math.logical_not(where_nans),
                                           tf.float32)))

        f1_abs = tf.abs(self.f1)
        f2_abs = f1_abs + tf.abs(self.f2 - self.f1)

        lowpass_upper = 2 * f2_abs[None, :] * sinc(
            2 * np.pi * f2_abs[None, :] * self.filter_space[:, None])
        lowpass_lower = 2 * f1_abs[None, :] * sinc(
            2 * np.pi * f1_abs[None, :] * self.filter_space[:, None])
        bandpass = lowpass_upper - lowpass_lower

        if self.normalize:
            maxvals = tf.reduce_max(tf.abs(bandpass), axis=0)
            bandpass /= maxvals
        if self.window:
            bandpass *= self.window_fn[:, None]

        return bandpass


class PCENCompression(tfkl.Layer):
    def __init__(self,
                 n_channels: int,
                 gain: float,
                 power: float,
                 bias: float,
                 eps: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.gain = tf.Variable(
            initial_value=tf.ones(n_channels) * inverse_softplus(gain),
            trainable=self.trainable,
            dtype=tf.float32,
            name=self.name + "_gain")
        self.power = tf.Variable(
            initial_value=tf.ones(n_channels) * inverse_softplus(power),
            trainable=self.trainable,
            dtype=tf.float32,
            name=self.name + "_power")
        self.bias = tf.Variable(
            initial_value=tf.ones(n_channels) * inverse_softplus(bias),
            trainable=self.trainable,
            dtype=tf.float32,
            name=self.name + "_bias")
        self.eps = tf.Variable(
            initial_value=tf.ones(n_channels) * inverse_softplus(eps),
            trainable=self.trainable,
            dtype=tf.float32,
            name=self.name + "_eps")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        gain = tf.nn.softplus(self.gain)
        power = tf.nn.softplus(self.power)
        bias = tf.nn.softplus(self.bias)
        eps = tf.nn.softplus(self.eps)

        spectro, spectro_smooth = tf.split(inputs, 2, axis=-1)
        # Adaptive gain control
        # Working in log-space gives us some stability, and a slight speedup
        smooth = tf.exp(-gain * (tf.math.log(eps) +
                                 tf.math.log1p(spectro_smooth / eps)))

        # Dynamic range compression
        # TODO add 0 cases again
        if False:  # power == 0:
            out = tf.math.log1p(spectro * smooth)
        elif False:  # bias == 0:
            out = tf.exp(power * (tf.math.log(spectro) + tf.math.log(smooth)))
        else:
            out = (bias ** power) * tf.math.expm1(
                power * tf.math.log1p(spectro * smooth / bias))

        return out
