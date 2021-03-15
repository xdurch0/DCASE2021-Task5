"""Code to build the Keras model.

"""
from typing import Tuple, Union

import librosa
import numpy as np
import tensorflow as tf
tfkl = tf.keras.layers


class LogMel(tfkl.Layer):
    """Compute Mel spectrograms and apply logarithmic compression."""

    def __init__(self, n_fft: int, hop_len: int, sr: int, pad: bool = True,
                 **kwargs):
        """Prepare variables for conversion.

        Parameters:
            n_fft: Size of FFT window.
            hop_len: Hop size between FFT applications.
            sr: Sampling rate of audio.
            pad: Whether to pad first/last FFT windows. This means frames will
                 be "centered" around time instead of "left-aligned".
            kwargs: Arguments for tfkl.Layer.

        """
        super().__init__(**kwargs)
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.pad = pad

        to_mel = librosa.filters.mel(sr, n_fft).T

        self.mel_matrix = tf.Variable(initial_value=to_mel,
                                      trainable=self.trainable,
                                      name=self.name + "_weights")
        self.compression = tf.Variable(
            initial_value=tf.constant(1e-8, dtype=tf.float32),
            trainable=self.trainable, name=self.name + "_compression")

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
        logmel = tf.math.log(mel + self.compression)

        return logmel


class SincConv(tfkl.Layer):
    """SincNet layer."""

    def __init__(self, filters: int, kernel_size: int, strides: int,
                 padding: str, normalize: bool = True, mel_init: bool = True,
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

        return bandpass


def baseline_block(inp: tf.Tensor, filters: int, dims: int,
                   scope: str = "") -> tf.Tensor:
    """Calculate a simple convolution block.

    Parameters:
        inp: The input to the block.
        filters: Number of filters in the convolution.
        dims: 2 or 1, for 2D or 1D convolution/pooling.
        scope: String to append to component names.

    Returns:
        pool: Result of max pooling.

    """
    if dims == 2:
        conv_fn = tfkl.Conv2D
        pool_fn = tfkl.MaxPool2D
    else:
        conv_fn = tfkl.Conv1D
        pool_fn = tfkl.MaxPool1D

    conv = conv_fn(filters, 3, padding="same", name=scope + "_conv")(inp)
    bn = tfkl.BatchNormalization(name=scope + "_bn")(conv)
    activation = tfkl.ReLU(name=scope + "_relu")(bn)
    pool = pool_fn(2, padding="same", name=scope + "_pool")(activation)
    return pool


def create_baseline_model(conf) -> tf.keras.Model:
    """Create a simple model structure for the Prototypical Network.

    Returns:
        The built model.

    """
    dims = conf.model.dims
    if dims not in [1, 2]:
        raise ValueError("Model only understands dims of 1 or 2.")

    inp = tf.keras.Input(shape=(None, 1))
    if conf.model.preprocess == "mel":
        preprocessed = LogMel(conf.features.n_fft, conf.features.hop_mel,
                              conf.features.sr, trainable=False,
                              name="logmel")(inp)
    elif conf.model.preprocess == "sinc":
        preprocessed = SincConv(conf.features.n_mels, conf.features.n_fft,
                                conf.features.hop_mel, "same",
                                name="sinc")(inp)
    else:
        raise ValueError("Invalid preprocessing specified.")

    if dims == 2:
        preprocessed = tfkl.Reshape((-1, 128, 1),
                                    name="add_channels")(preprocessed)

    b1 = baseline_block(preprocessed, 128, dims, scope="block1")
    b2 = baseline_block(b1, 128, dims, scope="block2")
    b3 = baseline_block(b2, 128, dims, scope="block3")
    b4 = baseline_block(b3, 128, dims, scope="block4")
    flat = tfkl.Flatten(name="flatten")(b4)

    model = BaselineProtonet(inp, flat, name="protonet")
    return model


class BaselineProtonet(tf.keras.Model):
    """Class for Functional model with custom training/evaluation step.

    Assumes a zipped dataset as input where each zip-element contains data for
    one class. This maps all inputs to embeddings, splits it into support and
    query sets, computes one prototype per class and then the distance of all
    query points to all prototypes.
    (Negative) Distances are converted to a probability and cross-entropy
    between this distribution and the true classes is used as a loss function.

    """
    @staticmethod
    def process_batch_input(data_batch: tuple,
                            k_way: Union[int, None] = None) -> Tuple[tf.Tensor,
                                                                     int,
                                                                     tf.Tensor,
                                                                     tf.Tensor]:
        """Stack zipped data batches into a single one.

        Parameters:
            data_batch: Tuple of data batches, one per class.
            k_way: If given, sample this many classes randomly out of all the
                  ones available. Note: This will replace n_classes in the
                  output!

        Returns:
            inputs_stacked: Single tensor where first dimension is
                            n_classes*batch_size_per_class.
            n_classes: How many classes there are in the output.
            n_support: Size of support sets.
            n_query: Size of query sets.

        """
        n_classes = len(data_batch)
        # TODO support (lol) different configurations
        n_support = tf.shape(data_batch[0])[0] // 2
        n_query = n_support

        if k_way is not None:
            inputs_stacked = tf.stack(data_batch, axis=0)
            class_picks = tf.random.shuffle(
                tf.range(n_classes, dtype=tf.int32))[:k_way]
            inputs_stacked = tf.gather(inputs_stacked, class_picks)

            feature_shape = tf.shape(inputs_stacked)[-2:]
            stacked_shape = tf.concat([[k_way * (n_support + n_query)],
                                       feature_shape], axis=0)
            inputs_stacked = tf.reshape(inputs_stacked, stacked_shape)

            n_classes = k_way
        else:
            inputs_stacked = tf.concat(data_batch, axis=0)

        return inputs_stacked, n_classes, n_support, n_query

    def proto_compute_loss(self, inputs_stacked: tf.Tensor,
                           n_classes: Union[int, tf.Tensor],
                           n_support: Union[int, tf.Tensor],
                           n_query: Union[int, tf.Tensor],
                           training: bool = False) -> Tuple[tf.Tensor,
                                                            tf.Tensor,
                                                            tf.Tensor]:
        """Compute the training loss for the Prototypical Network.

        Parameters:
            inputs_stacked: As returned by process_batch_input.
            n_classes: See above.
            n_support: See above.
            n_query: See above!!!!!!1
            training: Whether to run model in training mode (batch norm etc.).

        Returns:
            loss: Loss on this input batch.
            logits: Logits returned by the model.
            labels: Labels as constructed from the batch information.

        """
        embeddings_stacked = self(inputs_stacked, training=training)
        # assumes that embeddings are 1D, so stacked thingy is a
        #  matrix (b x d)
        embeddings_per_class = tf.reshape(
            embeddings_stacked, [n_classes, n_support + n_query, -1])
        support_set = embeddings_per_class[:, :n_support]
        query_set = embeddings_per_class[:, n_support:]
        query_set = tf.reshape(query_set, [n_classes * n_query, -1])

        # n_classes x d
        prototypes = tf.reduce_mean(support_set, axis=1)

        # distance matrix
        # for each element in the n_classes*n_query x d query_set, compute
        #  euclidean distance to each element in the n_classes x d
        #  prototypes
        # result could be n_classes*n_query x n_classes
        # can broadcast prototypes over first dim of query_set and insert
        #  one axis in query_set (axis 1).
        euclidean_dists = tf.norm(query_set[:, None] - prototypes[None],
                                  axis=-1)
        logits = -1 * euclidean_dists

        labels = tf.repeat(tf.range(n_classes, dtype=tf.int32),
                           repeats=[n_query])

        loss = self.compiled_loss(labels, logits,
                                  regularization_losses=self.losses)

        return loss, logits, labels

    def train_step(self, data: tuple) -> dict:
        """Perform a single training step given a batch of data.

        Parameters:
            data: Tuple of tensors; one element per class. Each element is a
                  3D tensor batch x time x channels (features).

        Returns:
            Dictionary of current metrics.

        """
        # process input as one batch of size
        #  b = n_classes * (n_support + n_query)
        inputs_stacked, n_classes, n_support, n_query = \
            self.process_batch_input(data, k_way=self.k_way)

        with tf.GradientTape() as tape:
            loss, logits, labels = self.proto_compute_loss(
                inputs_stacked, n_classes, n_support, n_query, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: tuple) -> dict:
        """Perform a single test/evaluation step given a batch of data.

        Parameters:
            data: Tuple of tensors; one element per class. Each element is a
                  3D tensor batch x time x channels (features).

        Returns:
            Dictionary of current metrics.

        """
        inputs_stacked, n_classes, n_support, n_query = \
            self.process_batch_input(data)

        loss, logits, labels = self.proto_compute_loss(
            inputs_stacked, n_classes, n_support, n_query, training=False)

        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def set_k_way(self, k_way: int):
        """Set k-way parameter for training the model.

        Should probably not be done like this, but I don't think we can
        interfere with the Functional __init__ easily.

        Parameters:
            k_way: The k-way setting to use. Each training step will sample this
                  many classes out of all the available ones.

        """
        self.k_way = k_way
