"""Code to build the Keras model.

"""
from typing import Tuple, Union, Optional

import tensorflow as tf
from omegaconf import DictConfig

from .layers import LogMel, SincConv, PCENCompression

tfkl = tf.keras.layers


def baseline_block(inp: tf.Tensor,
                   filters: int,
                   dims: int,
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
    activation = tfkl.Activation(tf.nn.swish, name=scope + "_relu")(bn)
    pool = pool_fn(2, padding="same", name=scope + "_pool")(activation)
    return pool


def create_baseline_model(conf: DictConfig,
                          print_summary: bool = False) -> tf.keras.Model:
    """Create a simple model structure for the Prototypical Network.

    Parameters:
        conf: hydra config object.
        print_summary: If True, print the model summary for information.

    Returns:
        The built model.

    """
    dims = conf.model.dims
    if dims not in [1, 2]:
        raise ValueError("Model only understands dims of 1 or 2.")

    if conf.features.type == "raw":
        time_shape = int(round(conf.features.seg_len * conf.features.sr))
        inp = tf.keras.Input(shape=(time_shape, 1))

        if conf.model.preprocess == "mel":
            preprocessed = LogMel(conf.features.n_mels,
                                  conf.features.n_fft,
                                  conf.features.hop_mel,
                                  conf.features.sr,
                                  name="logmel")(inp)

        elif conf.model.preprocess == "sinc":
            preprocessed = SincConv(conf.features.n_mels,
                                    conf.features.n_fft,
                                    conf.features.hop_mel,
                                    padding="same",
                                    name="sinc")(inp)

        else:
            raise ValueError("Invalid preprocessing specified.")

    elif conf.features.type == "pcen_lowpass":
        fps = conf.features.sr / conf.features.hop_mel
        time_shape = int(round(conf.features.seg_len * fps))
        inp = tf.keras.Input(shape=(time_shape, 2*conf.features.n_mels))
        preprocessed = PCENCompression(n_channels=conf.features.n_mels,
                                       gain=conf.features.gain,
                                       power=conf.features.power,
                                       bias=conf.features.bias,
                                       eps=conf.features.eps,
                                       name="pcen_compress")(inp)

    else:  # PCEN or Mel
        fps = conf.features.sr / conf.features.hop_mel
        time_shape = int(round(conf.features.seg_len * fps))
        inp = tf.keras.Input(shape=(time_shape, conf.features.n_mels))
        preprocessed = inp

    if dims == 2:
        preprocessed = tfkl.Reshape((-1, conf.features.n_mels, 1),
                                    name="add_channels")(preprocessed)

    preprocessed = tfkl.BatchNormalization(name="input_norm")(preprocessed)

    b1 = baseline_block(preprocessed, 128, dims, scope="block1")
    b2 = baseline_block(b1, 128, dims, scope="block2")
    b3 = baseline_block(b2, 128, dims, scope="block3")
    b4 = baseline_block(b3, 128, dims, scope="block4")

    if conf.model.pool == "all":
        b4 = tfkl.GlobalMaxPool2D(name="global_pool_all")(b4)
    elif conf.model.pool == "time":
        b4 = tfkl.Lambda(lambda x: tf.reduce_max(x, axis=1),
                         name="global_pool_time")(b4)
    elif conf.model.pool == "freqs":
        b4 = tfkl.Lambda(lambda x: tf.reduce_max(x, axis=2),
                         name="global_pool_freqs")(b4)

    flat = tfkl.Flatten(name="flatten")(b4)

    model = BaselineProtonet(inp, flat,
                             n_support=conf.train.n_shot,
                             n_query=conf.train.n_query,
                             distance_fn=conf.model.distance_fn,
                             k_way=conf.train.k_way,
                             binary_training=conf.train.binary,
                             name="protonet")

    if print_summary:
        print("---Model Summary---")
        print(model.summary())

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
    def __init__(self,
                 inputs: tf.Tensor,
                 outputs: tf.Tensor,
                 n_support: int,
                 n_query: int,
                 distance_fn: str,
                 k_way: Optional[int] = None,
                 binary_training: bool = False,
                 **kwargs):
        """Thin wrapper around the Functional __init__.

        Parameters:
            inputs: Input to the Model (tf.keras.Input).
            outputs: Output tensor of the model.
            n_support: Size of support set.
            n_query: Size of query set.
            distance_fn: Identifier for distance function to use for
                         classification.
            k_way: If given, number of classes to sample for each training step.
                   If not given, all classes will be used.
            binary_training: If True, training will be done in a binary fashion
                             by randomly choosing a "true" class each iteration.

        """
        super().__init__(inputs, outputs, **kwargs)
        self.k_way = k_way
        self.n_support = n_support
        self.n_query = n_query
        self.binary_training = binary_training

        if distance_fn == "euclid":
            self.distance_fn = lambda x, y: tf.norm(x - y, axis=-1)
        elif distance_fn == "euclid_squared":
            self.distance_fn = lambda x, y: tf.reduce_mean((x - y)**2, axis=-1)
        else:
            raise ValueError("Invalid distance_fn specified: "
                             "{}".format(distance_fn))

    def process_batch_input(self,
                            data_batch: tuple,
                            k_way: Union[int, None] = None) -> Tuple[tf.Tensor,
                                                                     int]:
        """Stack zipped data batches into a single one.

        Parameters:
            data_batch: Tuple of data batches, one per class.
            k_way: If given > 0, sample this many classes randomly out of all
                   the ones available. Note: This will replace n_classes in the
                   output!

        Returns:
            inputs_stacked: Single tensor where first dimension is
                            n_classes*batch_size_per_class.
            n_classes: How many classes there are in the output.

        """
        n_classes = len(data_batch)

        if k_way:
            inputs_stacked = tf.stack(data_batch, axis=0)
            class_picks = tf.random.shuffle(
                tf.range(n_classes, dtype=tf.int32))[:k_way]
            inputs_stacked = tf.gather(inputs_stacked, class_picks)

            feature_shape = tf.shape(inputs_stacked)[-2:]
            stacked_shape = tf.concat(
                [[k_way * (self.n_support + self.n_query)], feature_shape],
                axis=0)
            inputs_stacked = tf.reshape(inputs_stacked, stacked_shape)

            n_classes = k_way
        else:
            inputs_stacked = tf.concat(data_batch, axis=0)

        return inputs_stacked, n_classes

    def proto_compute_loss(self,
                           inputs_stacked: tf.Tensor,
                           n_classes: Union[int, tf.Tensor],
                           training: bool = False) -> Tuple[tf.Tensor,
                                                            tf.Tensor,
                                                            tf.Tensor]:
        """Compute the training loss for the Prototypical Network.

        Parameters:
            inputs_stacked: As returned by process_batch_input.
            n_classes: See above.
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
            embeddings_stacked, [n_classes, self.n_support + self.n_query, -1])
        support_set = embeddings_per_class[:, :self.n_support]
        query_set = embeddings_per_class[:, self.n_support:]
        query_set = tf.reshape(query_set, [n_classes * self.n_query, -1])

        # n_classes x d
        prototypes = tf.reduce_mean(support_set, axis=1)

        labels = tf.repeat(tf.range(n_classes, dtype=tf.int32),
                           repeats=[self.n_query])
        labels_onehot = tf.one_hot(labels, depth=n_classes)
        if self.binary_training:
            # we need to:
            # 1. pick a random class to be the "one" (vs all others)
            # 2. average the prototypes for all the other classes to a single
            #    "negative" prototype
            # 3. modify the labels such the "one" class gets 1, others get 0
            chosen_class = tf.random.uniform((), maxval=n_classes,
                                             dtype=tf.int32)

            chosen_onehot = tf.cast(tf.one_hot(chosen_class, depth=n_classes),
                                    tf.bool)
            chosen_others = tf.math.logical_not(chosen_onehot)

            positive_prototype = prototypes[chosen_class]
            negative_prototypes = tf.boolean_mask(prototypes, chosen_others,
                                                  axis=0)
            negative_prototype = tf.reduce_mean(negative_prototypes, axis=0)

            prototypes = tf.stack([negative_prototype, positive_prototype])

            labels = tf.where(labels == chosen_class, 1, 0)
            labels_onehot = tf.one_hot(labels, depth=2)

        # distance matrix
        # for each element in the n_classes*n_query x d query_set, compute
        #  euclidean distance to each element in the n_classes x d
        #  prototypes
        # result could be n_classes*n_query x n_classes
        # can broadcast prototypes over first dim of query_set and insert
        #  one axis in query_set (axis 1).
        distances = self.compute_distance(query_set, prototypes)
        logits = -distances

        loss = self.compiled_loss(labels_onehot, logits,
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
        inputs_stacked, n_classes = self.process_batch_input(data,
                                                             k_way=self.k_way)

        with tf.GradientTape() as tape:
            loss, logits, labels = self.proto_compute_loss(
                inputs_stacked, n_classes, training=True)

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
        inputs_stacked, n_classes = self.process_batch_input(data)

        loss, logits, labels = self.proto_compute_loss(
            inputs_stacked, n_classes, training=False)

        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def compute_distance(self,
                         queries: tf.Tensor,
                         prototypes: tf.Tensor) -> tf.Tensor:
        """Compute distance matrix between queries and prototypes.

        Parameters:
            queries: n x d.
            prototypes: k x d.

        Returns:
            n x k distance matrix where element (i, j) is the distance between
            query element i and prototype j.
        """
        return self.distance_fn(queries[:, None], prototypes[None])
