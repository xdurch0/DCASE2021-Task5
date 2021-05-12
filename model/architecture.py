from typing import Optional, Union

import tensorflow as tf
from omegaconf import DictConfig

from .layers import LogMel, SincConv, PCENCompression
from .model import (euclidean_distance, squared_euclidean_distance,
                    BaselineProtonet)

tfkl = tf.keras.layers


def squeeze_excite(inp: tf.Tensor,
                   reduction_factor: int = 8,
                   scope: str = ""):
    # TODO adapt to 1D?
    inp_filters = inp.shape[-1]
    reduction_filters = inp_filters // reduction_factor

    pooled = tfkl.GlobalAvgPool2D(name=scope + "_pool")(inp)
    compressed = tfkl.Dense(reduction_filters, name=scope + "_dense1")(pooled)
    acted = tfkl.Activation(tf.nn.swish, name=scope + "_act")(compressed)
    expand = tfkl.Dense(inp_filters, name=scope + "_dense2")(acted)
    gate = tfkl.Activation(tf.nn.sigmoid, name=scope + "_sigmoid")(expand)
    gate = tfkl.Reshape((1, 1, inp_filters), name=scope + "_reshape")(gate)

    return inp * gate


def baseline_block(inp: tf.Tensor,
                   filters: int,
                   filter_size: int,
                   dims: int,
                   activation: Optional[str] = None,
                   pool_size: Union[int, tuple] = 2,
                   use_se: bool = False,
                   scope: str = "") -> tf.Tensor:
    """Calculate a simple convolution block.

    Parameters:
        inp: The input to the block.
        filters: Number of filters in the convolution.
        filter_size: Size of convolutional filters.
        dims: 2 or 1, for 2D or 1D convolution/pooling.
        activation: None or string identifier for activation.
        pool_size: Size of max pooling region. If int, use quadratic window.
        use_se: If True, apply squeeze-and-excite block after activation.
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

    conv = conv_fn(filters,
                   filter_size,
                   padding="same",
                   name=scope + "_conv")(inp)
    bn = tfkl.BatchNormalization(name=scope + "_bn")(conv)

    if activation == "swish":
        activated = tfkl.Activation(tf.nn.swish,
                                    name=scope + "_activation")(bn)
    elif activation is None:
        activated = bn
    else:
        raise ValueError("Invalid activation {}".format(activation))

    if use_se:
        activated = squeeze_excite(activated, scope=scope + "_se")

    if ((isinstance(pool_size, int) and pool_size > 1)
            or isinstance(pool_size, tuple)):
        return pool_fn(pool_size,
                       padding="same",
                       name=scope + "_pool")(activated)
    else:
        return activated


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

    inp, preprocessed = preprocessing_block(conf)
    b4 = body_block(preprocessed, conf)
    distance_model = distance_block(b4, conf)

    model = BaselineProtonet(inp, b4,
                             n_support=conf.train.n_shot,
                             n_query=conf.train.n_query,
                             distance_fn=distance_model,
                             k_way=conf.train.k_way,
                             binary_training=conf.train.binary,
                             cycle_binary=conf.train.cycle_binary,
                             name="protonet")

    if print_summary:
        print("---Model Summary---")
        print(model.summary())

        print("---Distance Model---")
        print(distance_model.summary())

    return model


def preprocessing_block(conf):
    dims = conf.model.dims

    if conf.features.type == "raw":
        time_shape = int(round(conf.features.seg_len * conf.features.sr))
        inp = tf.keras.Input(shape=(time_shape, 1),
                             name="raw_input")

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
        time_shape = int((conf.features.seg_len * fps))
        # TODO acknowledge cropping for other feature types
        inp = tf.keras.Input(shape=(time_shape - conf.model.crop,
                                    2*conf.features.n_mels),
                             name="pcen_lowpass_input")
        preprocessed = PCENCompression(n_channels=conf.features.n_mels,
                                       gain=conf.features.gain,
                                       power=conf.features.power,
                                       bias=conf.features.bias,
                                       eps=conf.features.eps,
                                       name="pcen_compress")(inp)

    else:  # PCEN or Mel
        fps = conf.features.sr / conf.features.hop_mel
        time_shape = int((conf.features.seg_len * fps))
        inp = tf.keras.Input(shape=(time_shape, conf.features.n_mels),
                             name="spectrogram_input")
        preprocessed = inp

    if dims == 2:
        preprocessed = tfkl.Reshape((-1, conf.features.n_mels, 1),
                                    name="add_channel_axis")(preprocessed)

    preprocessed = tfkl.BatchNormalization(name="input_norm")(preprocessed)

    return inp, preprocessed


def body_block(preprocessed, conf):
    dims = conf.model.dims

    b1 = baseline_block(preprocessed, 128, 3,
                        dims=dims,
                        activation="swish",
                        use_se=conf.model.squeeze_excite,
                        scope="block1")
    b2 = baseline_block(b1, 128, 3,
                        dims=dims,
                        activation="swish",
                        use_se=conf.model.squeeze_excite,
                        scope="block2")
    b3 = baseline_block(b2, 128, 3,
                        dims=dims,
                        activation="swish",
                        use_se=conf.model.squeeze_excite,
                        scope="block3")
    b4 = baseline_block(b3, 128, 3,
                        dims=dims,
                        activation="swish",
                        use_se=conf.model.squeeze_excite,
                        scope="block4")

    # TODO this does not work for 1d inputs lol
    if conf.model.pool == "all":
        b4 = tfkl.GlobalMaxPool2D(name="global_pool_all")(b4)
    elif conf.model.pool == "time":
        b4 = tfkl.Lambda(lambda x: tf.reduce_max(x, axis=1),
                         name="global_pool_time")(b4)
    elif conf.model.pool == "freqs":
        b4 = tfkl.Lambda(lambda x: tf.reduce_max(x, axis=2),
                         name="global_pool_freqs")(b4)

    return b4


def distance_block(embedding_input, conf):
    distance_inp_shape = embedding_input.shape[1:]
    distance_inp1 = tf.keras.Input(shape=distance_inp_shape,
                                   name="distance_input1")
    distance_inp2 = tf.keras.Input(shape=distance_inp_shape,
                                   name="distance_input2")
    if conf.model.distance_fn == "euclid":
        flat1 = tfkl.Flatten(name="flatten1")(distance_inp1)
        flat2 = tfkl.Flatten(name="flatten2")(distance_inp2)
        distance = tfkl.Lambda(euclidean_distance,
                               name="euclidean")([flat1, flat2])

    elif conf.model.distance_fn == "euclid_squared":
        flat1 = tfkl.Flatten(name="flatten1")(distance_inp1)
        flat2 = tfkl.Flatten(name="flatten2")(distance_inp2)
        distance = tfkl.Lambda(squared_euclidean_distance,
                               name="squared_euclidean")([flat1, flat2])

    elif conf.model.distance_fn == "euclid_weighted":
        flat1 = tfkl.Flatten(name="flatten1")(distance_inp1)
        flat2 = tfkl.Flatten(name="flatten2")(distance_inp2)
        squared_diff = tfkl.Lambda(lambda x, y: (x - y)**2,
                                   name="squared_difference")([flat1, flat2])
        weighted = tfkl.Dense(
            1, kernel_initializer=tf.keras.initializers.ones(),
            use_bias=False)(squared_diff)
        distance = tfkl.Lambda(lambda x: tf.math.sqrt(x), name="root")(weighted)

    elif conf.model.distance_fn == "mlp":
        flat1 = tfkl.Flatten(name="flatten1")(distance_inp1)
        flat2 = tfkl.Flatten(name="flatten2")(distance_inp2)
        concat = tfkl.Concatenate(name="concatenate_inputs")([flat1, flat2])

        h1 = tfkl.Dense(1024, name="dense1")(concat)
        bn1 = tfkl.BatchNormalization(name="bn1")(h1)
        a1 = tfkl.Activation(tf.nn.swish,
                             name="activation1")(bn1)

        h2 = tfkl.Dense(256, name="dense2")(a1)
        bn2 = tfkl.BatchNormalization(name="bn2")(h2)
        a2 = tfkl.Activation(tf.nn.swish,
                             name="activation2")(bn2)

        h3 = tfkl.Dense(64, name="dense3")(a2)
        bn3 = tfkl.BatchNormalization(name="bn3")(h3)
        a3 = tfkl.Activation(tf.nn.swish,
                             name="activation3")(bn3)

        h4 = tfkl.Dense(1, name="dense4")(a3)
        bn4 = tfkl.BatchNormalization(name="bn4")(h4)
        distance = tfkl.Activation(tf.nn.softplus,
                                   name="activation4")(bn4)

    elif conf.model.distance_fn == "cnn":
        concat = tfkl.Concatenate(
            name="cnn_concatenate_inputs")([distance_inp1, distance_inp2])

        # TODO will crash if dims=1
        b1 = baseline_block(concat, 64, 3,
                            dims=2,
                            activation="swish",
                            pool_size=(1, 2),
                            use_se=conf.model.squeeze_excite,
                            scope="cnn_distance_block1")
        b2 = baseline_block(b1, 64, 3,
                            dims=2,
                            activation="swish",
                            pool_size=1,
                            use_se=conf.model.squeeze_excite,
                            scope="cnn_distance_block2")
        b3 = baseline_block(b2, 64, 3,
                            dims=2,
                            activation="swish",
                            pool_size=(2, 2),
                            use_se=conf.model.squeeze_excite,
                            scope="cnn_distance_block3")
        b4 = baseline_block(b3, 64, 3,
                            dims=2,
                            activation="swish",
                            pool_size=1,
                            use_se=conf.model.squeeze_excite,
                            scope="cnn_distance_block4")

        flat = tfkl.Flatten(name="cnn_distance_flatten")(b4)
        dense = tfkl.Dense(1, name="cnn_distance_to_size_1")(flat)
        bn = tfkl.BatchNormalization(name="cnn_distance_bn")(dense)
        distance = tfkl.Activation(tf.nn.softplus,
                                   name="cnn_distance_softplus")(bn)

    else:
        raise ValueError("Invalid distance_fn specified: "
                         "{}".format(conf.model.distance_fn))

    return tf.keras.Model([distance_inp1, distance_inp2], distance,
                          name="distance_model")
