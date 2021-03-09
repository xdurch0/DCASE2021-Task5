import tensorflow as tf
tfkl = tf.keras.layers


def baseline_block(inp):
    conv = tfkl.Conv2D(128, 3, padding="same")(inp)
    bn = tfkl.BatchNormalization()(conv)
    activation = tfkl.ReLU()(bn)
    pool = tfkl.MaxPool2D(2, padding="same")(activation)
    return pool


def create_baseline_model():
    inp = tf.keras.Input(shape=(17, 128))
    inp_channel = tfkl.Reshape((17, 128, 1))(inp)
    b1 = baseline_block(inp_channel)
    b2 = baseline_block(b1)
    b3 = baseline_block(b2)
    b4 = baseline_block(b3)
    flat = tfkl.Flatten()(b4)

    model = BaselineProtonet(inp, flat)
    return model


class BaselineProtonet(tf.keras.Model):
    def train_step(self, data):
        # data is a batch of tuples, 1 element per class
        # each element is again a tuple (input_batch, label_batch)
        # TODO could remove labels, as element i always has label i
        # we should extract the inputs, stack them into a big batch and
        #  run all of them through the model.
        # OR always take first half of each batch, stack them into support batch
        #  and take second half, stack those into query batch
        # could also do this division after running through the model, but could
        #  be more annoying (was ez lol)
        # then compute avg of the support embeddings per class
        # then compute euclidean distance between each query embedding and
        # all support embeddings
        # then compute softmax
        # then compute cross-entropy wrt real labels
        inputs = tuple(tup[0] for tup in data)
        labels = tuple(tup[1] for tup in data)

        n_classes = len(inputs)
        # TODO support (lol) different configurations
        n_support = tf.shape(inputs[0])[0] // 2
        n_query = n_support

        # process as one batch of size b = n_classes * (n_support + n_query)
        inputs_stacked = tf.concat(inputs, axis=0)
        with tf.GradientTape() as tape:
            embeddings_stacked = self(inputs_stacked, training=True)
            # assumes that embeddings are 1D, so stacked thingy is a
            #  matrix (b x d)
            embeddings_per_class = tf.reshape(
                embeddings_stacked, [n_classes, n_support + n_query, -1])
            support_set = embeddings_per_class[:, :n_support]
            query_set = embeddings_per_class[:, n_support:]
            query_set = tf.reshape(query_set, [n_classes*n_query, -1])

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
            # class_probs = tf.nn.softmax(logits, axis=-1)

            labels = tf.repeat(tf.range(n_classes, dtype=tf.int32),
                               repeats=[n_query])

            # do note that we use the *negative* distances as logits
            # loss = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=labels, logits=logits)
            loss = self.compiled_loss(labels, logits,
                                      regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(labels, logits)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
