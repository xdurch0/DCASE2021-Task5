"""Code to build the Keras model.

"""
from typing import Tuple, Union, Optional

import numpy as np
import tensorflow as tf


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
                 distance_fn: tf.keras.Model,
                 k_way: Optional[int] = None,
                 binary_training: bool = False,
                 cycle_binary: bool = False,
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
            cycle_binary: If True, in binary training, instead of choosing one
                          class at random, each batch will cycle through picking
                          each class as "positive" in turn, averaging the
                          results in a single loss. Has no effect if
                          binary_training is False.

        """
        super().__init__(inputs, outputs, **kwargs)
        self.k_way = k_way
        self.n_support = n_support
        self.n_query = n_query
        self.binary_training = binary_training
        self.cycle_binary = cycle_binary
        self.distance_fn = distance_fn

        if binary_training:
            if cycle_binary:
                self.logit_fn = self.cycle_binary_logit_fn
            else:
                self.logit_fn = self.binary_logit_fn
        else:
            self.logit_fn = self.multiclass_logit_fn

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

        embedding_shape = tf.shape(embeddings_stacked)[1:]
        stacked_shape = tf.concat([[n_classes, self.n_support + self.n_query],
                                   embedding_shape],
                                  axis=0)
        query_set_shape = tf.concat([[n_classes * self.n_query],
                                     embedding_shape],
                                    axis=0)

        # assumes that embeddings are 1D, so stacked thingy is a
        #  matrix (b x d)
        embeddings_per_class = tf.reshape(embeddings_stacked, stacked_shape)
        support_set = embeddings_per_class[:, :self.n_support]
        query_set = embeddings_per_class[:, self.n_support:]
        query_set = tf.reshape(query_set, query_set_shape)

        # n_classes x d
        prototypes = tf.reduce_mean(support_set, axis=1)

        labels = tf.repeat(tf.range(n_classes, dtype=tf.int32),
                           repeats=[self.n_query])

        logits, labels = self.logit_fn(query_set, prototypes, labels,
                                       n_classes=n_classes)

        labels_onehot = tf.one_hot(labels, depth=tf.reduce_max(labels) + 1)

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
        queries_repeated = tf.repeat(queries,
                                     repeats=tf.shape(prototypes)[0],
                                     axis=0)

        ndims = len(prototypes.shape)
        prototypes_tiled = tf.tile(prototypes,
                                   [tf.shape(queries)[0]] + (ndims - 1) * [1])

        distances = self.distance_fn([queries_repeated, prototypes_tiled])
        return tf.reshape(distances, [tf.shape(queries)[0],
                                      tf.shape(prototypes)[0]])

    def get_probability(self,
                        positive_prototype: Union[tf.Tensor, np.ndarray],
                        negative_prototype: Union[tf.Tensor, np.ndarray],
                        query_embeddings: Union[tf.Tensor, np.ndarray]) -> list:
        """Calculate the probability of queries belonging to the positive class.

        Parameters:
            positive_prototype: 1D, size d.
            negative_prototype: 1D, size d.
            query_embeddings: 2D, n x d.

        Returns:
            probs_ops: 1D, size n; for each row in query_embeddings,
                       contains the probability that this query belongs to the
                       positive class.

        """
        prototypes = tf.stack([positive_prototype, negative_prototype], axis=0)
        distances = self.compute_distance(query_embeddings, prototypes)
        logits = -distances

        probs = tf.nn.softmax(logits, axis=-1)
        probs_pos = probs[:, 0].numpy().tolist()

        return probs_pos

    def multiclass_logit_fn(self,
                            query_set: tf.Tensor,
                            prototypes: tf.Tensor,
                            labels: tf.Tensor,
                            **_kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Logits computation for "regular" multi-class training.

        Parameters:
            query_set: The... query set, n x d.
            prototypes: Tensor of prototypes, k x d.
            labels: n elements, label for each element of query set.

        Returns:
            Logits as well as unchanged labels.

        """
        distances = self.compute_distance(query_set, prototypes)
        logits = -distances

        return logits, labels

    def binary_logit_fn_core(self,
                             query_set: tf.Tensor,
                             prototypes: tf.Tensor,
                             labels: tf.Tensor,
                             n_classes: int,
                             chosen_class: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Do binary classification 1 vs rest.

        Parameters:
            query_set: As above.
            prototypes: As above.
            labels: As above.
            n_classes: Number of classes we have. Could be taken from
                       prototypes?
            chosen_class: Class index that will be treated as positive.

        Returns:
            Classification logits as well as the 0-1 labels.

        """
        chosen_onehot = tf.cast(
            tf.one_hot(chosen_class, depth=n_classes), tf.bool)
        chosen_others = tf.math.logical_not(chosen_onehot)

        positive_prototype = prototypes[chosen_class]
        negative_prototypes = tf.boolean_mask(prototypes, chosen_others,
                                              axis=0)
        # since all classes appear the same number of times, we can
        # compute the overall prototype as mean of means
        negative_prototype = tf.reduce_mean(negative_prototypes, axis=0)

        binary_prototypes = tf.stack([negative_prototype,
                                      positive_prototype])

        labels = tf.where(labels == chosen_class, 1, 0)

        distances = self.compute_distance(query_set, binary_prototypes)
        logits = -distances

        return logits, labels

    def binary_logit_fn(self,
                        query_set: tf.Tensor,
                        prototypes: tf.Tensor,
                        labels: tf.Tensor,
                        n_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Binary classification with randomly chosen 1-class.

        Parameters:
            query_set: As above.
            prototypes: As above.
            labels: As above.
            n_classes: Number of classes we have. Could be taken from
                       prototypes?

        Returns:
            Classification logits as well as the 0-1 labels.

        """
        chosen_class = tf.random.uniform((), maxval=n_classes, dtype=tf.int32)

        return self.binary_logit_fn_core(query_set, prototypes, labels,
                                         n_classes, chosen_class)

    def cycle_binary_logit_fn(self,
                              query_set: tf.Tensor,
                              prototypes: tf.Tensor,
                              labels: tf.Tensor,
                              n_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Binary classification where each class is treated as 1 in turn.

        Parameters:
            query_set: As above.
            prototypes: As above.
            labels: As above.
            n_classes: Number of classes we have. Could be taken from
                       prototypes?

        Returns:
            Classification logits as well as the 0-1 labels.

        """
        chosen_classes = tf.range(n_classes, dtype=tf.int32)

        array_size = n_classes
        total_logits = tf.TensorArray(tf.float32, size=array_size)
        total_labels = tf.TensorArray(tf.int32, size=array_size)

        for chosen_class in chosen_classes:
            logits, labels = self.binary_logit_fn_core(
                query_set, prototypes, labels, n_classes, chosen_class)

            total_logits = total_logits.write(chosen_class, logits)
            total_labels = total_labels.write(chosen_class, labels)

        logits = total_logits.concat()
        labels = total_labels.concat()

        return logits, labels


def euclidean_distance(inputs):
    inp1, inp2 = inputs
    return tf.norm(inp1 - inp2, axis=-1)


def squared_euclidean_distance(inputs):
    inp1, inp2 = inputs
    return tf.reduce_mean((inp1 - inp2)**2, axis=-1)
