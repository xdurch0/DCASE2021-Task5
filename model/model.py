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

        self.crop_size = 2  # TODO MAGIC NUMBER

        if binary_training:
            if cycle_binary:
                self.logit_fn = self.cycle_binary_logit_fn
            else:
                self.logit_fn = self.binary_logit_fn
        else:
            self.logit_fn = self.multiclass_logit_fn

        self.crop_layer = tf.keras.layers.experimental.preprocessing.RandomCrop(
            height=self.input_shape[1],
            width=self.input_shape[2],
            name="random_crop")
        self.crop_layer.build((self.input_shape[1] + self.crop_size,
                               self.input_shape[2],
                               1))

    def process_batch_input(self,
                            data_batch: tuple,
                            k_way: Union[int, None] = None) -> Tuple[tf.Tensor,
                                                                     tf.Tensor,
                                                                     tf.Tensor,
                                                                     tf.Tensor,
                                                                     int]:
        """Stack zipped data batches into a single one.

        Parameters:
            data_batch: Tuple of data batches, one per class.
            k_way: If given > 0, sample this many classes randomly out of all
                   the ones available. Note: This will replace n_classes in the
                   output!

        Returns:
            support: n_classes x n_support x ...
            query: n_classes*n_query x ... Note that this has a different shape
                   than support!! The first dimension is all examples of the
                   first class, followed by all of the second class, etc.
            n_classes: How many classes there are in the output.

        """
        n_classes = len(data_batch)

        def amazing_subfunction(inp, n_classes, mask_mode=False):
            if k_way:
                inputs_stacked = tf.stack(inp, axis=0)
                class_picks = tf.random.shuffle(
                    tf.range(1, n_classes, dtype=tf.int32))[:(k_way-1)]
                class_picks = tf.concat([[0], class_picks], axis=0)

                inputs_stacked = tf.gather(inputs_stacked, class_picks)

                # TODO don't hardcode
                if mask_mode:
                    stacked_query_shape = [k_way * self.n_query, 34]
                else:
                    stacked_query_shape = [k_way * self.n_query, 34, 256]

                support = inputs_stacked[:, :self.n_support]
                query = inputs_stacked[:, self.n_support:]
                query = tf.reshape(query, stacked_query_shape)

                n_classes = k_way
            else:
                support = tuple(d[:self.n_support] for d in inp)
                query = tuple(d[self.n_support:] for d in inp)

                support = tf.stack(support, axis=0)
                query = tf.concat(query, axis=0)

            return support, query, n_classes

        inputs = [d[0] for d in data_batch]
        masks = [d[1] for d in data_batch]
        inp_support, inp_query, n_classes = amazing_subfunction(inputs, n_classes)
        mask_support, mask_query, _ = amazing_subfunction(masks, n_classes, True)

        return inp_support, inp_query, mask_support, mask_query, n_classes

    def proto_compute_loss(self,
                           support_stacked: tf.Tensor,
                           query_stacked: tf.Tensor,
                           support_mask: tf.Tensor,
                           query_mask: tf.Tensor,
                           n_classes: Union[int, tf.Tensor],
                           training: bool = False) -> Tuple[tf.Tensor,
                                                            tf.Tensor,
                                                            tf.Tensor]:
        """Compute the training loss for the Prototypical Network.

        Parameters:
            support_stacked: As returned by process_batch_input.
            query_stacked: See above,
            support_mask: Yes.
            query_mask: Indeed.
            n_classes: See above.
            training: Whether to run model in training mode (batch norm etc.).

        Returns:
            loss: Loss on this input batch.
            logits: Logits returned by the model.
            labels: Labels as constructed from the batch information.

        """
        # extend support stacked by ALL croppings
        support_augmented = self.get_all_crops(support_stacked)
        # add axis in mask for feature dimension in support set
        # we assume there are exactly two!! (frequencies, channels)
        support_mask_augmented = self.get_all_crops(support_mask)[..., None, None]

        # augmented is c x n_croppings*n_support x ...
        # TODO if batch shape can be more than 1 axis, we can save lots of reshaping here
        new_shape = tf.concat(
            [[n_classes * (self.crop_size + 1)*self.n_support],
             self.input_shape[1:]],
            axis=0)

        support_stacked = tf.reshape(support_augmented, new_shape)
        support_embeddings = self(support_stacked, training=training)  # c * (n_augmented_supp) x d

        stacked_shape = tf.concat(
            [[n_classes, (self.crop_size + 1)*self.n_support],
             self.output_shape[1:]],
            axis=0)
        support_set = tf.reshape(support_embeddings, stacked_shape)

        masked_support = support_mask_augmented * support_set
        # n_classes x d, average over support set as well as time axis
        one_count = tf.reduce_sum(support_mask_augmented, axis=[1, 2])
        prototypes = tf.reduce_sum(masked_support, axis=[1, 2]) / one_count

        negative_support_mask = tf.cast(tf.math.logical_not(tf.cast(support_mask_augmented, tf.bool)), tf.float32)
        negative_masked_support = negative_support_mask * support_set
        negative_count = tf.reduce_sum(negative_support_mask, axis=[0, 1, 2])
        negative_prototype = tf.reduce_sum(negative_masked_support, axis=[0,1,2]) + prototypes[0] / (negative_count + one_count[0])

        prototypes = tf.concat([negative_prototype[None], prototypes], axis=0)

        # random crop on query set here instead of in architecture
        query_stacked = self.crop_layer(
            query_stacked[..., None], training=training)[..., 0]
        query_set = self(query_stacked, training=training)  # c * query x t x d

        # TODO don't hardcode, maybe adapt to random cropping
        # this is n_classes * n_query x 32
        query_mask = tf.cast(query_mask[:, 1:-1], tf.int32)

        # masks are all 1-0, so here we assign a different label to each class
        # TODO for this to quite make sense, NEGATIVE class has to be label 0?
        labels = tf.repeat(tf.range(1, n_classes+1, dtype=tf.int32),
                           repeats=[self.n_query])
        labels = query_mask * labels[:, None]

        # idea: if we reshape both query and labels such that time axis becomes
        # part of "batch axis", we can use the following code as-is??
        query_set = tf.reshape(query_set, (-1,) + self.output_shape[2:])
        labels = tf.reshape(labels, [-1])

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
        support, query, mask_support, mask_query, n_classes = self.process_batch_input(data,
                                                                                       k_way=self.k_way)

        with tf.GradientTape() as tape:
            loss, logits, labels = self.proto_compute_loss(
                support, query, mask_support, mask_query, n_classes, training=True)

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
        support, query, mask_support, mask_query, n_classes = self.process_batch_input(data)

        loss, logits, labels = self.proto_compute_loss(
            support, query, mask_support, mask_query, n_classes, training=False)

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

        ndims = 3 # TODO magic number
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

    def get_all_crops(self, inputs):
        # TODO generalize...
        left = inputs[:, :, :-2]
        mid = inputs[:, :, 1:-1]
        right = inputs[:, :, 2:]

        return tf.concat([left, mid, right], axis=1)


def euclidean_distance(inputs):
    inp1, inp2 = inputs
    return tf.norm(inp1 - inp2, axis=-1)


def squared_euclidean_distance(inputs):
    inp1, inp2 = inputs
    return tf.reduce_mean((inp1 - inp2)**2, axis=-1)
