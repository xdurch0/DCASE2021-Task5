import tensorflow as tf


# TODO write this metric if necessary
# class PlotConfusionMatrix(tf.keras.callbacks.Callback):
#     def on_test_end(self, logs=None):
#         confusion_matrix = logs["confusion_matrix"]


class ConfusionMatrix(tf.keras.metrics.MeanTensor):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes

    def update_state(self, y_true, y_pred, **kwargs):
        y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
        confusion_matrix.set_shape((self.n_classes, self.n_classes))
        super().update_state(confusion_matrix)
