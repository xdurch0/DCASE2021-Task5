import tensorflow as tf


# TODO write this metric if necessary
# class PlotConfusionMatrix(tf.keras.callbacks.Callback):
#     def on_test_end(self, logs=None):
#         confusion_matrix = logs["confusion_matrix"]


class ConfusionMatrix(tf.keras.metrics.MeanTensor):
    def update_state(self, y_true, y_pred, **kwargs):
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
        super().update_state(confusion_matrix)
