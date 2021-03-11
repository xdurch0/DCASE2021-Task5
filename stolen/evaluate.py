"""Functions for evaluating trained models.

Originally based on https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/baselines/deep_learning/util.py

"""
import numpy as np
import tensorflow as tf

from dataset import dataset_eval
from model import create_baseline_model


def get_probability(positive_prototype, negative_prototype, query_embeddings):
    """Calculate the probability of queries belonging to the positive class.

    Parameters:
        positive_prototype: 1D tensor-like, size d.
        negative_prototype: 1D tensor-like, size d.
        query_embeddings: 2D tensor-like, n x d.

    Returns:
        probs_ops: 1D tensor-like, size n; for each row in query_embeddings,
                   contains the probability that this query belongs to the
                   positive class.

    """
    prototypes = tf.stack([positive_prototype, negative_prototype], axis=0)
    dists = tf.norm(prototypes[None] - query_embeddings[:, None], axis=-1)
    logits = -1 * dists

    probs = tf.nn.softmax(logits, axis=-1)
    probs_pos = probs[:, 0].numpy().tolist()

    return probs_pos


def evaluate_prototypes(conf, hdf_eval, start_index_query, threshold=0.5):
    """Run the evaluation for a single dataset.

    Parameters:
        conf: hydra config object.
        hdf_eval: hdf5 file object containing positive, negative and query
                  features.
        start_index_query: Start frame of the query set with respect to the full
                          file (i.e. negative set).
        threshold: Float, threshold above which an output probability is
                   regarded as positive.

    Returns:
        onset: 1d numpy array of predicted onset times.
        offset: 1d numpy array of predicted "offset" (i.e. end-of-event) times.

    """
    hop_seg = int(conf.features.hop_seg * conf.features.sr //
                  conf.features.hop_mel)

    x_pos, x_neg, x_query = dataset_eval(hdf_eval, conf)

    dataset_query = tf.data.Dataset.from_tensor_slices(x_query)
    dataset_query = dataset_query.batch(conf.eval.query_batch_size)

    model = create_baseline_model()
    model.load_weights(conf.path.best_model)

    positive_embeddings = model(x_pos)
    positive_prototype = positive_embeddings.numpy().mean(axis=0)

    probs_per_iter = []

    batch_size_neg = conf.eval.negative_set_batch_size
    iterations = conf.eval.iterations

    for i in range(iterations):
        print("Iteration number {}".format(i))
        prob_pos_iter = []
        neg_indices = np.random.choice(len(x_neg), size=conf.eval.samples_neg,
                                       replace=False)
        negative_samples = x_neg[neg_indices]
        negative_samples = tf.data.Dataset.from_tensor_slices(negative_samples)
        negative_samples = negative_samples.batch(batch_size_neg)

        negative_embeddings = model.predict(negative_samples)
        negative_prototype = negative_embeddings.mean(axis=0)

        for batch in dataset_query:
            query_embeddings = model(batch)
            probability_pos = get_probability(
                positive_prototype, negative_prototype, query_embeddings)
            prob_pos_iter.extend(probability_pos)

        probs_per_iter.append(prob_pos_iter)
    prob_final = np.mean(np.array(probs_per_iter), axis=0)

    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > threshold, 1, 0)

    # prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = (start_index_query * conf.features.hop_mel /
                      conf.features.sr)

    onset = ((onset_frames + 1) * hop_seg * conf.features.hop_mel /
             conf.features.sr)
    onset = onset + str_time_query

    offset = ((offset_frames + 1) * hop_seg * conf.features.hop_mel /
              conf.features.sr)
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset
