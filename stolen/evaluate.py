import numpy as np
import tensorflow as tf

from dataset import dataset_eval
from model import create_baseline_model


def get_probability(positive_prototype, negative_prototype, query_embeddings):
    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set
     Out:
     - Probabiility array for the positive class
     """
    prototypes = tf.stack([positive_prototype, negative_prototype], axis=0)
    dists = tf.norm(prototypes[None] - query_embeddings[:, None], axis=-1)
    logits = -1 * dists

    probs = tf.nn.softmax(logits, dim=-1)
    probs_pos = probs[:, 0].numpy().tolist()

    return probs_pos


def evaluate_prototypes(conf=None, hdf_eval=None, strt_index_query=None):
    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr //
                  conf.features.hop_mel)

    x_pos, x_neg, x_query = dataset_eval(hdf_eval, conf)

    dataset_query = tf.data.Dataset.from_tensor_slices(x_query)
    dataset_query = dataset_query.batch(conf.eval.query_batch_size)

    model = create_baseline_model()
    model.load_weights("PATH")

    positive_embeddings = model(x_pos)
    positive_prototype = positive_embeddings.numpy().mean(axis=0)

    'List for storing the combined probability across all iterations'
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
    prob_final = np.mean(np.array(probs_per_iter),axis=0)

    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > 0.5, 1, 0)

    prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query

    offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset

