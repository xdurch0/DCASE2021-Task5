"""Main entry point for everything.

"""
import os

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig

from data.preparation import feature_transform, resample_all
from model.evaluate import get_probabilities, get_events
from model.architecture import create_baseline_model
from model.training import train_protonet
from stolen.post_process import post_processing


@hydra.main(config_name="config")
def main(conf: DictConfig):
    """Main entry point. Extract features, train, or evaluate.

    Parameters:
        conf: config as produced by hydra via YAML file.

    """
    if conf.set.resample:
        resample_all(conf)

    if conf.set.features_train:
        if not os.path.isdir(conf.path.feat_path):
            os.makedirs(conf.path.feat_path)
        if not os.path.isdir(conf.path.feat_train):
            os.makedirs(conf.path.feat_train)

        print("--Extracting training features--")
        n_extract_train = feature_transform(conf=conf, mode="train")
        print("Total number of training samples: {}".format(n_extract_train))
        print("Done!\n")

    if conf.set.features_eval:
        if not os.path.isdir(conf.path.feat_eval):
            os.makedirs(conf.path.feat_eval)

        print("--Extracting evaluation features--")
        n_extract_eval = feature_transform(conf=conf, mode='eval')
        print("Total number of evaluation samples: {}".format(n_extract_eval))
        print("Done!")

    if conf.set.train:
        if not os.path.isdir(conf.path.model):
            os.makedirs(conf.path.model)

        train_protonet(conf, conf.set.n_runs)

    if conf.set.get_probs:
        if not os.path.isdir(conf.path.results):
            os.makedirs(conf.path.results)

        all_feat_dirs = os.listdir(conf.path.feat_eval)
        all_prob_storage = dict()
        for feat_dir in all_feat_dirs:
            all_prob_storage[feat_dir] = []

        for index in range(conf.set.n_runs):
            print("\nGetting probabilities for model #{} out of {}".format(
                index + 1, conf.set.n_runs))

            tf.keras.backend.clear_session()
            model = create_baseline_model(conf)
            model.load_weights(conf.path.best_model + str(index) + ".h5")

            for feat_dir in all_feat_dirs:
                audio_name = feat_dir + '.wav'

                print("Processing audio file : {}".format(audio_name))

                probs, thresh_est = get_probabilities(conf,
                                          os.path.join(conf.path.feat_eval,
                                                       feat_dir),
                                          model)

                probs_path = os.path.join(
                    conf.path.results,
                    "probs_" + audio_name[:-4] + "_" + str(index))
                np.save(probs_path, probs)

                thresh_est_path = os.path.join(conf.path.results, "thingy_" + audio_name[:-4] + "_" + str(index))
                np.save(thresh_est_path, thresh_est)

                all_prob_storage[feat_dir].append(probs)

        # ensemble probs
        print("\nEnsembling...")
        for feat_dir in all_feat_dirs:
            audio_name = feat_dir + '.wav'
            probs = np.array(all_prob_storage[feat_dir]).mean(axis=0)
            probs_path = os.path.join(
                conf.path.results,
                "probs_" + audio_name[:-4] + "_" + str(conf.set.n_runs))
            np.save(probs_path, probs)

            # TODO!!
            thresh_est_path = os.path.join(conf.path.results,
                                           "thingy_" + audio_name[
                                                       :-4] + "_" + str(conf.set.n_runs))
            np.save(thresh_est_path, 1.)

    if conf.set.eval:
        if not os.path.isdir(conf.path.results):
            raise ValueError("Results folder does not exist yet. You have to "
                             "extract the probabilities first!")

        min_thresh = conf.eval.lowest_thresh
        max_thresh = conf.eval.highest_thresh
        n_threshs = int((max_thresh - min_thresh) / conf.eval.thresh_step) + 1
        thresholds = np.around(np.linspace(min_thresh, max_thresh, n_threshs),
                               decimals=2)

        all_feat_dirs = os.listdir(conf.path.feat_eval)

        for index in range(conf.set.n_runs + 1):  # +1 for ensemble probs
            print("\nEvaluating model #{} out of {}".format(index + 1,
                                                            conf.set.n_runs))
            name_dict = {t: np.array([]) for t in thresholds}
            onset_dict = {t: np.array([]) for t in thresholds}
            offset_dict = {t: np.array([]) for t in thresholds}

            for feat_dir in all_feat_dirs:
                audio_name = feat_dir + ".wav"

                print("Processing audio file : {}".format(audio_name))

                probs_path = os.path.join(
                    conf.path.results,
                    "probs_" + audio_name[:-4] + "_" + str(index) + ".npy")
                probs = np.load(probs_path)

                thresh_est_path = os.path.join(conf.path.results,
                                               "thingy_" + audio_name[
                                                           :-4] + "_" + str(
                                                   index))
                thresh_est = np.load(thresh_est_path)

                start_index_query = np.load(
                    os.path.join(conf.path.feat_eval, feat_dir,
                                 "start_index_query.npy"))
                on_off_sets = get_events(probs,
                                         thresholds,
                                         start_index_query,
                                         conf,
                                         thresh_est)

                for thresh, (onset, offset) in on_off_sets.items():
                    name = np.repeat(audio_name, len(onset))
                    name_dict[thresh] = np.append(name_dict[thresh], name)
                    onset_dict[thresh] = np.append(onset_dict[thresh], onset)
                    offset_dict[thresh] = np.append(offset_dict[thresh], offset)

            print("Writing {} files...".format(len(thresholds)))
            for thresh in thresholds:
                df_out = pd.DataFrame({'Audiofilename': name_dict[thresh],
                                       'Starttime': onset_dict[thresh],
                                       'Endtime': offset_dict[thresh]})
                csv_path = os.path.join(
                    conf.path.results,
                    'Eval_out_model{}_thresh{}.csv'.format(index, thresh))
                df_out.to_csv(csv_path, index=False)
                post_processing(conf.path.eval_dir, csv_path,
                                csv_path[:-4] + "_postproc.csv")


if __name__ == '__main__':
    main()
