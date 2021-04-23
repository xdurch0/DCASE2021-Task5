"""Main entry point for everything.

"""
import os
from glob import glob

import hydra
import h5py
import numpy as np
import pandas as pd
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

        all_feat_files = [file for file in glob(os.path.join(
            conf.path.feat_eval, '*.h5'))]

        for index in range(conf.set.n_runs):
            print("\nGetting probabilities for model #{} out of {}".format(
                index + 1, conf.set.n_runs))

            model = create_baseline_model(conf)
            model.load_weights(conf.path.best_model + str(index) + ".h5")

            for feat_file in all_feat_files:
                feat_name = feat_file.split('/')[-1]
                audio_name = feat_name.replace('h5', 'wav')

                print("Processing audio file : {}".format(audio_name))

                hdf_eval = h5py.File(feat_file, 'r')
                probs = get_probabilities(conf, hdf_eval, model)
                hdf_eval.close()

                probs_path = os.path.join(
                    conf.path.results,
                    "probs_" + audio_name[:-4] + "_" + str(index))
                np.save(probs_path, probs)

    if conf.set.eval:
        if not os.path.isdir(conf.path.results):
            raise ValueError("Results folder does not exist yet. You have to "
                             "extract the probabilities first!")

        min_thresh = conf.eval.lowest_thresh
        max_thresh = conf.eval.highest_thresh
        n_threshs = int((max_thresh - min_thresh) / conf.eval.thresh_step) + 1
        thresholds = np.around(np.linspace(min_thresh, max_thresh, n_threshs),
                               decimals=2)

        all_feat_files = [file for file in glob(os.path.join(
            conf.path.feat_eval, '*.h5'))]

        for index in range(conf.set.n_runs):
            print("\nEvaluating model #{} out of {}".format(index + 1,
                                                            conf.set.n_runs))
            name_dict = {t: np.array([]) for t in thresholds}
            onset_dict = {t: np.array([]) for t in thresholds}
            offset_dict = {t: np.array([]) for t in thresholds}

            model = create_baseline_model(conf)
            model.load_weights(conf.path.best_model + str(index) + ".h5")

            for feat_file in all_feat_files:
                feat_name = feat_file.split('/')[-1]
                audio_name = feat_name.replace('h5', 'wav')

                print("Processing audio file : {}".format(audio_name))

                hdf_eval = h5py.File(feat_file, 'r')

                probs_path = os.path.join(
                    conf.path.results,
                    "probs_" + audio_name[:-4] + "_" + str(index) + ".npy")
                probs = np.load(probs_path)

                on_off_sets = get_events(probs, thresholds, hdf_eval, conf)
                hdf_eval.close()

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
