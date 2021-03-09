import os

import hydra
import tensorflow as tf
from omegaconf import DictConfig

from dataset import tf_dataset
from Feature_extract import feature_transform
from model import create_baseline_model


def train_protonet(model, train_dataset, val_dataset, conf):
    opt = tf.optimizers.Adam(conf.train.lr_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    def lr_scheduler(epoch, lr):
        if epoch > 0 and not epoch % conf.train.scheduler_step_size:
            return lr / conf.train.scheduler_gamma
        return lr

    callback_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    # TODO don't hardcode
    # TODO use validation set only once.........
    oversampled_size = 110485
    steps_per_epoch = (int(oversampled_size*0.75) //
                       (2*conf.train.n_shot * conf.train.kway))
    val_steps = (int(oversampled_size*0.25) //
                 (2*conf.train.n_shot * conf.train.kway))
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=conf.train.epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps,
                        callbacks=[callback_lr])

    return history


@hydra.main(config_name="config")
def main(conf: DictConfig):

    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)

    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)

    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)

    if conf.set.features:
        print(" --Feature Extraction Stage--")
        Num_extract_train,data_shape = feature_transform(conf=conf,mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))

        Num_extract_eval = feature_transform(conf=conf,mode='eval')
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval))
        print(" --Feature Extraction Complete--")

    if conf.set.train:
        if not os.path.isdir(conf.path.Model):
            os.makedirs(conf.path.Model)

        train_dataset, val_dataset = tf_dataset(conf)

        model = create_baseline_model()
        train_protonet(model, train_dataset, val_dataset, conf)
        #print("Best accuracy of the model on training set is {}".format(best_acc))


    """
    if conf.set.eval:

        device = 'cuda'


        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = [file for file in glob(os.path.join(conf.path.feat_eval,'*.h5'))]

        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')

            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            onset,offset = evaluate_prototypes(conf,hdf_eval,device,strt_index_query)

            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.root_dir,'Eval_out.csv')
        df_out.to_csv(csv_path,index=False)
    """


if __name__ == '__main__':
    main()
