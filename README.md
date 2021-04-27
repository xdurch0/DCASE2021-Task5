# DCASE2021-Task5
Code for http://dcase.community/challenge2021/task-few-shot-bioacoustic-event-detection

Most files are originally based on 
[the provided deep learning baseline](https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/baselines/deep_learning).

Usage is "almost" the same.


## Config parameters


### set
| Parameter | What does it do |
| ---- | ----- |
| resample | If true, resample all audio files to the specified sampling rate. Must be run before the other steps.
| features_train | If true, extract features for training set.
| features_eval | If true, extract features for validation set.
| train | If true, train models.
| get_probs | If true, get event probabilities for the validation set.
| eval | If true, evaluate trained models based on previously extracted probabilities.
| n_runs | How many models to train and evaluate. Note: Will crash if trying to evaluate more models than are available.

### path
| Parameter | What does it do |
| ---- | ----- |
| root_dir | Convenience path to directory so that other paths can be relative. Strictly speaking not necessary.
| train_dir | Path to the downloaded, unpacked training data.
| eval_dir | Path to the downloaded, unpacked validation data.
| feat_path | Directory where features will be extracted to (or are assumed to be found in).
| feat_train | Directory with training features.
| feat_eval | Directory with validation features.
| model | Directory where trained models are stored.
| best_model | Base name to use for best result of a single training run.
| last_model | Base name to use for the final result of a single training run.
| results | Directory where evaluation results are stored.

### features
| Parameter | What does it do |
| ---- | ----- |
| type | What kind of preprocessing is done. Several are available.
| seg_len | Length of a data "segment", in seconds. Depending on other parameters, this exact size may not be achievable.
| hop_seg | Hop size between consecutive segments. Again, the exact hop size may not be achievable.
| fmax | Maximum frequency to use for mel spectrogram. Probably want to keep this at sr // 2.
| sr | Sampling rate to use for the data. All data is resampled to this rate.
| n_fft | Window size for STFT.
| n_mels | Number of mel frequency bins to use.
| hop_mel | Hop size for the STFT extraction.
| time_constant | PCEN argument.
| gain | PCEN argument. In case of trainable PCEN, serves as the initial value.
| bias | PCEN argument. In case of trainable PCEN, serves as the initial value.
| power | PCEN argument. In case of trainable PCEN, serves as the initial value.
| eps | PCEN argument. In case of trainable PCEN, serves as the initial value.
| center | If true, pad half a window size at the beginning and end of the data such that frames are "centered" and not left-aligned.
| use_negative | Number of negative segments to extract *per recording* in the training set. Can be `all` in which case _all_ training audio is extracted as negative examples.

### train
| Parameter | What does it do |
| ---- | ----- |
| n_shot | Size of support set per class per episode.
| n_query | Size of query set per class per episode.
| k_way | How many classes to use per episode. Classes are randomly chosen each iteration; remaining data is discarded. `0` uses all classes.
| lr | Initial learning rate for `Adam`.
| scheduler_gamma | Multiplication factor for learning rate decrease.
| patience | How many epochs to wait with no validation improvement before reducing learning rate. 3 times this value is used for early stopping.
| epochs | Maximum number of epochs to train.
| test_split | Experimental: Name of subset to use as held-out set during training.
| binary | If true, each episode one class is randomly chosen as "positive" and all others receive a "negative" label.
| cycle_binary If true, binary training will cycle through classes in each batch, picking each class as positive once and averaging results. No effect if binary is false.
| label_smoothing | Apply this amount of label smoothing to the cross-entropy loss.

### model
| Parameter | What does it do |
| ---- | ----- |
| dims | Can be 1 or 2. If 2, view input spectrograms as an image with one channel and do 2D convolution. If 1, view it as a time series with many input channels and do 1D convolution.
| preprocess | Only used if `features.type` is `raw`. In that case, this gives the preprocessing layer used by the model.
| pool | `all`, `freqs`, `time` or `none`. Respective dimensions are summarized via global max pooling.
| distance_fn | Which distance function to use between embeddings and prototypes.

### eval
| Parameter | What does it do |
| ---- | ----- |
| samples_neg | How many samples to use for the negative prototype.
| iterations | How many iterations to average the predictions over.
| query_batch_size | Well?
| negative_batch_size | Guess what.
| thresholding | `absolute` or `relative`. The latter is very bad, lol.
| lowest_thresh | Lowest threshold to evaluate with.
| highest_thresh | Hmmmm.
| thresh_step | Hmmmmmmmm.
