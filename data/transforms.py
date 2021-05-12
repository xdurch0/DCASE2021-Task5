import warnings
from typing import Union

import librosa
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from scipy import ndimage
from scipy import signal


class RawExtractor:
    def __init__(self,
                 conf: DictConfig):
        self.center = conf.features.center
        self.pad_len = conf.features.n_fft // 2

    # TODO no padding here you dummy
    def extract_feature(self, audio: np.ndarray) -> np.ndarray:
        if self.center:
            audio = np.pad(audio, self.pad_len, mode="reflect")
        return audio[:, None].astype(np.float32)


class FeatureExtractor:
    def __init__(self,
                 conf: DictConfig):
        self.sr = conf.features.sr
        self.n_fft = conf.features.n_fft
        self.hop = conf.features.hop_mel
        self.n_mels = conf.features.n_mels
        self.fmax = conf.features.fmax
        self.center = conf.features.center

        self.type = conf.features.type

        time_constant = conf.features.time_constant
        if isinstance(time_constant, float):
            self.time_constant = [time_constant]
        else:
            self.time_constant = [float(t.strip()) for t in time_constant.split(",")]
        self.pcen_power = conf.features.power
        self.bias = conf.features.bias
        self.gain = conf.features.gain
        self.eps = conf.features.eps

    def extract_feature(self,
                        audio: np.ndarray) -> np.ndarray:
        audio *= 2**31  # TODO check if this actually matters

        mel_spec = librosa.feature.melspectrogram(audio,
                                                  sr=self.sr,
                                                  n_fft=self.n_fft,
                                                  hop_length=self.hop,
                                                  n_mels=self.n_mels,
                                                  fmax=self.fmax,
                                                  power=1,
                                                  center=self.center)
        if self.type == "pcen":
            # TODO adapt for multiple time constants
            features = librosa.core.pcen(mel_spec,
                                         sr=self.sr,
                                         hop_length=self.hop,
                                         time_constant=self.time_constant,
                                         power=self.pcen_power,
                                         bias=self.bias,
                                         gain=self.gain,
                                         eps=self.eps)

        elif self.type == "pcen_lowpass":

            features = [pcen_lowpass(mel_spec,
                                     sr=self.sr,
                                     hop_length=self.hop,
                                     time_constant=t) for t in self.time_constant]
            features = np.concatenate([mel_spec] + features, axis=0)

        elif self.type == "logmel":
            features = np.log(mel_spec**2 + 1e-8)

        else:
            raise ValueError("Invalid type {} in "
                             "FeatureExtractor".format(self.type))

        features = features.T.astype(np.float32)

        return features


def pcen_lowpass(S, sr=22050, hop_length=512, time_constant=0.400, b=None,
                 max_size=1, ref=None, axis=-1, max_axis=None, zi=None):

    if time_constant <= 0:
        raise ValueError('time_constant={} must be strictly '
                         'positive'.format(time_constant))

    if max_size < 1 or not isinstance(max_size, int):
        raise ValueError('max_size={} must be a positive '
                         'integer'.format(max_size))

    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter

        b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)

    if not 0 <= b <= 1:
        raise ValueError('b={} must be between 0 and 1'.format(b))

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('pcen was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call pcen(np.abs(D)) instead.')
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ValueError('Max-filtering cannot be applied to 1-dimensional '
                             'input')
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ValueError('Max-filtering a {:d}-dimensional '
                                     'spectrogram requires you to specify '
                                     'max_axis'.format(S.ndim))
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    if zi is None:
        # Make sure zi matches dimension to input
        shape = tuple([1] * ref.ndim)
        zi = np.empty(shape)
        zi[:] = signal.lfilter_zi([b], [1, b - 1])[:]

    # Temporal integration
    S_smooth, zf = signal.lfilter([b], [1, b - 1], ref, zi=zi, axis=axis)

    return S_smooth


def pcen_compress(spectro, spectro_smooth, gain, bias, power, eps):
    # Adaptive gain control
    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) +
                             np.log1p(spectro_smooth / eps)))
    # Dynamic range compression
    # TODO add 0 cases again
    if False:
        out = np.log1p(spectro * smooth)
    elif False:
        out = np.exp(power * (np.log(spectro) + np.log(smooth)))
    else:
        out = (bias ** power) * np.expm1(
            power * np.log1p(spectro * smooth / bias))

    return out


def extract_feature(audio_path: str,
                    feature_extractor: Union[FeatureExtractor, RawExtractor],
                    conf: DictConfig) -> np.ndarray:
    """Load audio and apply feature extractor.

    Parameters:
        audio_path: Path to audio file.
        feature_extractor: FeatureExtractor object to... guess what.
        conf: hydra config object.

    Returns:
        Features in shape time x features.

    """
    y, sr = librosa.load(audio_path, sr=None)
    if sr != conf.features.sr:
        raise ValueError("Audio should have been resampled at this stage. Found"
                         "{}Hz, expected {}Hz.".format(sr, conf.features.sr))

    y = feature_extractor.extract_feature(y)
    return y


def resample_audio(audio_path: str,
                   sr: int):
    """Resample and save a single audio file.

    Parameters:
        audio_path: Path to audio file.
        sr: Target sampling rate.

    """
    y, _ = librosa.load(audio_path, sr=sr)
    wavfile.write(audio_path[:-4] + "_{}hz.wav".format(sr), sr, y)
