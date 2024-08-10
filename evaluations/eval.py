import librosa
import torch
import numpy as np
EPS = 1e-12


class AudioMetrics:
    def __init__(self, rate):
        self.rate = rate
        self.hop_length = int(rate / 100) # about 100 frames in a second.
        self.n_fft = int(2048 / (44100 / rate)) # about 45ms in a frame for STFT.

    def read(self, pred, target):
        pred, _ = librosa.load(pred, sr=self.rate, mono=True)
        target, _ = librosa.load(target, sr=self.rate, mono=True)
        return pred, target

    def wav_to_spectrogram(self, wav):
        f = np.abs(librosa.stft(wav, hop_length=self.hop_length, n_fft=self.n_fft)).T
        f = torch.tensor(f[None, None, ...])
        # [1, 1, #time bins, #frequency bins]
        # [1, 1, 100s+2, 1+n_fft/2] where s is data length in seconds.
        return f

    def evaluation(self, est, target, file):
        """evaluate between two audio
        Args:
            est (str or np.array): _description_
            target (str or np.array): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # import time; start = time.time()
        if type(est) != type(target):
            raise ValueError(
                "The input value should either both be numpy array or strings"
            )
        if type(est) == type(""): #Maybe file paths are given?
            est_wav, target_wav = self.read(est, target)
        else:
            assert len(list(est.shape)) == 1 and len(list(target.shape)) == 1, (
                "The input numpy array shape should be [samples,]. Got input shape %s and %s. "
                % (est.shape, target.shape)
            )


        assert (
            abs(target_wav.shape[0] - est_wav.shape[0]) < 200
        ), "Error: Length mismatch between target and estimated audio waveforms %s and %s" % (
            str(target_wav.shape),
            str(est_wav.shape),
        )

        min_len = min(target_wav.shape[0], est_wav.shape[0])
        target_wav, est_wav = target_wav[:min_len], est_wav[:min_len]

        target_sp = self.wav_to_spectrogram(target_wav)
        est_sp = self.wav_to_spectrogram(est_wav)

        result = {}

        # frequency domain. Other metrics might be added.
        result["lsd"] = self.lsd(est_sp.clone(), target_sp.clone())

        for key in result:
            result[key] = float(result[key])
        return result

    def lsd(self, est, target):
        lsd = torch.log10(target**2 / ((est + EPS) ** 2) + EPS) ** 2
        lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
        return lsd[..., None, None]