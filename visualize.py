import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import numpy as np
import pandas as pd
import IPython.display as ipd
import os
import filter
import utils
matplotlib.use('module://backend_interagg')



def visualize_mel(S, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=24000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel spectrogram with cutoff frequency = {} Hz'.format(int(sr / 2)))
    if sr != 48000:
        plt.axhline(y = sr/2, color = "r", linestyle = "--")
        plt.text(0, sr / 2, 'Cutoff Frequency = {} Hz'.format(int(sr / 2)), color='y', fontsize=10, va='bottom')
    plt.tight_layout()
    plt.show()

    return S, img

def visualize_wav_to_mel(y, sr, save_path = None, fname = None):

    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=24000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=24000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel spectrogram with cutoff frequency = {} Hz'.format(int(sr/2)))
    if sr != 48000:
      plt.axhline(y=sr/2, color='r', linestyle='--')
      plt.text(0, sr/2, 'Cutoff Frequency = {} Hz'.format(int(sr/2)), color='y', fontsize=10, va='bottom')
    plt.tight_layout()
    if save_path and fname:
        fname = os.path.join(save_path, fname)
        fig.savefig(fname, format="png")
    plt.show()

    return S, img

def visualize_wav(y, sr):
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.show()

def audiolize_wav(y, sr):
    return ipd.Audio(data=y, rate=sr)

def visualize_and_audiolize(y, sr, save_path, mel_fname):
    mel_spec, mel_img = visualize_wav_to_mel(y, sr, save_path, mel_fname)
    visualize_wav(y, sr)
    return audiolize_wav(y, sr)


def compare_visualize(ys, srs, save_path, fname):
  assert srs[0] >= srs[1], "HR audio should be on the left!"
  fname = os.path.join(save_path, fname)
  y_rebuilt = filter.resample_audio(ys[1], srs[1], srs[0])
  mel_spec1 = librosa.feature.melspectrogram(y=ys[0], sr=srs[0], fmax = 24000)
  mel_spec2 = librosa.feature.melspectrogram(y=y_rebuilt, sr=srs[0], fmax = 24000)
  mel_spec_db1 = librosa.power_to_db(mel_spec1, ref=np.max)
  mel_spec_db2 = librosa.power_to_db(mel_spec2, ref=np.max)

# Plot the Mel spectrograms side by side
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  librosa.display.specshow(mel_spec_db1, sr=srs[0], x_axis='time', y_axis='mel', fmax = 24000)
  plt.axhline(y=srs[0]/2, color='r', linestyle='--')
  plt.text(0, srs[0]/2, 'Cutoff Frequency', color='y', fontsize=10, va='bottom')
  plt.colorbar(format='%+2.0f dB')
  plt.title('HR audio with cutoff frequency = {} Hz'.format(int(srs[0]/2)))

  plt.subplot(1, 2, 2)
  librosa.display.specshow(mel_spec_db2, sr=srs[0], x_axis='time', y_axis='mel', fmax = 24000)
  plt.axhline(y=srs[1]/2, color='r', linestyle='--')
  plt.text(0, srs[1]/2, 'Cutoff Frequency', color='y', fontsize=10, va='bottom')
  plt.colorbar(format='%+2.0f dB')
  plt.title('LR audio with cutoff frequency = {} Hz'.format(int(srs[1]/2)))

  plt.tight_layout()
  plt.show()
  ipd.display(audiolize_wav(ys[0], srs[0]))
  ipd.display(audiolize_wav(ys[1], srs[1]))

