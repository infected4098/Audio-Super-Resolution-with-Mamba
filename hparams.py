#Following MMA: https://github.com/infected4098/MMA/tree/main
import numpy as np
import torch


info = {
  "project_name" : "Speech super resolution using MMA as a backbone",
  "initial_work" : "2024/05/09",
  "researchers" : ["Yong Joon Lee", "Chan Woo Kim"],
  "seed": 1229
}
train = {
  "batch_size": 8,
  "lr": 0.0002,
  "weight_decay": 0.999,
  "num_workers": 16,
  "n_gpu": 4,
  "opt_eps": 1e-9,
  "beta1": 0.8,
  "beta2": 0.99,
  "n_epoch" : 3000,
  "checkpoint_interval" : 5000,
  "summary_interval" : 100,
  "validation_interval" : 1000,
  "stdout_interval" : 5
}


data = {
  "confidential"
}

ckpt = {
"confidential"
}


# Following NVSR
audio = {
  "sr": 48000,
  "window" : "hann",
  "nfft": 2048,
  "n_mels" : 80,
  "window_size" : 1024,
  "fmax_for_loss" : np.NaN,
  "fmax" : 8000,
  "fmin" : 0,
  "hop": 512,
  "UPR": 3,
  "segment_size": 25600 #about 0.53 sec
}

embed = {
  "cnn_stem_channels_1D" : [24,48,96,192,384],
  "cnn_stem_strides_1D": [4, 4, 2, 2, 2],
  "cnn_stem_kernel_sizes_1D" : [20, 16, 8, 8, 8]
}

hifigan = {
  "upsample_rates": [4, 4, 2, 2, 2],
  "upsample_kernel_sizes": [8, 8, 4, 4, 4],
  "upsample_initial_channel": 384, # should be the same as "cnn_stem_channels_1D[-1] "
  "resblock_kernel_sizes": [3, 7, 11],
  "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
}

mamba = {
  "d_model" : 384, #should be the same as "cnn_stem_channels_1D"[-1]
  "d_state" : 16,
  "d_conv" : 4,
  "expand":2,
  "n_blocks":12

}

dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1
    }


class CustomDict:
  def __init__(self):
    # Directly set the internal dictionary without triggering __setattr__
    super().__setattr__('_dict', {})

  def __setattr__(self, name, value):
    # Avoid recursion by handling '_dict' attribute directly
    if name == '_dict':
      super().__setattr__(name, value)
    else:
      self._dict[name] = value

  def __getattr__(self, name):
    try:
      return self._dict[name]
    except KeyError:
      raise AttributeError(f"'CustomDict' object has no attribute '{name}'")

  def __getitem__(self, name):
    return self._dict[name]

  def __setitem__(self, name, value):
    self._dict[name] = value

  def __iter__(self):
    return iter(self._dict)

  def items(self):
    return self._dict.items()

  def keys(self):
    return self._dict.keys()

  def values(self):
    return self._dict.values()

hyperparameters = CustomDict()

dicts = [("info", info), ("train", train), ("data", data), ("ckpt", ckpt),
                   ("audio", audio), ("embed", embed), ("hifigan", hifigan), ("mamba", mamba), ("dist_config", dist_config)]
for name, d in dicts:
    setattr(hyperparameters, name, d)

