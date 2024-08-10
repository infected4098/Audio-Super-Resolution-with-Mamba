from utils import *
import filter
import hparams
import visualize
#hparams to be defined: UPR, Input sampling rate

UPR = 3


#y_high, sr_high = load_audio(hparams.data["example_audio"]) # assert sr == 48000
y_high, sr_high = load_audio(hparams.data["example_audio"], sr = 48000) # assert sr == 48000

sr_low_calc, cutoff_freq = calc_sr_by_UPR(sr_high, UPR)
y_low, sr_low = filter.build_lowres(y_high, sr_high, cutoff_freq) #assert sr_low_calc = cutoff_freq
print(sr_high, sr_low, y_high.shape, y_low.shape)
y_rebuilt, sr_rebuilt = filter.resample_audio(y_low, sr_low, sr_high)

#sampling rate stays the same, but the signal is deteriorated

#norm_logmel = wav_to_log_mel(y_high, sr_high, 128, normalize = False)
visualize.audiolize_wav(y_high, sr_high)