import python_speech_features as psf
import math
import scipy.io.wavfile as wav
import numpy as np

names = ['asr','cnn','dnn','hmm','tts']
p = 'feature/py_yx/yx_'
for n in names:
    for i in range(1,6):
        filepath = p+n+str(i)+".fea"
        print(filepath)
        f = open(filepath,"w+")
        (rate,sig) = wav.read("data/yx/yx_"+n+str(i)+".wav")
        numr = round(rate*0.03)
        tr = psf.mfcc(sig, samplerate=rate, winlen=0.03, winstep=0.02, numcep=14, nfft=numr, nfilt=40, appendEnergy=True)
        for t in tr:
            f.write(",".join([str(x) for x in t]))
            f.write("\n")
# print(tr)
