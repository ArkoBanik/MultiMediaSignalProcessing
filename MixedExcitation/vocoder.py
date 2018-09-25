import numpy as np
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft

fs, data = wavfile.read('s5.wav')
#print(fs)
fp = 1/fs

def analysis(Tframe,Tskip):
	# Read your input audio file. Extract frames of duration Tf rame seconds at every Tskip seconds. Use
	#Hamming window to scale the frames. Compute FFT of windowed frames. Get a rough integer estimate for
	#the pitch period of each frame based on the autocorrelation method described by Eqs. 12 and 13 in the paper
	
	index_skip = round(Tskip * fs)
	frame_size = round(Tframe * fs)
	num_frames = math.floor(data.size/frame_size)
	frame = np.empty([num_frames,frame_size])
	window = signal.hamming(frame_size)
	
	#create frames
	for i in range(num_frames):
		data_index_offset = i * index_skip
		for j in range(frame_size):
			frame[i][j] = data[data_index_offset + j] * window[j]*window[j]

	fourier_frames = np.empty([num_frames,frame_size],dtype=complex)
	for i in range(num_frames):
		fourier_frames[i] = fft(frame[i])


	plt.plot(frame[18])
	plt.show()


analysis(.025,.01)	

# def synthesis():

