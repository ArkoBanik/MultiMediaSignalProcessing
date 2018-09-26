import numpy as np
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft,fftshift

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
	bigN = frame_size * num_frames
	bigS = fft(data,bigN)
	#create frames and function w^2(n)*s(n) for autocorrelation
	for i in range(num_frames):
		data_index_offset = i * index_skip
		for j in range(frame_size):
			frame[i][j] = data[data_index_offset + j] * window[j]*window[j]

	#transform frames
	fourier_frames = np.empty([num_frames,frame_size],dtype=complex)
	for i in range(num_frames):
		fourier_frames[i] = fft(frame[i])

	pitch_periods = []

	for i in range(len(frame)):
		corr = autocorr(frame[i])
		##### showing graphs ########
		#plt.plot(corr)
		#plt.pause(.05)
		#plt.cla()
		##########################
		start_sample = 20
		end_sample = 90
		pp= (start_sample + np.argmax(corr[start_sample:end_sample]))
		pp_t = pp/fs
		pitch_periods.append(pp)
	plt.show()
	# process_frame(frame[20])
	print(pitch_periods)

#pitch refinement and spectral envelop

	for pitch in pitch_periods:
		Prange = [pitch+0.2*i for i in range(-10,11)]
		Perrors = np.empty(21)
		for Pidx in range(len(Prange)):
			P = Prange[Pidx]
			indexbands = [(int(np.floor((m-0.5)*(bigN/P))),int(np.floor((m+0.5)*(bigN/P)))) for m in range(1,int(P))]
			banderrors = np.empty(int(P))
			bandAm = np.empty(int(P))
			banddecisions = np.empty(int(P))
			#for the following steps, we define E across the band, not the whole signal
			print("eu width attempt",int(np.floor(bigN/P)))
			E_u = np.ones(int(np.floor(bigN/P)))


			for bandidx in range(len(indexbands)):
				band = indexbands[bandidx]
				E_v = fftshift(signal.hamming(band[1]-band[0]))
				Am_u = getAm(bigS,E_u,band)
				Am_v = getAm(bigS,E_v,band)
				error_u = getAmError(bigS,Am_u,E_u,band)
				error_v = getAmError(bigS,Am_v,E_v,band)
				#print("EU",error_u)
				#print("EV",error_v)
				if error_u <= error_v:
					banderrors[bandidx] = error_u
					bandAm[bandidx] = Am_u
					banddecisions[bandidx] = 0
				else:
					banderrors[bandidx] = error_v
					bandAm[bandidx] = Am_v
					banddecisions[bandidx] = 1
			Perrors[Pidx] = np.sum(banderrors)
		print("Pitch Estimate:",pitch)
		print(Prange)
		print(Perrors)


################################################
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def getPsi(bigP,phiFrames):
	k=0
	sum = 0
	while k*bigP < len(phiFrames):
		#not sure if we should use bigP or bigP-1 here??
		sum += bigP * phiFrames[k*bigP]
		k += 1
	# print("last k*bigP value",(k-1)*bigP)
	# print("last k",(k-1))
	# print("last P", bigP)
	return sum

def getAm(S,E,band):
	n =len(E)
	sum = 0
	for i in range(n):
		#print("bandstartidx",band[0])
		#print("i",i)
		sum += ((S[band[0]+i]*np.conj(E[i]))/(np.absolute(E[i])**2))
	return sum

def getAmError(S,A,E,band):
	n = len(E)
	sum = 0
	for i in range(n):
		#print(band[0]+i)
		#print("Signal",np.absolute(S[band[0]+i]))
		#print("Guess",(np.absolute(A))*np.absolute(E[i]))
		sum += (((np.absolute(S[band[0]+i])-((np.absolute(A))*np.absolute(E[i])))**2)/(2*math.pi))
	return sum

analysis(.025,.01)

# def synthesis():
