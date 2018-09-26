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
	bigN = len(data)
	bigS = fft(data)
	#create frames and function w^2(n)*s(n) for autocorrelation
	for i in range(num_frames):
		data_index_offset = i * index_skip
		for j in range(frame_size):
			frame[i][j] = data[data_index_offset + j] #* window[j]*window[j]

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
	refinedPitches = np.empty(len(pitch_periods))
	for pitchidx in range(len(pitch_periods)):
		pitch = pitch_periods[pitchidx]
		Prange = [pitch+0.2*i for i in range(-10,11)]
		Perrors = np.empty(21)
		for Pidx in range(len(Prange)):
			P = Prange[Pidx]
			omega = ((2*math.pi)/P)
			indexbands = [(int(np.floor((m-0.5)*(omega))),int(np.floor((m+0.5)*(omega)))) for m in range(1,int(P))]
			banderrors = np.empty(int(P))
			bandAm = np.empty(int(P))
			banddecisions = np.empty(int(P))
			#for the following steps, we define E across the band, not the whole signal
			#print("eu width attempt",int(np.floor(bigN/P)))
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
		refinedP = Prange[np.argmin(Perrors)]
		#print("New Pitch Estimate:",refinedP)
		refinedPitches[pitchidx] = refinedP
	#refined pitch frequencies
	analysisout = []
	for refPidx in range(len(refinedPitches)):
		refP = refinedPitches[refPidx]
		#refinedPval,Am list, V/UV as 1/0
		omega = ((2*math.pi)/refP)
		refindexbands = [(int(np.floor((m-0.5)*(omega))),int(np.floor((m+0.5)*(omega)))) for m in range(1,int(refP))]
		refbanderrors = np.empty(int(refP))
		refbandAm = np.empty(int(refP))
		refbanddecisions = np.empty(int(refP))
		E_u = np.ones(int(np.floor(bigN/refP)))
		for bandidx in range(len(refindexbands)):
			band = refindexbands[bandidx]
			E_v = fftshift(signal.hamming(band[1]-band[0]))
			Am_u = getAm(bigS,E_u,band)
			Am_v = getAm(bigS,E_v,band)
			error_u = getAmError(bigS,Am_u,E_u,band)
			error_v = getAmError(bigS,Am_v,E_v,band)
			#print("EU",error_u)
			#print("EV",error_v)
			if error_u <= error_v:
				refbanderrors[bandidx] = error_u
				refbandAm[bandidx] = Am_u
				refbanddecisions[bandidx] = 0
			else:
				refbanderrors[bandidx] = error_v
				refbandAm[bandidx] = Am_v
				refbanddecisions[bandidx] = 1
		analysisout.append((refP,refbandAm,refbanddecisions))
	return analysisout,frame_size


#SYNTH HERE


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
	# Epad = np.append(np.zeros(band[0]),np.array(E))
	# Epad = np.append(Epad,np.zeros(len(S)-band[1]+1))
	sumnum = 0
	Econj = np.conj(E)
	denom = np.absolute(E)**2
	sumdenom = np.sum(denom)
	for i in range(band[1]-band[0]):
		sumnum += S[band[0]+i]*Econj[i]
	A = sumnum/sumdenom
	return A

def getAmError(S,A,E,band):
	sum = 0
	AE = A*E
	diff = 0
	for i in range(band[1]-band[0]):
		diff = S[band[0]+i] - AE[i]
		sum += np.absolute(diff)**2
	err = sum/(2*math.pi)
	return err

def printout(analout):
	for a in range(len(analout)):
		print("frame:",a,"P estimate",analout[a][0])
		print("Am",analout[a][1])
		print("Voiced/Unvoiced",[int(i) for i in analout[a][2]])




def synthesis(frame_size, anal_out):
	sv_frames = []
	su_frames = []
	# voiced bands
	for f in range(len(anal_out)):
		Pf = anal_out[f][0]
		Ams = anal_out[f][1]
		v_uv = anal_out[f][2]
		sv = np.zeros(frame_size)
		Theta = np.zeros((len(Ams),frame_size))
		for n in range(frame_size):
			Sum = 0
			for m in range(len(Ams)):
				theta_local = 0
				A = 0
				if(n != 0):
					theta_local = Theta[m][n-1]
					omega = 2 *math.pi / Pf
					if(f < len(anal_out) -1):
						Pf1 = anal_out[f+1][0]
						omega = (f+1 - (n/frame_size))* (2 *math.pi / Pf) + ((n/frame_size)-f)*(2*math.pi/Pf1)
					Theta[m][n] = theta_local + m*omega
					if(v_uv[m]):
						A = Ams[m]
						if(f < len(anal_out)-1):
							A1 = 0
							if( m < len(anal_out[f+1][1])):
								if(anal_out[f+1][2][m]):
									A1 = anal_out[f+1][1][m]
							A = (f+1 - (n/frame_size))*Ams[m] + ((n/frame_size)-f)*A1
				Sum += A*math.cos(Theta[m][n])
			sv[n] = Sum
		sv_frames.append(sv)
	
	# unvoiced 
	S = fft(data)
	for f in range(len(anal_out)):
		v_uv = anal_out[f][2]
		su = np.zeros(frame_size)
		for n in range(frame_size):
			for m in range(len(v_uv)):
				print()







out,frame_size = analysis(.025,.01)
synthesis(frame_size,out)
