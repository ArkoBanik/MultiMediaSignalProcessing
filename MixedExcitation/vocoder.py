import numpy as np
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft,fftshift
import sounddevice as sd

fs, datafile = wavfile.read('s5.wav')

# fs, datafile = wavfile.read('arko.wav')
# datafile = datafile[:,0]

fp = 1/fs
# datafile = datafile + np.random.normal(0,0.1, len(datafile))

dm = np.mean(datafile)
data = datafile - dm
sqrs = [i**2 for i in data]
denom = math.sqrt(np.sum(sqrs))
data = data/denom

def analysis(Tframe,Tskip):
	# Read your input audio file. Extract frames of duration Tf rame seconds at every Tskip seconds. Use
	#Hamming window to scale the frames. Compute FFT of windowed frames. Get a rough integer estimate for
	#the pitch period of each frame based on the autocorrelation method described by Eqs. 12 and 13 in the paper

	index_skip = round(Tskip * fs)
	frame_size = round(Tframe * fs)
	num_frames = math.floor(data.size/index_skip)
	frame = np.empty([num_frames,frame_size])
	window = signal.hamming(frame_size)

	#create frames and function w^2(n)*s(n) for autocorrelation
	for i in range(num_frames):
		data_index_offset = i * index_skip
		for j in range(frame_size):
			if ((data_index_offset +j)>=len(data)):
				frame[i][j] = 0
			else:
				frame[i][j] = data[data_index_offset + j] * window[j]*window[j]

	#transform frames
	fourier_frames = np.empty([num_frames,1024],dtype=complex)
	for i in range(num_frames):
		fourier_frames[i] = fft(frame[i],1024)

	pitch_periods = []

	for i in range(len(frame)):
		corr = autocorr(frame[i])
		##### showing graphs ########
		##########################
		start_sample = 20
		end_sample = 90
		pp= (start_sample + np.argmax(corr[start_sample:end_sample]))
		pp_t = pp/fs
		pitch_periods.append(pp)
	#print(pitch_periods)

#pitch refinement and spectral envelop
	refinedPitches = np.empty(len(pitch_periods))
	for pitchidx in range(len(pitch_periods)):
		pitch = pitch_periods[pitchidx]
		Prange = [pitch+0.2*i for i in range(-10,11)]
		Perrors = np.empty(21)
		for Pidx in range(len(Prange)):
			P = Prange[Pidx]
			omega = ((1024)/P)
			indexbands = [(int(np.floor((m-0.5)*(omega))),int(np.floor((m+0.5)*(omega)))) for m in range(1,int(P))]
			banderrors = np.empty(len(indexbands))
			bandAm = np.empty(len(indexbands))
			banddecisions = np.empty(len(indexbands))
			#for the following steps, we define E across the band, not the whole signal

			for bandidx in range(len(indexbands)):
				band = indexbands[bandidx]
				bw = int(np.floor(band[1]-band[0]))
				E_u = np.ones(1024)
				E_v = (fft(signal.hamming(bw),1024))
				Am_u = getAm(fourier_frames[pitchidx],E_u,band)
				Am_v = getAm(fourier_frames[pitchidx],E_v,band)
				error_u = getAmError(fourier_frames[pitchidx],Am_u,E_u,band)
				error_v = getAmError(fourier_frames[pitchidx],Am_v,E_v,band)
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
		refinedPitches[pitchidx] = refinedP

	plt.plot(refinedPitches)
	plt.title('Refined Pitches with Gaussian Noise Variance 0.1')
	plt.savefig('Refined_pitches_1.png')

#refined pitch frequencies
	analysisout = []
	refP_error_per_frame = []
	for refPidx in range(len(refinedPitches)):
		refP = refinedPitches[refPidx]
		#refinedPval,Am list, V/UV as 1/0
		omega = ((1024)/refP)
		refindexbands = [(int(np.floor((m-0.5)*(omega))),int(np.floor((m+0.5)*(omega)))) for m in range(1,int(refP))]
		refbanderrors = np.empty(len(refindexbands))
		refbandAm = np.empty(len(refindexbands))
		refbanddecisions = np.empty(len(refindexbands))
		for bandidx in range(len(refindexbands)):
			band = refindexbands[bandidx]
			bw = int(np.floor(band[1]-band[0]))
			E_u = np.ones(1024)
			E_v = (fft((signal.hamming(bw)),1024))
			Am_u = getAm(fourier_frames[refPidx],E_u,band)
			Am_v = getAm(fourier_frames[refPidx],E_v,band)
			error_u = getAmError(fourier_frames[refPidx],Am_u,E_u,band)
			error_v = getAmError(fourier_frames[refPidx],Am_v,E_v,band)
			if error_u <= error_v:
				refbanderrors[bandidx] = error_u
				refbandAm[bandidx] = Am_u
				refbanddecisions[bandidx] = 0
			else:
				refbanderrors[bandidx] = error_v
				refbandAm[bandidx] = Am_v
				refbanddecisions[bandidx] = 1
		refP_error_per_frame.append(np.sum(refbanderrors))
		analysisout.append((refP,refbandAm,refbanddecisions,refindexbands))

	plt.cla()
	plt.clf()
	plt.plot(refP_error_per_frame)
	plt.title("Error per frame with Gaussian Noise Variance 0.1")
	plt.savefig('Error per frame_1.png')

	return analysisout,fourier_frames

def synthesis(anal_out,fourier_frames):
# voiced
	sv = []
	K = int(.01*fs)
	for f in range(len(anal_out)):
		Pf = anal_out[f][0]
		Ams = anal_out[f][1]
		v_uv = anal_out[f][2]
		Theta = np.zeros((len(Ams),K*len(anal_out)))
		for nidx in range(int(K)):
			n = np.floor(f*K+nidx)
			n = int(n)
			## Calculate omega
			#if last frame, omega = 2pi/Pf
			if (f == (len(anal_out)-1)):
				omega = ((2*math.pi)/Pf)
			else:
				Pf1 = anal_out[f+1][0]
				omega = (((f + 1 - (n/K))*((2*math.pi)/Pf)) + (((n/K) - f)*((2*math.pi)/Pf1)))
			sum = 0
			for m in range(len(Ams)):
				## Calculate A
				# if band m unvoiced, Amf = 0
				Amf = (Ams[m] if v_uv[m] else 0)
				# if last frame, Am[n] = Amf
				if (f == (len(anal_out)-1)):
					A = Amf
				else:
					#if band m at frame f+1 unvoiced, Amf1 = 0
					Amf1 = (anal_out[f+1][1][m] if (m < len(anal_out[f+1][1]) and anal_out[f+1][2][m]) else 0)
					A = (((f + 1 - (n/K))*(Amf)) + (((n/K) - f)*(Amf1)))

				## Calculate Theta
				if (n == 0):
					thetaprev = 0
				else:
					thetaprev = Theta[m][nidx-1]
				newtheta = thetaprev + m*omega
				Theta[m][nidx] = newtheta

				# Calculate s_v[n]
				sum += A*math.cos(newtheta)
			sv.append(sum)

# unvoiced
	s_u = []
	ufs = []
	for f in range(len(anal_out)):
		v_uv = anal_out[f][2]
		bands = anal_out[f][3]
		# find UV regions
		U = [(0 + 0j) for x in range(1024)]
		S = fourier_frames[f]
		regions = []
		for m in range(len(bands)):
			if(v_uv[m] == 0):
				(a,b) = bands[m]
				sig = calculate_sig(S,a,b)
				Gaus = np.random.normal(0,0.5*sig,(b-a)) + 1j*np.random.normal(0,0.5*sig,(b-a))
				regions.append((a,b))
				k = 0
				for i in range(a,b):
					U[i] = Gaus[k]
					k+=1
		uf = np.real(ifft(U,1024))
		ufs.append(uf)

	for f in range(len(anal_out)):
		su = np.zeros(K)
		for nidx in range(K):
			n = nidx + f*K
			if(f == (len(anal_out)-1)):
				second = 0
			else:
				second =  (((n/K) - f)*ufs[f+1][(n - ((f+1)*K))])

			su[nidx] = (((f + 1 - (n/K))*ufs[f][(n-f*K)]) + second)
		s_u.extend(su)

	##elementwise add
	reconstructed = np.real(np.add(s_u,sv))
	reconstructed = reconstructed*denom
	reconstructed = reconstructed + dm

	wavfile.write('testout_s5.wav',fs,reconstructed)
	return reconstructed
################################################
## HELPERS ##
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
	return sum

def getAm(S,E,band):
	center_idx = len(E)//2
	start_idx = center_idx - (band[1]-band[0])//2
	sumnum = 0
	Econj = np.conj(E)
	sumdenom = 0
	j = 0
	for i in range(band[0], band[1]):
		sumnum += S[i]*Econj[start_idx +j]
		sumdenom += E[start_idx +j]**2
		j += 1
	A = sumnum/sumdenom
	return A

def getAmError(S,A,E,band):
	AE = np.real(np.real(A)*E)
	diffsq = [((S[band[0] + i]-AE[i])**2) for i in range(band[1]-band[0])]
	Sum = np.sum(diffsq)
	err = Sum/(2*math.pi)
	return err

def calculate_sig(S,a,b):
	Sum = 0
	for i in range(a,b):
		Sum += np.absolute(S[i])**2
	Sum = Sum * (1/(b-a))
	return Sum

######################################################

out,fourier_frames = analysis(.025,.01)
reconstructed = synthesis(out,fourier_frames)
