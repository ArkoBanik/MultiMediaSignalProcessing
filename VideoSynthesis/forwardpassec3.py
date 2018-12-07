import numpy as np
import sklearn.metrics
iw1 = np.loadtxt('iw1.csv',delimiter=',')
iw2 = np.loadtxt('iw2.csv',delimiter=',')
iw3 = np.loadtxt('iw3.csv',delimiter=',')
b1 = np.loadtxt('b1.csv',delimiter=',')
b2 = np.loadtxt('b2.csv',delimiter=',')
b3 = np.loadtxt('b3.csv',delimiter=',')
lw1 = np.loadtxt('lw1.csv',delimiter=',')
lw2 = np.loadtxt('lw2.csv',delimiter=',')
lw3 = np.loadtxt('lw3.csv',delimiter=',')

audiotrain = np.loadtxt('avtrainaudio.csv',delimiter=',')
audiotest = np.loadtxt('avvalidateaudio.csv',delimiter=',')

visualtrain = np.loadtxt('avtrainvisual.csv',delimiter=',')
visualtest = np.loadtxt('avvalidatevisual.csv',delimiter=',')

print(audiotest.shape[1])
out1 = []
for colidx in range(audiotest.shape[1]):
    x1 = np.reshape(audiotest[:,colidx],(1,60))
    wx1 = np.matmul(x1,iw1.T)
    wxb1 = wx1 + b1
    out = np.matmul(wxb1,lw1)
    out1.append(out[0])

out2 = []
for colidx in range(audiotest.shape[1]):
    x2 = np.reshape(audiotest[:,colidx],(1,60))
    wx2 = np.matmul(x2,iw2.T)
    wxb2 = wx2 + b2
    out = np.matmul(wxb2,lw2)
    out2.append(out[0])

out3 = []
for colidx in range(audiotest.shape[1]):
    x3 = np.reshape(audiotest[:,colidx],(1,60))
    wx3 = np.matmul(x3,iw3.T)
    wxb3 = wx3
    out = np.matmul(wxb3,lw3)
    out3.append(out[0])

print(np.array(out1).shape)
output = np.array([out1,out2,out3])
print(sklearn.metrics.mean_squared_error(visualtest,output))
