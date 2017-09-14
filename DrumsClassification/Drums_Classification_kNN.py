
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import scipy.io.wavfile as wsio
from scipy.signal import hilbert
import numpy as np
import cv2

TOT_SIZE = 44100
DECAY_TH = 0.4
REAL_TH = 0.01
N = 512
FS = 44100
NB_FEATURES = 6

def normalize(dat):
    return dat.astype(np.float32)/max(np.abs(dat.astype(np.float32)))

def getEnvelope(dat):
    s = np.abs(normalize(dat))
    s_filt = np.convolve(s,np.ones((N)))/float(N)
    s_filt = normalize(s_filt)
    s_filt = s_filt[int(N-1):]
    return s_filt

def getADRfromEnv(s):
    sig = getEnvelope(s)
    m = max(sig)
    A = np.where(sig==m)[0][0]
    tmp = sig[A:]
    D = np.where(tmp<=(m*DECAY_TH))[0][0]
    tmp = sig[A+D:]
    R = np.where(tmp<=REAL_TH)[0][0]
    return A,D,R

def getFFT(dat):
    yf = fft(dat)
    xf = fftfreq(len(dat),1/FS)
    xf = fftshift(xf)
    yplot = fftshift(yf)
    fftSig = normalize(1.0/len(dat) * np.abs(yplot))
    ampls = fftSig[int(len(xf)/2):]
    freqs = xf[int(len(xf)/2):]
    return freqs,ampls

def getFreqPeak(dat):
    f,A = getFFT(dat)
    m = max(A)
    idx = np.where(A==m)[0][0]
    Peak = f[idx]
    return Peak/22050

def getPower(dat,A,D,R):
    sig = np.abs(dat)
    ptot = np.sum(sig*sig)
    sigAD = sig[0:A+D]
    pA = np.sum(sigAD*sigAD)
    sigR = sig[A+D:A+D+R]
    pR = np.sum(sigR*sigR)
    return pA/ptot,pR/ptot
    

##def getEntropyInBands(dat):
##    f,A = getFFT(dat)
##    ptot = np.sum(-20*np.log(A))
##    print(ptot)
##    return ptot

def loadRawDatas():
    with np.load('Dataset_raw/Drums_8Classes_Train_Light.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    with np.load('Dataset_raw/Drums_8Classes_Test_Light.npz') as data:
        test = data['test']
        test_labels = data['test_labels']
    return train, train_labels, test, test_labels
    
def extractFeatures(data):
    data_features = np.zeros((data.shape[0],NB_FEATURES))    
    for i in range(len(data)):
        A,D,R = getADRfromEnv(data[i,:])
        f0 = A/TOT_SIZE                         #Attack
        f1 = D/TOT_SIZE                         #Decay
        f2 = R/TOT_SIZE                         #Release
        f3 = (A+D+R)/TOT_SIZE                   #Total
        f4 = getFreqPeak(data[i,0:(A+D)])       #A+D Peak
        f5 = getFreqPeak(data[i,(A+D):(A+D+R)]) #Release Peak
        #f6,f7 = getPower(data[i,0:(A+D+R)],A,D,R)        #Power
        v = np.array((f0,f1,f2,f3,f4,f5))#,f6,f7))
        data_features[i,:] = v
    return data_features.astype(np.float32)


if __name__ == "__main__":

    train, train_labels, test, test_labels = loadRawDatas()
    
    train_features = extractFeatures(train)
    test_features = extractFeatures(test)

    knn = cv2.ml.KNearest_create()
    knn.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    res,result,neighb,dist = knn.findNearest(test_features,k=5)

    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print("Accuracy : ", accuracy,"%")

