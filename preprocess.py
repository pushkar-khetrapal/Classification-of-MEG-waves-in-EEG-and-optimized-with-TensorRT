import tensorflow as tf
print("Tensorflow version: ", tf.version.VERSION)

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import time


## Band Pass Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_eeg_data(file_path):
    csv = pd.read_csv(file_path).fillna(-1)    # reading CSV file
    csv = csv.drop(['EEG.P7',	'EEG.O1',	'EEG.O2',	'EEG.P8'], axis=1)  #removing 4 channels
    electrode = csv.columns[2:12]

    # Baseline Correction
    baseline = []
    for j in range(len(electrode)):
        l2 = csv[electrode[j]][:128*30].values
        baseline.append(l2)
    baseline = np.sum(np.array(baseline), axis=1)/(128*30) ## average

    # blink - jaw - empty
    blink = []
    jaw = []
    emptyList = []
    for i in range(46065):
        if csv['MarkerValueInt'][i] != -1:     # -1 -> NaN
        emptyList.append(i)
        l1 = []
        for j in range(len(electrode)):
            l2 = csv[electrode[j]][i-25:i+39].values     # 25 -> -0.2 second, 39 -> 0.3
            # Baseline correction -> band pass filter -> PSD
            l1.append(scipy.signal.welch(butter_bandpass_filter(l2-baseline[j], 1, 50, 128))[1])
        if int(csv['MarkerValueInt'][i]) == 22: #blinking
            blink.append(l1)
        elif int(csv['MarkerValueInt'][i]) == 23: # jaw clenching
            jaw.append(l1)
    blink = np.array(blink)
    jaw = np.array(jaw)

    empty = []     # for empty, taking middle point of two markers 
    a = emptyList[0]
    b = emptyList[1]
    for i in range(2, int(len(emptyList)/2)):
        l1 = []
        for j in range(len(electrode)):
            c = int((a+b)/2)       # c is middle of events
            l2 = csv[electrode[j]][c-32:c+32].values
            l1.append(scipy.signal.welch(butter_bandpass_filter(l2-baseline[j], 1, 50, 128))[1])
            a = b
            b = emptyList[i]
        empty.append(l1)
    empty = np.array(empty)
    print('blink marker shape : ', blink.shape)
    print('Jaw marker shape : ', jaw.shape)
    print('Empty shape : ', empty.shape)


    events = np.zeros((114,3))
    ones = np.ones((38,))
    # blinking
    events[:38,0] = events[:38,0] + ones
    # jaw
    events[38:76,1] = events[38:76,1] + ones
    # empty 
    events[76:,2] = events[76:,2] + ones

    blink = blink[:38]
    jaw = jaw[:38]
    empty = empty[:38]
    data = np.concatenate((blink, jaw, empty), axis = 0)  # concatenate all the events
    x_train, x_test, y_train, y_test = train_test_split(data, events, test_size=0.2, random_state=42)

  
    lis = []
    for i in range(10):    ## making 10 StandardScaler object since, taking 0 electrodes
        sc = StandardScaler()
        lis.append(sc)

    x_train_new = []
    for i in range(10): # fiting and transforming the x_train
        x_train_new.append(lis[i].fit_transform(x_train[:,i,:]).reshape(y_train.shape[0], 33, 1))

    x_test_new = []
    for i in range(10):  # transforming the x_test
        x_test_new.append(lis[i].transform(x_test[:,i,:]).reshape(y_test.shape[0], 33, 1))

    return x_train_new, x_test_new, y_train, y_test

