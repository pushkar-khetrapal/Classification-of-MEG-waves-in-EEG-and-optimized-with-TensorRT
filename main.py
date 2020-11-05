from preprocess import get_eeg_data
from deepmodel import train_eeg_model 
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorrt import EEG_FrozenGraph, TfEngine, TftrtEngine,

def main(file_path):
  
    x_train_new, x_test_new, y_train, y_test = get_eeg_data(file_path) # getting the processed data
    model = train_eeg_model(x_train_new, x_test_new, y_train, y_test) # creating model
    
    for i in range(10):
        t0 = time.time()
        y_pred = model.predict(x_test_new)   # inferencing 10 times beacause the initial pass is just a warm-up
    t1 = time.time()
    x = np.argmax(y_pred, axis= 1)
    y = np.argmax(y_test, axis=1)  
    print('----------------------------------------------------------------')
    print('Confusion Matrix : ', confusion_matrix(x, y))
    print('Normal Tensorflow model time:', t1 - t0)
    print('----------------------------------------------------------------')
    frozen_graph = EEG_FrozenGraph(model)
    tf_engine = TfEngine(frozen_graph)
    tftrt_engine = TftrtEngine(frozen_graph, 23, 'FP16')
    y0_tftrt, t = tftrt_engine.infer(x_test_new)
    print('----------------------------------------------------------------')
    print('Tensorflow TensorRT time', t)
    print('----------------------------------------------------------------')
file_path = "eeg_artefact_dataset/S02_21.08.20_14.34.32.csv"
main(file_path)