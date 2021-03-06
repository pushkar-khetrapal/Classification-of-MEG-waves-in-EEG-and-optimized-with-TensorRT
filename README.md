# Classification-of-MEG-waves-in-EEG-and-optimized-with-TensorRT

The most common Brain Computer Interfaces paradigm is while recoding EEG signals from brain, the involvement of noise is much higher. The noise can be of outside or inside. The inside noise is generated by muscles. The two major muscles that produces noise in EEG signals are Eye Blinking and Jaw Clenching. Removing these noise is an essential part of pre-processing. I'm trying to classify these signals in realtime EEG recoding so that, in future while understanding EEG pattern, we can neglect these timestraps.

## Given
A csv file of EEG recording with 14 electrodes to classify Eye blinking, Jaw Clenching and Nothing.
Electrodes were given : AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
Sampling Rate : 128
86 markers - 2 for baseline and rest blinking and jaw clenching
electrode placement : 10-20 System

```
(Note : This code is demo of brainwave classification using tensorflow, the priority of this notebook is to develop fast, optimized inference and accuracy. The model overfits due to less training data.)
```
## Achievements
1. Extracted features from given EEG signal.
2. Developed 1-D Hybrid Convolutional Neural Network with 10-inputs -- 1-output.
3. Optimized inference by 6.22x of normal tensorflow model.

## Approach
I developed a hybrid 1-D Convolutional Neural Network with 10 electrodes (AF3, F7, F3, FC5, T7, T8, FC6, F4, F8, AF4), since given task is to classify eye blinking and jaw clenching. The most affected area of blinking is frontal area so we are using Frontal electrodes (Af3, F7, F3, F4, F8, Af4) to measure the variation in voltage. And for jaw clenching the most variation in voltage found in Central electrodes(T7, Fc5, Fc6, T8).

## Preprocessing EEG
1. Setting the baseline by taking the average of first 30 seconds of recording (Individual channel).
2. Extracting 0.5 second window (-0.2 to 0.3 of induction) i.e, 64 sampling points
3. Subtracting the baseline.
4. Extracting frequencies of range 1-50 Hz using Band pass filter. The given task is related to muscles movement and this affects all the frequencies with great amplitude. But for more variation has seen in 1-50 Hz.
5. After refining the EEG further I take PSD of wave because lately I'm performing EEG classification task and I found PSD is one of the best feature of wave to get more accuracy.
6. Seperating data into 2 sets training and testing. Since the data is small, the division is done in 0.8:0.2 ratio.
7. Standardizing the given training waves and transforming the testing waves.
6. After extraction of features, seperating and standardizing the EEG wave. We have 4 variables :

```
Variable Shape
x_train_new - (10, 91, 64, 1) - 10 electrodes, 91 total examples for training, 64 sampling points
x_test_new - (10, 23, 64, 1) - 10 electrodes, 23 total examples for testing, 64 sampling points
y_train - (91, 3) - 91 total examples for training, 3 classes
y_test - (23, 3) - 23 total examples for testing, 3 classes
```

## Deep Learning Model

![](/media/archimegeeg.jpeg)

I made 1-D Hybrid Convolutional Neural Network. 10 inputs - 1 output Network. This model takes output of every electrode and process in seperate space and predicts class.
The main motivation behind this architecture is every position of brain respond differently for given task. If we saw EEG as data from 64 different source then merging into one big array will not give right method for prediction though the model will converge perfectly but this will not a good way to move forward with. And moreover the magnitude will be different of same frequencies captured from different electrodes. So, this method might less accucate modeling. This task is very simple but if go for more complex then it might be problem.

## Why Convolution?
-> Convolution operation is a method which maps current not-distuighable space to high distuighable space by convolving a set of numbers. The EEG has low SNR so, the quality of wave is very less. We need something more powerful which seperates the waves into its classes.

## Model Architecture : (This is model is very simple since the task is simple)

Each input passes through 2 Conv1-D (32 and 16 kernels) of size 3 -> dropout -> flatten Concatenate all the outputs from each head concatenated output -> dense(150, activation = 'relu') -> dense(50, activation = 'relu') -> softmax(3) (inbetween batchnorm and dropout layers)

## Nvidia TensorRT
Further the main task is to optimizing the deep learning model. We can build small model but, this losses the quality of prediction. So, I using tensorRT to optimize the deep learning graph.

NVIDIA TensorRT™ is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications.
## The time taken by tensorflow model : 0.011925220489501953
## The time taken after converting : 0.0019164085388183594 (Given numbers might change after every run)
tensorflow model into optimized
graph with tensorRT
which is 6.222692211993033x faster than normal model
(fun fact : if you on local GPU or cloud this will reduce to 10x or more.)
Confusion Matrix :
[[7 0 1]
[1 8 1]
[0 0 5]]

## How to use

```
git clone https://github.com/pushkar-khetrapal/Classification-of-MEG-waves-in-EEG-and-optimized-with-TensorRT.git
cd Classification-of-MEG-waves-in-EEG-and-optimized-with-TensorRT
sh dep.sh
python main.py
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.