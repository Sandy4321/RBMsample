# RBMsample
A test implementation of Restricted Boltzmann Machine (RBM)

rbmtest.py is a sample script for testing. it produces several figures followings.
We define a teacher RBM with random weight, and obtain samples over the RBM with Gibbs sampling. Then we regard the obtained data as traning one, and train a new RBM.
The teacher RBM is defined as (`nvis`, `nhid`) = (8,2), which are the number of visible units and hidden ones respectively, and the visible units representations have 2^8 = 256 states, so we show whole states samples as a histgram in the top row of the figure.
The student RBM is defined as the same structure of the teacher one. The middle figure shows the states histogram from the students RBM, and the bottom figure shows the one with training by the obtained samples from the teacher RBM.
![fig1](https://github.com/shouno/RBMsample/blob/master/StateHistogram_VisUnits.png)
