# Siamese_Network

## Overview

A Siamese network consists of two identical neural networks, called sister networks as shown in the Figure below. Each subnetwork shares the same weight.First, two convolutional network layers with 32 and 64 filter numbers are used respectively. Secondly, a max pooling layer with a 2x2 of pool size is used. Thirdly, the input is flattened to a 128x1 vector. Finally, the 2x1 vector is fed into the contrastive loss function.

![image](https://github.com/JunwookHeo/Siamese_Network/blob/master/Siamese%20CNN.jpg)


The last layer(2x1) is to display the distribution in a 2D graph.
So, it should be change to improve the performance.


## Contrastive Loss
The objective of the Siamese network is to identify the similarity or difference of two input images. D(Pi, Pj) is the Euclidean distance between two image inputs – Pi and Pj. If two images are from the same equivalence classes, the pair is called a positive pair which Yij=0. If two images are from the different equivalence classes, the pair is called a negative pair which Yij=1. The target of the network is to optimise the contrastive loss function so that the loss values of the positive pairs and negative pairs should keep decreasing (Hadsell, Chopra & Lecun, 2006).
L(Pi, Pj) =(1-yij) 12D(Pi, Pj)2 + yij*12max(0, m-D(Pi, Pj))2			(1)
where m > 0 is a margin. The Euclidean distance D(Pi, Pj) = (Pi-Pj)2
