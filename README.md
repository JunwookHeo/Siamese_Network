# Siamese_Network

## Overview

A Siamese network consists of two identical neural networks, called sister networks as shown in the Figure below. Each subnetwork shares the same weight.First, two convolutional network layers with 32 and 64 filter numbers are used respectively. Secondly, a max pooling layer with a 2x2 of pool size is used. Thirdly, the input is flattened to a 128x1 vector. Finally, the 2x1 vector is fed into the contrastive loss function.

![image](https://github.com/JunwookHeo/Siamese_Network/blob/master/Siamese%20CNN.jpg)


The last layer(2x1) is to display the distribution in a 2D graph.
So, it should be change to improve the performance.


## Contrastive Loss
The objective of the Siamese network is to identify the similarity or difference of two input images. D(P<sub>i</sub>, P<sub>j</sub>) is the Euclidean distance between two image inputs – P<sub>i</sub> and P<sub>j</sub>. If two images are from the same equivalence classes, the pair is called a positive pair which Y<sub>ij</sub>=0. If two images are from the different equivalence classes, the pair is called a negative pair which Y<sub>ij</sub>=1. The target of the network is to optimise the contrastive loss function so that the loss values of the positive pairs and negative pairs should keep decreasing.


<a href="https://www.codecogs.com/eqnedit.php?latex=L(P_i,&space;P_j)&space;=(1-y_i_j)*1/2D(P_i,&space;P_j)^2&space;&plus;&space;y_i_j*1/2max(0,&space;m-D(P_i,&space;P_j))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(P_i,&space;P_j)&space;=(1-y_i_j)*1/2D(P_i,&space;P_j)^2&space;&plus;&space;y_i_j*1/2max(0,&space;m-D(P_i,&space;P_j))^2" title="L(P_i, P_j) =(1-y_i_j)*1/2D(P_i, P_j)^2 + y_i_j*1/2max(0, m-D(P_i, P_j))^2" /></a>


where m > 0 is a margin. The Euclidean distance <a href="https://www.codecogs.com/eqnedit.php?latex=D(P_i,&space;P_j)&space;=&space;\sqrt(\sum&space;(P_i-P_j)^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D(P_i,&space;P_j)&space;=&space;\sqrt{\sum&space;(P_i-P_j)^2}" title="D(P_i, P_j) = \sqrt{\sum (P_i-P_j)^2}" /></a>

## References

http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

