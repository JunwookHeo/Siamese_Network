# -*- coding: utf-8 -*-
"""
Siamese network

If reload library errors occure
Tools->Preferences->Python interpreter-> Uncheck Enable UMR

"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import random
from keras.datasets import mnist
from keras.models import Model,Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, SGD, Adadelta
from keras import backend as K

import tensorflow as tf

import pandas as pd
import time
import os

#np.random.seed(1337)

# The flag to save results to file
is_save_result = True
# The flag to display test set images
is_disp_test_image = False
# Show analysis data such as histogram of distance 
is_show_data_for_analysis = True
# Show the difference of accuracy by threshold
is_anlysis_threshold_accuracy = False
# Input data set group 1
DGROUP1 = [2,3,4,5,6,7]
# Input data set group 2
DGROUP2 = [0,1,8,9]
# Input Image Size
img_rows, img_cols = 28, 28

# Epoch
epochs = 15
# Batch size
BATCH_SIZE = 128
#Threshold for accuracy
threshold_accuracy = 0.5

def euclidean_distance(vects):
    '''
    Compute euclidean distance
    Two vectors shoud have same shape
    @param
      vects : two vectors to be calculated
    @return
      euclidean distance
    Euclidean distance : 𝐷(𝑃𝑖,𝑃𝑗) = √∑(𝑃𝑖 − 𝑃𝑗)^2 
    '''
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    '''
    Output shape of euclidean distance
    @param
      shapes : Output shape of the last layer - [(None, 2), (None, 2)]
    @return
      Output shape : (None, 1)
    '''
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
      y_true : label
        0 : Positive pair
        1 : Negative pair
      y_pred : euclidean distance
    @return
      Loss
    Loss : 𝐿(𝑃𝑖,𝑃𝑗) = (1 − 𝑦𝑖𝑗)*𝐷(𝑃𝑖,𝑃𝑗)^2 + 𝑦𝑖𝑗 ∗ 𝑚𝑎𝑥(0,𝑚 − 𝐷(𝑃𝑖,𝑃𝑗))^2 
    '''
    m = 1
    square_pred = K.square(y_pred) # Distance of positive pair
    margin_square = K.square(K.maximum(m - y_pred, 0)) # Distance of negative pair
    return ((1-y_true) * square_pred + y_true * margin_square)

def create_pairs(x, y):
    '''
    Create positive and negative pairs of training and validation data set
    The length of x and y should be the same.
    Demension of x : (n x 28 x 28 x 1)
    Demension of y : (n x 1)
    @param
      x : x train or validation set
      y : y train or validation set (Label)
    @return
      pairs : pairs input data set
      labels : Label of pairs eg) 0 or 1
      digits : digits of pairs eg) 0 to 9
    '''
    pairs = []
    labels = []
    digits = []
    classes, _ = np.unique(y, return_counts=True)
    digit_indices = [np.where(y == c)[0] for c in classes]
    num_classes = len(digit_indices)
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1

    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            digits += [[y[z1], y[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            digits += [[y[z1], y[z2]]]
            labels += [0, 1]
            
    return np.array(pairs), np.array(labels), np.array(digits)

def create_test_pairs(x, y, ptype=0):
    '''
    Create positive and negative pairs of test data set
    The length of x and y should be the same.
    Demension of x : (n x 28 x 28 x 1)
    Demension of y : (n x 1)
    @param
      x : x test set
      y : y test set (Label)
    @return
      pairs : pairs input data set
      labels : Label of pairs eg) 0 or 1
      ptypes : The type of pairs
          0 : a pair of [2,3,4,5,6,7]
          1 : a pair of other groups, [2,3,4,5,6,7] union [0,1,8,9]
          2 : a pair of [0,1,8,9]     
      digits : digits of pairs eg) 0 to 9
    '''
    def get_random_pair(n):
        '''
        Choose two random number to create input pairs
        @param
          n : boundary
        @return
          Two numbers which are selected
        '''
        p = np.random.choice(n, 2, replace=False)
        return  p[0], p[1]
   
    pairs = []
    labels = []
    digits = []
    
    if ptype is 0:       gt = DGROUP1
    elif ptype is 1 :    gt = DGROUP1 + DGROUP2
    elif ptype is 2 :    gt = DGROUP2
    else:               raise Exception('Not support type of test set')
    
    gt = sorted(gt) # Group to be computed to create pairs
    classes, _ = np.unique(y, return_counts=True) # Total test set including from 0 to 9
    idxt = [np.where(y == g)[0] for g in classes]
    n = min([len(idxt[d]) for d in range(len(gt))]) - 1 # Minimum numbers of each digit

    for g in gt:
        # Identify the same group and different group
        if (g in DGROUP1):  
            gsame = [d for d in DGROUP1 if d is not g]
            gdiff = DGROUP2
        else:                     
            gsame = [d for d in DGROUP2 if d is not g]
            gdiff = DGROUP1
        
        # Create pairs for each digit
        for j in range(n):
            # Create a positive pair
            p1, p2 = get_random_pair(n)
            z1, z2 = idxt[g][p1], idxt[g][p2]
            pairs += [[x[z1], x[z2]]]
            digits += [[y[z1], y[z2]]]
            labels += [0]
            
            # Create a nagative pair in the same group
            g2 = random.choice(gsame)
            z1, z2 = idxt[g][p1], idxt[g2][p2]
            pairs += [[x[z1], x[z2]]]
            digits += [[y[z1], y[z2]]]
            labels += [1]
            
            # Create a positive pair
            p1, p2 = get_random_pair(n)
            z1, z2 = idxt[g][p1], idxt[g][p2]
            pairs += [[x[z1], x[z2]]]
            digits += [[y[z1], y[z2]]]
            labels += [0]
            
            # Create a nagative pair in the different groups if type is 1
            if ptype is 1:
                g2 = random.choice(gdiff)
            else:
                g2 = random.choice(gsame)
            z1, z2 = idxt[g][p1], idxt[g2][p2]
            pairs += [[x[z1], x[z2]]]
            digits += [[y[z1], y[z2]]]
            labels += [1]
    
    return np.array(pairs), np.array(labels), np.array(digits)    

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    @param
      input_shape : shape of the input vector
    @return
      model : Network model
    '''
    model = Sequential()
    # (28x28x1)
    model.add(Conv2D(32, kernel_size=(3, 3), 
                     activation='relu', 
                     input_shape=input_shape))
    # (26x26x32)
    model.add(Conv2D(64, (3, 3), 
                     activation='relu'))
    # (24x24x64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # (12x12x64)
    model.add(Dropout(0.2))
    model.add(Flatten())
    # (9216x1)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    # (128x1)
    model.add(Dense(2, activation='relu'))
    # (2x1)

    return model

def create_base_network2(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    @param
      input_shape : shape of the input vector
    @return
      model : Network model
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred, threshold):
    '''Compute classification accuracy with the threshold on distances.
    @param
      y_true : 0 for positive pair, 1 negative pair
      y_pred : values predicted by the Siamese network
      threshold : threshold to identify positive or negative pair
    @return
      accuracy : 0 to 1
    '''
    pred = y_pred.ravel() > threshold
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    This function is called only by Keras library, so it should be implemented
    using Keras beckend api.
    @param
      y_true : 0 for positive pair, 1 negative pair
      y_pred : values predicted by the Siamese network
    @return
      accuracy : 0 to 1
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred > threshold_accuracy, y_true.dtype)))

def disp_pair_image_input_dataset(X=None):
    '''
    Display images of input data set
    @param
      X : Pair 28 x 28 images of input data set
    '''
    if X is None:
        return
    
    # Create a new figure with 16x16 axes objects inside it (subplots)
    fig, axs = plt.subplots(16, 16, sharex=True, sharey=True, figsize=(6,6)) 
    axs = axs.flatten()  # The returned axs is actually a matrix holding the handles to all the subplot axes objects
    graymap = plt.get_cmap("gray")

    # Draw input images(16x16)
    for i, x in enumerate(X[:16*8:]):
        im_mat = np.reshape(x[0], (img_rows, img_cols)) #, order="F")
        axs[i*2].imshow(im_mat, cmap=graymap, interpolation="None")
        axs[i*2].xaxis.set_visible(False)  # Hide the axes labels for clarity
        axs[i*2].yaxis.set_visible(False) 
        
        im_mat = np.reshape(x[1], (img_rows, img_cols)) #, order="F")
        axs[i*2+1].imshow(im_mat, cmap=graymap, interpolation="None")
        axs[i*2+1].xaxis.set_visible(False)  # Hide the axes labels for clarity
        axs[i*2+1].yaxis.set_visible(False)
    plt.show()
    
    
def randomize(X, Y):
    '''
    Randomize data set by Label
    @param
      X : X data, eg) X Train Set or X Test Set
      Y : Label eg) 0 or 1(posivit pair or negative pair)
    @Return
      Randomized data set
    '''
    idxs = np.random.permutation(Y.shape[0])
    Xs = X[idxs, :, :]
    Ys = Y[idxs]

    return Xs, Ys

def get_out_file_name(name):
    '''
    Cteate output directory if it is not.
    And get a output file name which is combined with time information.
    @param
      name : string followed time info
    @Return
      Full name of a file to be saved
    '''
    out_path = 'out'
    tm = time.strftime('%m-%d-%Y %H-%M-%S-')
    out_file = os.path.join(out_path, tm + name)
    os.makedirs(out_path, exist_ok=True)

    return out_file

def save_result_to_csv(df, name):
    '''
    It save a test result to csv file.
    @param
      df : DataFrame of Pandas
      name : String of a name to save file
    '''
    if is_save_result is True:
        df.to_csv(get_out_file_name(name + '.csv'))
    
def load_mnist_data():
    '''
    Load MNIST data set and return Training, validation and test data sets
    @Return
      (Xtr, Ytr) : Training set with 60% of label [2,3,4,5,6,7]
      (Xva, Yva) : Validation set with 20% of label [2,3,4,5,6,7]
      (Xte, Yte) : Test set with label [0,1,8,9] and 20% of [2,3,4,5,6,7]
    '''
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')
    x_train /= 255
    x_test /= 255
    
    # Concatenate all data set
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    
    # Split data set to two groups : [2,3,4,5,6,7] and [0,1,8,9]
    Yidx = [np.where(Y == i) for i in DGROUP1]
    Xdg1 = np.concatenate(([X[i] for i in Yidx]))
    Ydg1 = np.concatenate(([Y[i] for i in Yidx]))
    Yidx = [np.where(Y == i) for i in DGROUP2]
    Xdg2 = np.concatenate(([X[i] for i in Yidx]))
    Ydg2 = np.concatenate(([Y[i] for i in Yidx]))

    Xdg1, Ydg1 = randomize(Xdg1, Ydg1)
    
    # Use 60% of [2,3,4,5,6,7] for training set
    spl1 = int(len(Ydg1)*0.6)
    Xtr = Xdg1[0:spl1]
    Ytr = Ydg1[0:spl1]
    
    # Use 20% of [2,3,4,5,6,7] for validatation set
    spl2 = int(len(Ydg1)*0.8)
    Xva = Xdg1[spl1:spl2]
    Yva = Ydg1[spl1:spl2]

    # add 20% of [2,3,4,5,6,7] and 100% of [0,1,8,9] for test set
    Xte = np.concatenate((Xdg2, Xdg1[spl2:]))
    Yte = np.concatenate((Ydg2, Ydg1[spl2:]))
            
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)

    
def disp_graph_of_history(h):
    '''
    Display graph of loss and accuracy of training set and validation sets
    @param
      h : history of Tensorflow Keras
    '''
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(h.history['loss'], 'b.-', label='Train set')
    axs[0].plot(h.history['val_loss'], 'g.-', label='Test set')

    axs[1].plot(h.history['accuracy'], 'b.-', label='Train set')
    axs[1].plot(h.history['val_accuracy'], 'g.-', label='Test set')
    
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Accuray')
    axs[1].set_xlabel('Epoch')
    
    axs[0].legend()
    axs[1].legend()
    
    plt.show()

def disp_histogram_distance(model, X, Y):
    '''
    Display distribution density of distance each positive and negative pairs
    The resolution is (len(Y))/500
    @param
      model : keras model of network
      X : train or validate data pairs
      Y : labels of train or validation      
    '''
    steps = int(len(Y)/500) # Steps of distribution graph
    y_pred = model.predict([X[:, 0], X[:, 1]])
    idx = np.where(Y==1)
    h = np.histogram(y_pred[idx], steps)
    y = h[0]/np.sum(h[0])
    x = [(h[1][i]+h[1][i-1])/2 for i in range(1,len(h[1]))]
    plt.plot(x, y, label='Netative pairs')
    idx = np.where(Y==0)
    h = np.histogram(y_pred[idx], steps)
    y = h[0]/np.sum(h[0])
    x = [(h[1][i]+h[1][i-1])/2 for i in range(1,len(h[1]))]
    plt.plot(x, y, label='Positive pairs')
    
    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.savefig(get_out_file_name('histogram.jpg'))
    plt.show()
    

def disp_distribution_output(network, X, D):
    '''
    Display distribution of outputs of the last layer
    The outputs are the outputs of the last layer in the network model and
    they are used for inputs of the contrastive loss function.
    The output should be 2-dimension.
    This is to visualize outputs of each ditits
    @param
      network : keras network to analyze its output
      X : train, validate or test data pairs
      D : digits of train, validation or test      
    '''
    intermediate_layer_model = Model(inputs=network.layers[0].input,
                                 outputs=network.layers[-1].output)
    
    x = np.reshape(X, (X.shape[0]*2, X.shape[2], X.shape[3], X.shape[4]))
    d = np.reshape(D, (D.shape[0]*2))
    y = intermediate_layer_model.predict(x)
    
    cs, _ = np.unique(d, return_counts=True)
    idx = [np.where(d == c)[0] for c in cs]
    Ys = [y[i] for i in idx]

    colormap = plt.cm.hsv
    colors = [colormap(i/10.) for i in range(10)]
    
    for i, yd in enumerate(Ys):
        plt.scatter(yd[:, 0], yd[:, 1], c=colors[cs[i]],  marker='.', label='{}'.format(cs[i]))
    plt.legend()
    plt.savefig(get_out_file_name('distribution.jpg'))
    plt.show()
    
        
def run_train_n_test(train_pairs, train_y, train_d, val_pairs, val_y, val_d, x_test, y_test):
    '''
    Create and train Siames network
    @param
      train_pairs : image pairs for train
      train_y : labels of train pairs
      train_d : digits of train pairs
      val_pairs : image pairs for validation
      val_y : labels of validation pairs
      val_d : digits of validation pairs
      test_pairs : image pairs for test
      test_y : labels for test
      type_y : types of test pairs
    '''
    # input_shape : (28 x 28 x 1)
    input_shape = (img_rows, img_cols, 1)
    # network definition
    base_network = create_base_network(input_shape)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  
    model = Model([input_a, input_b], distance)
    
    # Optimizer
    #Optimizer = RMSprop()
    #Optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Optimizer = Adadelta()
    
    # Train
    model.compile(loss=contrastive_loss, optimizer=Optimizer, metrics=[accuracy])
    hist = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
              batch_size=BATCH_SIZE,
              epochs=epochs,
              validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y))

    if is_show_data_for_analysis is True:
        print('Histogram of distance of train set')
        disp_histogram_distance(model, train_pairs, train_y)
        print('Histogram of distance of validate set')
        disp_histogram_distance(model, val_pairs, val_y)
        
        print('Distribution of train set')
        disp_distribution_output(base_network, train_pairs, train_d)
        print('Distribution of validate set')
        disp_distribution_output(base_network, val_pairs, val_d)
    
    # Evaluate accuracy of each case
    # Create Data Frame to save test result
    df = pd.DataFrame()

    # To analyze accuracy by threshold from 0.3 to 0.8
    if is_anlysis_threshold_accuracy is True:
        thresholds = np.linspace(0.3, 0.8, 26)
    else:
        thresholds = [threshold_accuracy]
        
    df['index'] = thresholds
    
    # Evaluate three types of test sets
    # [2,3,4,5,6,7] x [2,3,4,5,6,7]
    # [2,3,4,5,6,7 Union [0,1,8,9]
    # [0,1,8,9] x [0,1,8,9]
    for i in range(3):
        # create training+test positive and negative pairs
        test_pairs, test_y, digit_y = create_test_pairs(x_test, y_test, ptype=i)
        if is_disp_test_image is True:
            for j in range(0, len(test_y), int(len(test_y)/10)):
                disp_pair_image_input_dataset(test_pairs[int(j):])
        
        # compute final accuracy on test sets
        y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
        
        accs = []
        for th in thresholds:
            te_acc = compute_accuracy(test_y, y_pred, th)
            accs += [te_acc]
            print('Case {} : Accuracy on test set: {:0.2f}'.format(i, 100 * te_acc))
        
        # Display accuracy plot
        if is_anlysis_threshold_accuracy is True:
            plt.plot(thresholds, accs)
            plt.xlabel('Threshold')
            plt.ylabel('Accuracy')
            plt.savefig(get_out_file_name('threshold.jpg'))
            plt.show()
            
        # to save accuracy
        df['Case {}'.format(i)] = accs
        
        if is_show_data_for_analysis is True:
            print('Histogram of distance of test set in the case {}'.format(i))
            disp_histogram_distance(model, test_pairs, test_y)
            print('Distribution of test set in the case {}'.format(i))
            disp_distribution_output(base_network, test_pairs, digit_y)
        
    # save accuracy of test set
    save_result_to_csv(df, 'AccuracyTest')
    
    # Display Graph of Loss and Accuracy
    disp_graph_of_history(hist)
    # Save History to CSV file
    save_result_to_csv(pd.DataFrame(hist.history), 'History')
    
def run_task():
    '''
    Load train and test date set and run training and testing
    '''
    # the data, split between train and test sets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()
        
    # create training+test positive and negative pairs
    train_pairs, train_y, train_d = create_pairs(x_train, y_train)
    val_pairs, val_y, val_d = create_pairs(x_val, y_val)
    
    run_train_n_test(train_pairs, train_y, train_d, val_pairs, val_y, val_d, x_test, y_test)
    

def verify_contrastive_loss():
    '''
    Verify the contrastive_loss implemented
    Case 1 : To test pair of vectors in variou array size
        array size : generate randomly from 1 to 128
    Case 2 : Distance is fixed in positive pair case and negative pair case
        X1 : (1, 1) --> (1, 1)
        X2 : (0, 0) --> (0, 0)
    Case 3 : Distance decreases in positive pair case and negative pair case
        X1 : (1, 1) --> (1, 1)
        X2 : (0, 0) --> (1, 1)
    Case 4 : Distance decreases and then increase in positive pair case and negative pair case
        X1 : (1, 1) --> (0, 0)
        X2 : (0, 0) --> (1, 1) 
    '''
    def run_session(X1, X2, Yp, Yn, graph=False):
        '''
        Run TensorFlow session
        X1 and X2 are pair array to test loss function
        @param
          X1 : input array
          X2 : input array
          Yp : All positive pairs
          Yn : All negative pairs
          graph : Display graph
        '''
        x1 = tf.constant(X1, dtype=float)
        x2 = tf.constant(X2, dtype=float)
        yp = tf.constant(Yp, dtype=float)
        yn = tf.constant(Yn, dtype=float)
        
        m_dist = euclidean_distance((x1,x2))
        mp_loss = contrastive_loss(yp, m_dist)
        mn_loss = contrastive_loss(yn, m_dist)
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            dist = sess.run(m_dist)
            lossp = sess.run(mp_loss)
            lossn = sess.run(mn_loss)

            if graph == True:
                plt.plot(dist, label='distance')
                plt.plot(lossp, label='loss of positive pairs')
                plt.plot(lossn, label='loss of negative pairs')
                plt.legend()
                plt.xlabel('Pair No.')
                plt.ylabel('Loss,Distance')
                plt.savefig(get_out_file_name('loss_test.jpg'))
                plt.show()
    
    n = 100 # size of input array
    # Positive pair : Loss should be fixed
    Yp =  np.zeros((n, 1), dtype=float)
    # Negative pair : Loss should be fixed
    Yn =  np.ones((n, 1), dtype=float)

    # Case 1 
    # Test arrays with random vector size
    size = np.random.randint(1, 1024, 100, dtype=int)
    print(size)
    for i in size:
        X1 = np.random.rand(n, i)/n**0.5 # to avoid that all loss become zero deviding by n^0.5
        X2 = np.random.rand(n, i)/n**0.5
        print('Testing the array with size {}'.format(i))
        run_session(X1, X2, Yp, Yn, False)
    
    # Case 2
    size = 2 # array size
    # Distance between X1 and X2 decreases
    # X1 is fixed at the point (1, 1)
    X1 = np.ones((n, size), dtype=float)
    # X2 is fixed at the point (0, 0)
    # so distance will reduce
    X2 = np.zeros((n, size), dtype=float)
    run_session(X1, X2, Yp, Yn, True)
    
    # Case 3
    # Distance between X1 and X2 decreases
    # X1 is fixed at the point (1, 1)
    X1 = np.ones((n, size), dtype=float)
    # X2 moves from the point (0, 0) to (1, 1)
    # so distance will reduce
    X2 = np.zeros((n, size), dtype=float)
    X2 = np.array([x+i/n for i,x in enumerate(X2)])
    run_session(X1, X2, Yp, Yn, True)
    
    # Case 4
    # Distance between X1 and X2 decases and increases
    # X1 moves from the point (1, 1) to (0, 0)
    X1 = np.ones((n, size), dtype=float)
    X1 = np.array([x-i/n for i,x in enumerate(X1)])    
    # X2 moves from the point (0, 0) to (1, 1)
    # so distance will decrease and then increase
    X2 = np.zeros((n, size), dtype=float)
    X2 = np.array([x+i/n for i,x in enumerate(X2)])
    run_session(X1, X2, Yp, Yn, True)

if __name__ == '__main__':
    start = time.perf_counter()
    #verify_contrastive_loss()
    for i in range(1):
        run_task()
    end = time.perf_counter()
    print('Total time[Min] : ', (end - start)/60.)
    


    
    
