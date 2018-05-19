from __future__ import print_function

import argparse
import os
import sys
import time
import csv
from sklearn.svm import SVC
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.ensemble import RandomForestClassifier
from cv2 import resize

from network import *

DATA_DIR = './data/'
MODEL_DIR = './model-mnist-train-32x32.npy'
RESTORE_MODEL_FROM = None
SAAK_COEF_DIR = './coef-mnist-test-32x32.npy'
RESTORE_COEF_FROM = None

# get the input arguments 
def get_argument():
    parser = argparse.ArgumentParser(description="Compute Saak Coefficient for MNIST")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
            help="directory to store mnist dataset")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
            help="directory to store extracted Saak anchor vectors")
    parser.add_argument("--restore-model-from", type=str, default=RESTORE_MODEL_FROM,
            help="stored saak model file (there will be no training if this parameter is provided)")
    parser.add_argument("--restore-coef-from", type=str, default=RESTORE_COEF_FROM,
            help="stored saak coefficients file (there will be no computation if this parameter is provided)")
    parser.add_argument("--saak-coef-dir", type=str, default=SAAK_COEF_DIR,
            help="di")
    return parser.parse_args()

# Performing f_test
def f_test(train_coef, labels, thresh=0.999):
    f_val,p = f_classif(train_coef, labels)
    low_conf = p>0.05
    f_val[low_conf] = 0
    where_are_NaNs = np.isnan(f_val)
    f_val[where_are_NaNs] = 0
    idx = f_val > np.sort(f_val)[::-1][int(np.count_nonzero(f_val) * thresh) - 1]
    selected_feat = train_coef[:, idx]
    print('f-test selected feature shape is {}'.format(selected_feat.shape))
    return selected_feat, idx
    
# Reducing the feature dimension
def reduce_feat_dim(feat, dim):
    pca = PCA(svd_solver='full', n_components=dim)
    reduced_feat = pca.fit_transform(feat)
    print('pca reduced feature shape is {}'.format(reduced_feat.shape))
    return reduced_feat, pca

#Performing classification
def rf_classifier(feat, y, kernel='rbf'):
    clf = RandomForestClassifier()
    print('Fitting data to random forest')
    clf.fit(feat, y)
    print('Completed Data fitting')
    return clf

def main():
    args = get_argument()

    # initialize tf session
    sess = tf.Session()

    # load MNIST data
    mnist = read_data_sets(args.data_dir, reshape=False, validation_size=20000)
    print("Input MNIST image shape: " + str(mnist.train.images.shape))

    # resize MNIST images to 32x32
    train_images = [resize(img,(32,32)) for img in mnist.train.images]
    train_images = np.expand_dims(train_images, axis=3)
    print("Resized MNIST images: " + str(train_images.shape))

    # extract saak anchors
    if args.restore_model_from is None:
        #hari change 1
        anchors = get_saak_anchors(train_images, sess)
        np.save(args.model_dir, {'anchors': anchors})
    else:
        print("\nRestore from existing model:")
        data = np.load(args.restore_model_from).item()
        anchors = data['anchors']
        print("Restoration succeed!\n")

    # build up saak model
    print("Build up Saak model")
    model = SaakModel()
    model.load(anchors)

    # prepare testing images
    print("Prepare testing images")
    input_data = tf.placeholder(tf.float32)
    test_images = [resize(img,(32,32)) for img in mnist.test.images]
    test_images = np.expand_dims(test_images, axis=3)

    # compute saak coefficients for testing images
    if args.restore_coef_from is None:
        print("Compute saak coefficients")
        out = model.inference(input_data, layer=0)
        test_coef = sess.run(out, feed_dict={input_data: test_images})
        train_coef = sess.run(out, feed_dict={input_data: train_images})
        # save saak coefficients
        print("Save saak coefficients")
        np.save(args.saak_coef_dir, {'train': train_coef, 'test': test_coef})
    else:
        print("Restore saak coefficients from existing file")
        data = np.load(args.restore_coef_from).item()
        #reshape and save it as csv - last batch has both train and test
        train_coef = data['train']
        test_coef = data['test']


    train_coef = np.reshape(train_coef, [train_coef.shape[0], -1])
    test_coef = np.reshape(test_coef, [test_coef.shape[0], -1])
    print("Saak feature dimension: " + str(train_coef.shape[1]))


    print("\nDo classification using Random Forest classifier ")
    selected_train_feat, idx = f_test(train_coef, mnist.train.labels, 0.999)
    reduced_train_feat, pca = reduce_feat_dim(selected_train_feat, dim = 32)
    clf = rf_classifier(reduced_train_feat, mnist.train.labels)
    pred_train = clf.predict(reduced_train_feat)
    
    train_acc = accuracy_score(mnist.train.labels,pred_train )
    print('training acc is {}'.format(train_acc))
    
    selected_test_feat = test_coef[:,idx]
    reduced_test_feat = pca.transform(selected_test_feat)
    
    print("\nDo classification using Random Forest classifier")
    test_acc = perform_classification(reduced_train_feat, mnist.train.labels, reduced_test_feat, mnist.test.labels)
    print("test Accuracy is {}".format(test_acc))



########################################################################################




#################################################

if __name__ == "__main__":
    main()
