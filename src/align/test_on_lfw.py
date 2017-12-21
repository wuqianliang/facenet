"""Find face and plotting bounding box for each face."""
# MIT License
# 
# Copyright (c) 2017 wuqianliang@ismartearth.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import cv2
import models.inception_resnet_v1 as network
import utils
import dlib
from align_dlib import AlignDlib
import skimage.io
import skimage.transform
import sklearn.metrics.pairwise as pw
from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print (fmt % (time()-_tstart_stack.pop()))

def compare_pic(feature1, feature2):
#    print(feature1, feature2)
    predicts = pw.cosine_similarity(feature1, feature2);
    return predicts;

def read_image(filepath):

    X = np.empty((1,3,160,160));
    im = skimage.io.imread(filepath ,as_grey=False);
    image = skimage.transform.resize(im, (160, 160))*255;
    #print(image.shape)
    X[0,0,:,:] = (image[:,:,0]).reshape(1,160,160);
    X[0,1,:,:] = (image[:,:,1]).reshape(1,160,160);
    X[0,2,:,:] = (image[:,:,2]).reshape(1,160,160);

    return X
    # averageImg = [129.1863, 104.7624, 93.5940];
    #X = np.empty((1,3,160,160));
    #filename = filepath.split('\n');
    #filename = filename[0];
    #im = skimage.io.imread(filename, as_grey=False);
    #image = skimage.transform.resize(im, (160, 160))*255;
    #X[0,0,:,:] = (image[:,:,0] - 128).reshape(1,160,160);
    #X[0,1,:,:] = (image[:,:,1] - 128).reshape(1,160,160);
    #X[0,2,:,:] = (image[:,:,2] - 128).reshape(1,160,160);

    #return X;

def main(args):
    #sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))

    #dataset = facenet.get_dataset(args.input_dir)
    align_dlib = AlignDlib('/home/arthur/facenet.git/trunk/src/align/shape_predictor_68_face_landmarks.dat')
    DATA_BASE = "/home/arthur/caffe-master/data/lfw/";
    POSITIVE_TEST_FILE = "/home/arthur/facenet.git/trunk/src/align/positive_pairs_path.txt";
    NEGATIVE_TEST_FILE = "/home/arthur/facenet.git/trunk/src/align/negative_pairs_path.txt";

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
            print('Creating networks and loading parameters')
            ###################tensorflow predictor init#############################
            FLAGS = utils.load_tf_flags()
            facenet.load_model(FLAGS.model_dir)
            facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            ###########################################################

            minsize = 10 # minimum size of face
            threshold = [ 0.5, 0.5, 0.7 ]  # three steps's threshold
            factor = 0.709 # scale factor

            # Add a random key to the filename to allow alignment using multiple processes
            random_key = np.random.randint(0, high=99999)
            # Read video file frame
            #cap = cv2.VideoCapture('/home/wuqianliang/test/VID_20171013_121412.mp4')
            # start capture video
            print('start test on lfw dataset')
            def get_feature(image_x):
                img=cv2.imread(image_x)
                img=img[:,:,0:3]
                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                #print(bounding_boxes)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0 and len(bounding_boxes)==1:
                    bbox= bounding_boxes[0]
                    det = bbox[0:5]
                    detect_confidence = det[4]
                    #print(detect_confidence)
                    if detect_confidence > 0.8:#0.8:
                        left    =int(bbox[0])
                        top     =int(bbox[1])
                        right   =int(bbox[2])
                        bottom  =int(bbox[3])
                        alignedFace=align_dlib.align(
                                160,   # 96x96x3
                                img,
                                dlib.rectangle(left,top,right,bottom),
                                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
                        #print(alignedFace.shape)
                        cropped=alignedFace.reshape((1,160,160,3))
                        emb_dict = list(sess.run([embeddings], feed_dict={images_placeholder: np.array(cropped), phase_train_placeholder: False })[0])
                        return emb_dict
                return None


            for thershold in np.arange(0.25, 0.35, 0.01):
                True_Positive = 0;
                True_Negative = 0;
                False_Positive = 0;
                False_Negative = 0;

                # Positive Test
                f_positive = open(POSITIVE_TEST_FILE, "r");
                PositiveDataList = f_positive.readlines(); 
                f_positive.close( );
                f_negative = open(NEGATIVE_TEST_FILE, "r");
                NegativeDataList = f_negative.readlines(); 
                f_negative.close( );
                for index in range(len(PositiveDataList)):
                    filepath_1 = PositiveDataList[index].split(' ')[0];
                    filepath_2 = PositiveDataList[index].split(' ')[1][:-1];
                    feature_1 = get_feature(DATA_BASE + filepath_1);
                    feature_2 = get_feature(DATA_BASE + filepath_2); 
                    if feature_1 is None or feature_2 is None:
                        print(filepath_1,filepath_2,'not detected')            
                        continue 
                    result = compare_pic(feature_1, feature_2);
                    if result >= thershold:
                        print ('Same Guy\n')
                        True_Positive += 1;
                    else:
                        print( 'Wrong\n\n')
                        False_Positive += 1;
                
                for index in range(len(NegativeDataList)):
                    filepath_1 = NegativeDataList[index].split(' ')[0];
                    filepath_2 = NegativeDataList[index].split(' ')[1][:-1];
                    feature_1 = get_feature(DATA_BASE + filepath_1);
                    feature_2 = get_feature(DATA_BASE + filepath_2);

                    if feature_1 is None or feature_2 is None:
                        print(filepath_1,filepath_2,'not detected')
                        continue 
                    result = compare_pic(feature_1, feature_2);
                    if result >= thershold:
                        print ('Wrong Guy\n')
                        False_Negative += 1;
                    else:
                        print ('Correct\n')
                        True_Negative += 1; 

                print ("thershold: " + str(thershold));
                print ("Accuracy: " + str(float(True_Positive + True_Negative)/TEST_SUM) + " %");
                print ("True_Positive: " + str(float(True_Positive)/TEST_SUM) + " %");
                print ("True_Negative: " + str(float(True_Negative)/TEST_SUM) + " %");
                print ("False_Positive: " + str(float(False_Positive)/TEST_SUM) + " %");
                print ("False_Negative: " + str(float(False_Negative)/TEST_SUM) + " %");


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
