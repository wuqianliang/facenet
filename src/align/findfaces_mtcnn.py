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


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    ###################tensorflow predictor init#############################
    FLAGS = utils.load_tf_flags()
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    embeddings = network.inference(images_placeholder, FLAGS.pool_type, FLAGS.use_lrn, 1.0, phase_train=phase_train_placeholder)
    ema = tf.train.ExponentialMovingAverage(1.0)
    saver = tf.train.Saver(ema.variables_to_restore())
    ckpt = tf.train.get_checkpoint_state(os.path.expanduser(FLAGS.model_dir))
    saver.restore(sess, ckpt.model_checkpoint_path)
    ###########################################################

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 10 # minimum size of face
    threshold = [ 0.5, 0.5, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    # Read video file frame
    cap = cv2.VideoCapture('/home/wuqianliang/test/VID_20171013_121412.mp4')
    while(cap.isOpened()):
        ret, img = cap.read()
        # Add code###########
        img = img[:,:,0:3]
        img = cv2.resize(img, (780, 364), interpolation=cv2.INTER_AREA) 
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            print('detected faces ',nrof_faces)
            for bbox in bounding_boxes:
                det = bbox[0:5]
                detect_confidence = det[4]
                print(det)
                if detect_confidence > 0.8:
                    cv2.rectangle(img,(int(det[0]),int(det[1])),(int(det[2]),int(det[3])),(55,255,155),5)
                    cropped = img[int(det[1]):int(det[3]),int(det[0]):int(det[2]),:]
                    cropped = cv2.resize(cropped, (96, 96), interpolation=cv2.INTER_CUBIC )
                    cv2.imshow('cropped detected face',cropped)
                    #######################
                    
                    #######################
            cv2.imshow('image detected face', img)
            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break
    cap.release()
    cv2.destroyAllWindows()

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
