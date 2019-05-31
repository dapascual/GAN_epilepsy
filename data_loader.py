from __future__ import print_function
import tensorflow as tf
from ops import *
import numpy as np

def read_and_decode(filename_queue, canvas_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'seiz_raw': tf.FixedLenFeature([], tf.string),
                'nonseiz_raw': tf.FixedLenFeature([], tf.string),
            })
    seiz = tf.decode_raw(features['seiz_raw'], tf.float32)
    seiz.set_shape(canvas_size)
    seiz = tf.cast(seiz, tf.float32)
    seiz = (2./65535.) * tf.cast((seiz), tf.float32)
    nonseiz = tf.decode_raw(features['nonseiz_raw'], tf.float32)
    nonseiz.set_shape(canvas_size)
    nonseiz = tf.cast(nonseiz, tf.float32)
    nonseiz = (2./65535.) * tf.cast((nonseiz), tf.float32)
    
    return seiz, nonseiz
