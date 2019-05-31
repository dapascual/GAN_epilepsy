from __future__ import print_function
import tensorflow as tf
import numpy as np
from model import GAN
import os
from tensorflow.python.client import device_lib
import scipy.io as sio

devices = device_lib.list_local_devices()

### MODEL PARAMETERS ###
flags = tf.app.flags
flags.DEFINE_integer("seed",111, "Random seed (Def: 111).")
flags.DEFINE_integer("epoch", 100, "Epochs to train (Def: 100).")
flags.DEFINE_integer("batch_size", 100, "Batch size (Def: 100).")
flags.DEFINE_integer("save_freq", 50, "Batch save freq (Def: 50).")  #!
flags.DEFINE_integer("canvas_size", 2048, "Canvas size (Def: 2048).")   #flags.DEFINE_integer("canvas_size", 2**14, "Canvas size (Def: 2^14).")
flags.DEFINE_integer("l1_remove_epoch", 350, "Epoch where L1 in G is "
                                           "removed (Def: 350).")     
flags.DEFINE_float("init_l1_weight", 100., "Init L1 lambda (Def: 100)") 
flags.DEFINE_string("save_path", "gan_results", "Path to save out model ")
flags.DEFINE_float("g_learning_rate", 0.0001, "G learning_rate (Def: 0.0001)") 
flags.DEFINE_float("d_learning_rate", 0.0004, "D learning_rate (Def: 0.0004)") 
flags.DEFINE_float("beta_1", 0, "Adam beta 1 (Def: 0.)")    
flags.DEFINE_float("beta_2", 0.9, "Adam beta 2 (Def: 0.5)")
flags.DEFINE_string("e2e_dataset", "data/TFrecords/gan.tfrecords", "TFRecords")
flags.DEFINE_string("save_transformed_path", "test_set_transformed", "Path to save seizures")
flags.DEFINE_string("test_set", None, "name of test wav (it won't train)")
flags.DEFINE_string("weights", None, "Weights file")
FLAGS = flags.FLAGS


def main(_):

    ### INITIALIZATION
    print('Parsed arguments: ', FLAGS.__flags)
    # make save path if it is required
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    
    np.random.seed(FLAGS.seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    udevices = []
    for device in devices:
        if len(devices) > 1 and 'cpu' in device.name:
            # Use cpu only when we dont have gpus
            continue
        print('Using device: ', device.name)
        udevices.append(device.name)

    # execute the session
    with tf.Session(config=config) as sess:
        model = GAN(sess, FLAGS, udevices)
        ### TRAIN MODEL ###
        if FLAGS.test_set is None:
            model.train(FLAGS, udevices)
        ### TRANSFORM NON-SEIZURES INTO SEIZURES ###
        else:
            if FLAGS.weights is None:
                raise ValueError('weights must be specified!')
            print('Loading model weights...')
            if not os.path.exists(FLAGS.save_transformed_path):
                os.makedirs(FLAGS.save_transformed_path)
            model.load(FLAGS.save_path, FLAGS.weights)
	    
            nonseiz_files = [os.path.join(FLAGS.test_set, sample) for sample in os.listdir(FLAGS.test_set) if sample.endswith('.mat')] 
            for m, sample_file in enumerate(nonseiz_files):	
                mat_content = sio.loadmat(sample_file)
                sample_data = np.array(mat_content['non_seiz'], dtype=np.float32)
                sample_data = np.reshape(sample_data, (FLAGS.canvas_size))
                samplename = sample_file.split('/')[-1]           
                wave = sample_data.astype(np.float32)
                wave = (2./65535.) * (wave)
            
                print('test wave shape: ', wave.shape)
                print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
                seiz_wave = model.transform(wave)
                seiz_wave = (65535./2.)*seiz_wave
                print('seiz wave min:{}  max:{}'.format(np.min(seiz_wave), np.max(seiz_wave)))
                samplename = 'GAN_seizure_' + samplename
                sio.savemat(os.path.join(FLAGS.save_transformed_path, samplename), {'GAN_seiz':seiz_wave})
                print('Done transforming {} and saved to {}'.format(sample_file,
                                     os.path.join(FLAGS.save_transformed_path, samplename)))

if __name__ == '__main__':
    tf.app.run()
