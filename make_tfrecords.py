from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import argparse
import codecs
import timeit
import struct
import toml
import re
import sys
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encoder_proc(seiz_filename, out_file):
    """ Read the seizure and non seizure files and write to TFRecords.
        out_file: TFRecordWriter.
    """
    mat_content = sio.loadmat(seiz_filename)
    seiz_signals = np.array(mat_content['seiz'], dtype=np.float32)
    nonseiz_signals = np.array(mat_content['non_seiz'], dtype=np.float32)				
    assert seiz_signals.shape == nonseiz_signals.shape, nonseiz_signals.shape

    for (seiz, nonseiz) in zip(seiz_signals, nonseiz_signals):
        seiz_raw = seiz.tostring()
        nonseiz_raw = nonseiz.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'seiz_raw': _bytes_feature(seiz_raw),		
            'nonseiz_raw': _bytes_feature(nonseiz_raw)}))
        out_file.write(example.SerializeToString())

def main(opts):
    #### SET UP OUTPUT FILEPATH ####
    # make save path if it does not exist
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    
    pat = opts.patient
    out_filepath = os.path.join(opts.save_path, opts.out_file)
    # if wrong extension or no extension appended, put .tfrecords:
    if os.path.splitext(out_filepath)[1] != '.tfrecords':
        out_filepath += '.tfrecords'

    else:
        out_filename, ext = os.path.splitext(out_filepath)
        out_filepath = out_filename + '_leave_out_' + pat + ext
    # check if out_file exists and if force flag is set
    if os.path.exists(out_filepath) and not opts.force_gen:
        raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to '
                         'overwrite. Skipping this speaker.'.format(out_filepath))
    elif os.path.exists(out_filepath) and opts.force_gen:
        print('Will overwrite previously existing tfrecords')
        os.unlink(out_filepath)				

    #### CREATE TFRECORDS ####
    with open(opts.cfg) as cfh:		
        # read and config file
        cfg_desc = toml.loads(cfh.read())
        beg_enc_t = timeit.default_timer()
        out_file = tf.python_io.TFRecordWriter(out_filepath)	
        
        for dset_i, (dset, dset_desc) in enumerate(cfg_desc.items()):
            print('-' * 50)
            seiz_dir = dset_desc['seiz']
            seiz_files = [os.path.join(seiz_dir, seiz) for seiz in
                           os.listdir(seiz_dir) if seiz.endswith('.mat') and not seiz.startswith(pat)]
            nfiles = len(seiz_files)
            for m, seiz_file in enumerate(seiz_files):
                print('Processing seiz file {}/{} {}{}'.format(m + 1, nfiles, seiz_file,' '*10), end='\r')
                sys.stdout.flush()
                encoder_proc(seiz_file, out_file)
        out_file.close()
        end_enc_t = timeit.default_timer() - beg_enc_t
        print('')
        print('*' * 50)
        print('Total processing and writing time: {} s'.format(end_enc_t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert set to TFRecords')
    parser.add_argument('--cfg', type=str, default='cfg/e2e_maker.cfg',
                        help='File containing the description of datasets '
                             'to extract the info to make the TFRecords.')
    parser.add_argument('--save_path', type=str, default='data/TFrecords/',
                        help='Path to save the dataset')
    parser.add_argument('--out_file', type=str, default='gan.tfrecords',
                        help='Output filename')
    parser.add_argument('--force-gen', dest='force_gen', action='store_true',
                        help='Flag to force overwriting existing dataset.')
    parser.add_argument('--patient', type=str, help='Patient left out')
    parser.set_defaults(force_gen=False)
    opts = parser.parse_args()
    main(opts)
