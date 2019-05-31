from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from ops import *
import numpy as np

class Generator(object):

    def __init__(self, gan):
        self.gan = gan

    def __call__(self, nonseiz_w, is_ref, z_on=True):
        # TODO: remove c_vec
        """ Build the graph propagating (nonseiz_w) --> x
        On first pass will make variables.
        """
        gan = self.gan

        ### NOISE GENERATION ###
        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape, initializer=z_init, trainable=False)
                    if z.device != "/device:GPU:0":
                        # this has to be created into gpu0
                        print('z.device is {}'.format(z.device))
                        assert False
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

        ### INITIALIZATION ###
        if hasattr(gan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
        if is_ref:
            print('*** Building Generator ***')
            print('*** Encoder Layers: ', gan.g_enc_depths)
            print('*** Decoder Layers: ', gan.g_dec_depths)
        
        in_dims = nonseiz_w.get_shape().as_list()
        h_i = nonseiz_w

        if len(in_dims) == 2:
            h_i = tf.expand_dims(nonseiz_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')

        kwidth = 31
        skips = []
        A = []

        ### GENERATOR NETWORK ###
        with tf.variable_scope('g_ae'):
            #ENCODER
            for layer_idx, layer_depth in enumerate(gan.g_enc_depths):                
                h_i_dwn = sndownconv(h_i, layer_depth, kwidth=kwidth,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   name='enc_{}'.format(layer_idx))
                if is_ref:
                    print('Downconv {} -> {}'.format(h_i.get_shape(), h_i_dwn.get_shape()))
                h_i = h_i_dwn
                if layer_idx < len(gan.g_enc_depths) - 1:
                    if is_ref:
                        print('Adding skip connection downconv {}'.format(layer_idx))
                    # store skip connection
                    skips.append(h_i)
                    init = tf.constant(np.ones(h_i.get_shape()), dtype=tf.float32)
                    A.append(tf.get_variable(name='A_{}'.format(layer_idx), initializer=init, dtype=tf.float32)) 
                h_i = leakyrelu(h_i)

            #NOISE IN THE LATENT CODE
            if z_on:
                z = make_z([gan.batch_size, h_i.get_shape().as_list()[1],
                            gan.g_enc_depths[-1]])
                h_i =  tf.concat(2, [z, h_i])

            #DECODER (reverse order)
            for layer_idx, layer_depth in enumerate(gan.g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                
                h_i_dcv = sndeconv(h_i, out_shape, kwidth=kwidth, dilation=2,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 name='dec_{}'.format(layer_idx))               
                if is_ref:
                    print('Deconv {} -> {}'.format(h_i.get_shape(), h_i_dcv.get_shape()))
                h_i = h_i_dcv
                if layer_idx < len(gan.g_dec_depths) - 1:
                    h_i = leakyrelu(h_i)
                    # fuse skip connection
                    skip_ = tf.multiply(skips[-(layer_idx + 1)], A[-(layer_idx + 1)])
                    if is_ref:
                        print('Fusing skip connection of shape {}'.format(skip_.get_shape()))
                    h_i = h_i + skip_ 
                else:
                    if is_ref:
                        print('-- Dec: tanh activation --')
                    h_i = tf.tanh(h_i)
            # OUTPUT
            if is_ref:
                print('Amount of skip connections: ', len(skips))
                print('Last sample shape: ', h_i.get_shape())
                print('*************************')
            gan.generator_built = True
            # ret feats contains the features to be returned
            ret_feats = [h_i]
            if z_on:
                ret_feats.append(z)
            return ret_feats
