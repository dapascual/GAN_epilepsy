from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from ops import *
import numpy as np

def discriminator(self, wave_in, reuse=False):
        """
        wave_in: waveform input
        """
        ### INITIALIZATION ###
        in_dims = wave_in.get_shape().as_list()
        hi = wave_in
        if len(in_dims) == 2:
            hi = tf.expand_dims(wave_in, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Discriminator input must be 2-D or 3-D')

        ### DISCRIMINATOR NETWORK ###
        with tf.variable_scope('d_model') as scope:
            if reuse:
                scope.reuse_variables()           
            else:
                print('*** Discriminator summary ***')
            for block_idx, fmaps in enumerate(self.d_num_fmaps):
                with tf.variable_scope('d_block_{}'.format(block_idx)):     
                    if not reuse:
                        print('D block {} input shape: {}'.format(block_idx, hi.get_shape()))   
                    downconv_init = tf.truncated_normal_initializer(stddev=0.02)
                    hi = sndownconv(hi, self.d_num_fmaps[block_idx], kwidth=31, pool=2,
                                    init=downconv_init)     
                    if not reuse:
                        print('downconved shape: {}'.format(hi.get_shape()), end=' *** ')
                    hi = self.vbn(hi, 'd_vbn_{}'.format(block_idx))
                    hi = leakyrelu(hi)
          
            if not reuse:
                print('discriminator deconved shape: ', hi.get_shape())
            d_logit_out = conv1d(hi, kwidth=1, num_kernels=1,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 name='logits_conv')
            d_logit_out = tf.squeeze(d_logit_out)
            d_logit_out = fully_connected(d_logit_out, 1, activation_fn=None)
            if not reuse:
                print('discriminator output shape: ', d_logit_out.get_shape())
                print('*****************************')
            return d_logit_out
