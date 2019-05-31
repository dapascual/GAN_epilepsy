from __future__ import print_function
import tensorflow as tf
from generator import *
from discriminator import *
import numpy as np
from data_loader import read_and_decode
from bnorm import VBN
from ops import *
import timeit
import os

class Model(object):

    def __init__(self, name='BaseModel'):
        self.name = name

    ### SAVE and LOAD model functions ###
    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess, os.path.join(save_path, model_name))                        

    def load(self, save_path, model_file=None):
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True


class GAN(Model):
    """ Generative Adversarial Network """
    def __init__(self, sess, args, devices, infer=False, name='GAN'):
        ### MODEL PARAMETERS ###
        super(GAN, self).__init__(name)
        self.args = args
        self.sess = sess
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.devices = devices
        # clip D values
        self.d_clip_weights = False
        # apply VBN
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        self.disc_updates = 1
        # canvas size
        self.canvas_size = args.canvas_size
        # num fmaps for AutoEncoder
        self.g_enc_depths = [64, 64, 128, 128, 256, 256, 512, 1024]
        self.g_dec_depths = self.g_enc_depths[:-1][::-1] + [1] 
        # Define D fmaps
        self.d_num_fmaps = [64, 64, 128, 128, 256, 256, 512, 1024] 
        self.e2e_dataset = args.e2e_dataset
        # G's supervised loss weight
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        # define the functions
        self.discriminator = discriminator
        # register G non linearity
        self.generator = Generator(self)
        self.build_model(args)

    ### BUILD MODEL AND OPTMIZER ###
    def build_model(self, config):
        all_d_grads = []
        all_g_grads = []
        d_opt = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta_1, beta2=config.beta_2)
        g_opt = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta_1, beta2=config.beta_2)

        for idx, device in enumerate(self.devices):
            with tf.device("/%s" % device):
                with tf.name_scope("device_%s" % idx):
                    with variables_on_gpu0():
                        self.build_model_single_gpu(idx)
                        d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=self.d_vars)	 
                        g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=self.g_vars)	
                        all_d_grads.append(d_grads)
                        all_g_grads.append(g_grads)
                        tf.get_variable_scope().reuse_variables()
	
        avg_d_grads = average_gradients(all_d_grads)
        avg_g_grads = average_gradients(all_g_grads)
        self.d_opt = d_opt.apply_gradients(avg_d_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)

    #### FORMULATE LOSS FUNCTIONS ###
    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer([self.e2e_dataset])
            self.get_seiz, self.get_nonseiz = read_and_decode(filename_queue, self.canvas_size)
        # load the data to input pipeline
        seiz_batch, nonseiz_batch = tf.train.shuffle_batch([self.get_seiz, self.get_nonseiz],
                                             batch_size=self.batch_size,
                                             num_threads=2, capacity=1000 + 3 * self.batch_size,
                                             min_after_dequeue=1000, name='seiz_and_nonseiz')
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_seiz = []
            self.gtruth_nonseiz = []

        self.gtruth_seiz.append(seiz_batch)
        self.gtruth_nonseiz.append(nonseiz_batch)

        # add channels dimension to manipulate in D and G
        seiz_batch = tf.expand_dims(seiz_batch, -1)
        nonseiz_batch = tf.expand_dims(nonseiz_batch, -1)
        if gpu_idx == 0:
            ref_Gs = self.generator(nonseiz_batch, is_ref=True)
            print('num of G returned: ', len(ref_Gs))
            self.reference_G = ref_Gs[0]
            self.ref_z = ref_Gs[1]

            # make a dummy copy of discriminator to create the variables
            dummy_joint = tf.concat(2, [seiz_batch, nonseiz_batch])
            dummy = discriminator(self, dummy_joint, reuse=False)
        # build generator
        G, z = self.generator(nonseiz_batch, is_ref=False)
        self.Gs.append(G)
        self.zs.append(z)

        D_rl_joint = tf.concat(2, [seiz_batch, nonseiz_batch])
        D_fk_joint = tf.concat(2, [G, nonseiz_batch])
        # build discriminator
        d_rl_logits = discriminator(self, D_rl_joint, reuse=True)
        d_fk_logits = discriminator(self, D_fk_joint, reuse=True)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            self.g_adv_losses = []
            self.d_rl_losses = []
            self.d_fk_losses = []
            self.d_losses = []

        ### Discriminator loss ###
        d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
        d_fk_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))
        g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))
        d_loss = d_rl_loss + d_fk_loss
        ### Generator loss ###
        g_l1_loss = self.l1_lambda * tf.reduce_mean(tf.abs(tf.sub(G, seiz_batch)))
        g_loss = g_adv_loss + g_l1_loss

        self.g_l1_losses.append(g_l1_loss)
        self.g_adv_losses.append(g_adv_loss)
        self.g_losses.append(g_loss)
        self.d_rl_losses.append(d_rl_loss)
        self.d_fk_losses.append(d_fk_loss)
        self.d_losses.append(d_loss)

        if gpu_idx == 0:
            self.get_vars()

    ### GET D AND G VARIABLES ###
    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars_dict = {}
        self.g_vars_dict = {}
        for var in t_vars:
            if var.name.startswith('d_'):
                self.d_vars_dict[var.name] = var
            if var.name.startswith('g_'):
                self.g_vars_dict[var.name] = var
        self.d_vars = self.d_vars_dict.values()
        self.g_vars = self.g_vars_dict.values()
        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars]
        else:
            print('Not clipping D weights')

    ### VIRTUAL BATCH NORMALIZATION ###
    def vbn(self, tensor, name):
        if self.disable_vbn:
            class Dummy(object):
                # Do nothing here, no bnorm
                def __init__(self, tensor, ignored):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    ### TRAIN MODEL ###
    def train(self, config, devices):
        ### INITIALIZATION ###
        print('Initializing optimizers...')
        d_opt = self.d_opt
        g_opt = self.g_opt
        num_devices = len(devices)
        init = tf.global_variables_initializer()       
        print('Initializing variables...')
        self.sess.run(init)        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('Sampling some seizures to store sample references...')
        sample_nonseiz, sample_seiz, \
        sample_z = self.sess.run([self.gtruth_nonseiz[0], self.gtruth_seiz[0], self.zs[0]])
        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size
        print('Batches per epoch: ', num_batches)
        ### LOAD MODEL IF EXISTING ##
        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        
        ### TRAINING ###
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                # Train discriminator
                for d_iter in range(self.disc_updates):
                    _d_opt, d_fk_loss, d_rl_loss = self.sess.run([d_opt, self.d_fk_losses[0],
                                               self.d_rl_losses[0]])
                    if self.d_clip_weights:
                       self.sess.run(self.d_clip)
                # Train generator
                _g_opt, g_adv_loss, g_l1_loss = self.sess.run([g_opt, self.g_adv_losses[0],
                                               self.g_l1_losses[0]])
                end = timeit.default_timer()
                batch_timings.append(end - start)
                
                # Print information in console for debugging
                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, '
                      ' g_adv_loss = {:.5f}, g_l1_loss = {:.5f}, time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter, config.epoch * num_batches,
                                                    curr_epoch, d_rl_loss,
                                                    d_fk_loss, g_adv_loss,
                                                    g_l1_loss,end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    fdict = {self.gtruth_nonseiz[0]:sample_nonseiz,
                             self.zs[0]:sample_z}
                    canvas_w = self.sess.run(self.Gs[0],
                                             feed_dict=fdict)
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m,
                                                           np.max(canvas_w[m]),
                                                           np.min(canvas_w[m])))   
                # Optionally, deactivate L1 regularization                                           
                if batch_idx >= num_batches:
                    curr_epoch += 1
                    batch_idx = 0
                    # check if we have to deactivate L1
                    if curr_epoch >= config.l1_remove_epoch and self.deactivated_l1 == False:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    
                # Done training
                if curr_epoch >= config.epoch:
                    print('Done training; epoch limit {} reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    break

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)

    ### TRANSFORM NON-SEIZURE INTO SEIZURE ###
    def transform(self, x):
        """ transform a non-seizure x into seizure
            x: numpy array containing the normalized nonseiz waveform
        """
        t_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i  < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = (self.canvas_size) - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]
            print('Transforming chunk {} -> {}'.format(beg_i, beg_i + length))
            fdict = {self.gtruth_nonseiz[0]:x_}
            canvas_w = self.sess.run(self.Gs[0], feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                # get rid of last padded samples
                canvas_w = canvas_w[:-pad]
            if t_res is None:
                t_res = canvas_w
            else:
                t_res = np.concatenate((t_res, canvas_w))
        return t_res

