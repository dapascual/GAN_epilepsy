#!/bin/bash

list='pat_1 pat_2 pat_3 pat_4 pat_5 pat_6 pat_7 pat_8 pat_9 pat_10 pat_11 pat_12 pat_13 pat_14 pat_15 pat_16 pat_17 pat_18 pat_19 pat_20 pat_21 pat_22 pat_23 pat_24 pat_25 pat_26 pat_27 pat_28 pat_29 pat_30'


for i in $list; do
      
    CUDA_VISIBLE_DEVICES="0,1" python main.py --save_path gan_results/leave_out_$i/ --e2e_dataset data/TFrecords/gan_leave_out_$i.tfrecords
    CUDA_VISIBLE_DEVICES="0,1" python main.py --init_noise_std 0. --save_path gan_results/leave_out_$i/ --weights GAN --test_wav test_set/$i/ --save_transformed_path test_set_transformed/$i/

done
