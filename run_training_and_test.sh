#!/bin/bash

list='pat_102 pat_7302 pat_8902 pat_11002 pat_16202 pat_21602 pat_21902 pat_22602 pat_23902 pat_26102 pat_30002 pat_30802 pat_32502 pat_32702 pat_45402 pat_46702 pat_55202 pat_56402 pat_58602 pat_59102 pat_75202 pat_79502 pat_85202 pat_92102 pat_93902 pat_96002 pat_103002 pat_109502 pat_111902 pat_114902'


#setenv LD_LIBRARY_PATH /usr/local/cudnn5/lib64:$LD_LIBRARY_PATH

list='pat_102'

for i in $list; do
      
    CUDA_VISIBLE_DEVICES="0,1" python main.py --save_path gan_results/leave_out_$i/ --e2e_dataset data/TFrecords/gan_leave_out_$i.tfrecords
    CUDA_VISIBLE_DEVICES="0,1" python main.py --init_noise_std 0. --save_path gan_results/leave_out_$i/ --weights GAN --test_wav test_set/$i/ --save_transformed_path test_set_transformed/$i/

done
