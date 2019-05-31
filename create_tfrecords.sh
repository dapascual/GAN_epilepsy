#!/bin/bash

list='pat_102 pat_7302 pat_8902 pat_11002 pat_16202 pat_21902 pat_22602 pat_23902 pat_26102 pat_30002 pat_30802 pat_32502 pat_32702 pat_45402 pat_46702 pat_55202 pat_56402 pat_58602 pat_59102 pat_75202 pat_79502 pat_85202 pat_92102 pat_93902 pat_96002 pat_103002 pat_109502 pat_111902 pat_114902'

for i in $list; do
    echo $i
    python make_tfrecords.py --force-gen --cfg cfg/e2e_maker.cfg --patient $i --save_path data/TFrecords/
done
