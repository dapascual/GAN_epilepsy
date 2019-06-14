# Synthetic Epileptic Brain Activities Using Generative Adversarial Networks

This repository contains the GAN model used in the paper "Synthetic Epileptic Brain Activities Using Generative Adversarial Networks". Our code is constructed using as a starting point the "SEGAN: Speech Enhancement Generative Adversarial Network".

In this work we generate 4-second long EEG samples of epileptic seizures for two electrode channels (i.e. F7T3 and F8T4) and use these samples to train a Random Forest classifier for epilepsy detection. We use the database from the EPILEPSIAE project, which contains 30 patients with a total of 277 seizures. Unfortunately, this database is only available upon purchase.

### Setup

The project is developped using Python 2.7 and Tensorflow 0.12.1. To install the dependencies you can run in your terminal:
```
pip install -r requirements.txt
```
The code is prepared to read the data from Matlab (.mat) files.


### Folder structure

In order to run the model, the data should be stored following this structure:
  - ./data/trainset: All the training samples in .mat format. Each file should contain two arrays named `seiz` and `non_seiz`.The files should be named: 	`pat_<id>_GAN_<i>.mat`, where `<id>` is the id number of the patient and `<i>` is the sample number.
  - ./test_set:All the non seizure samples to transform into seizures. These samples should be contained inside of .mat files in an array called `non_seiz`. The files should be named: `pat_<id>_GAN_test_<i>.mat`, where `<id>` is the patient id and `<i>` is the sample number. The sample files should be place inside folders, one folder per patient with the folder names: `pat_<id>`

The rest of the folders will contain the data produced at different stages:
 - ./data/TFrecords: destination of the TFrecords files created from the trainset.
 - ./gan_results: destination of the model files resulting from the training. These files should be stored inside of folders named: `leave_out_pat_<id>`
 - ./test_set_transformed: destination of the seizures transformed from non-seizures.
  
  ### Run training and test of GAN
  
  To run the code it is necessary to firstly create the TFrecords files:
  
  ```
  ./create_tfrecords.sh
  ```
  
  Once the TFrecords are created the model can be trained and launched using:
  
  ```
  ./run_training_and_test.sh
  ```
  
  This script will train and test the model for all the patients, i.e., leaving out one patient from the training set and generating seizure samples for that one patient. Since this process is repeated for all the patients, it may take a long time to complete. In order to train and test for only one patient, change the entry `list` in the script.
  
## Evaluation of synthetic seizures

To evaluate the resulting synthetic samples, go to the `evaluation` folder, where the matlab scripts are located. Before being able to run these scripts, place the data according to the following folder structure:

 - ./evaluation/data/RF_testset: This folder should contain one file per patient named `pat_<id>_testRF.mat`, which is the test set for the Random Forest classifier. These files should contain a matrix named `X_seiz` of size Nx2048, and a matrix named `X_non_seiz` of size 2Nx2048, where N is the number seizure samples of the patient without overlap. 
 - ./evaluation/data/RF_trainset_nonseiz: This folder should contain one file per patient named `pat_<id>_trainRF_nonseiz.mat`. These files should contain a matrix called `X_non_seiz` of size Mx2048, where M is the number of non seizure samples considered for the patient. M is advised to range between 2000 and 6000.
 - ./evaluation/data/RF_trainset_seiz: This folder should contain one file per patient named `pat_<id>_trainRF_seiz.mat`. These files should contain a matrix called `X_seiz` of size Nx2048, where N is the number of seizure samples of the patient.

Once the data is placed in the correct folders, the two scripts to launch are `baseline_for_GAN.m`, which performs the baseline evaluation and stores the results in the file `Results_baseline_GAN.mat`, and `test_GAN.m`, which performs the evaluation of the synthetic seizure samples and stores the result in `Results_test_GAN.mat`.

