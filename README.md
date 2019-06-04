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
  - ./data/trainset: All the training samples in .mat format. Each file should contain two arrays named 'seiz' and 'non_seiz'.The files should be named: 	'pat_<id>_GAN_<i>.mat', where <id> is the id number of the patient and <i> is the sample number.
  - ./test_set:All the non seizure samples to transform into seizures. These samples should be contained inside of .mat files in an array called ´non_seiz´. The files should be named: 'pat_<id>_GAN_test_<i>.mat', where <id> is the patient id and <i> is the sample number. The sample files should be place inside folders, one folder per patient with the folder names: 'pat_<id>'

The rest of the folders will contain the data produced at different stages:
 - ./data/TFrecords: destination of the TFrecords files created from the trainset.
 - ./gan_results: destination of the model files resulting from the training. These files should be stored inside of folders named: leave_out_pat_<id>
 - ./test_set_transformed: destination of the seizures transformed from non-seizures.
  
  ### Run
  
  To run the code it is necessary to firstly create the TFrecords files:
  
  ```
  ./create_tfrecords.sh
  ```
  
  Once the TFrecords are created the model can be trained and launched using:
  
  ```
  ./run_training_and_test.sh
  ```
  
  This script will train and test the model for all the patients, i.e., leaving out one patient from the training set and generating seizure samples for that one patient. Since this process is repeated for all the patients, it may take a long time to complete. In order to train and test for only one patient, change the entry `list` in the script.
  
