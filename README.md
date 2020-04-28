# OptimalStrategiesAgainstGenerativeAttacks
This repository contains the official code for the paper:

**Optimal Strategies Against Generative Attacks**  
Roy Mor, Erez Peterfreund, Matan Gavish, Amir Globerson  

This work was published as an oral presentation in the International Conference of Learning Representations ([ICLR][iclr]), 2020.

## Table of Contents

* [Setup](#setup)
* [Usage](#usage)
    * [Game value plots](#game-value-plots)
    * [Training GIM on Gaussians](#training-gim-on-gaussians)
    * [Training GIM on Images](#training-gim-on-images)
    * [Evaluating GIM on the authentication task](#evaluating-gim-on-the-authentication-task)
* [Citation](#citation)

## Setup
See requirements.txt for the required packages to using this repo.
## Usage 
### Game value plots
In this section, we describe how to use the theoretic game value functions and plot several game value plots as seen in the paper and presentation.
* To print out the game value for a specific assignment of m,n,k,d, run for example the following command line:
```console
$ python theory/theoretic_game_value.py -m 1 -n 5 -k 10 -d 10
```
* To plot the theoretic game value as a function of n/m for different values of d, run:
```console
$ python plots/plot_game_value_of_n_over_m_for_diff_d.py
```
* To plot the theoretic game value as a function of n/m for different values of rho, run:
```console
$ python plots/plot_game_value_of_n_over_m_for different_rho_values.py
```
* To plot the theoretic game value as a function of rho and delta, run:
```console
$ python plots/plot_game_value_of_rho_delta.py --d 100 --plot_type nash_game_value
```
* To plot the theoretic game value when fixing the attacker to be the ML attacker (described in the paper) as a function of rho and delta, run:
```console
$ python plots/plot_game_value_of_rho_delta.py --d 100 --plot_type ml_attacker_game_value
```
* To plot the difference in game value between the ml attacker and the optimal attacker as a function of rho and delta, run:
```console
$ python plots/plot_game_value_of_rho_delta.py --d 100 --plot_type game_value_diff_ml_vs_opt
```
### Training GIM on Gaussians
To train GIM on Gaussians simply run:
```console
$ python train_gim_on_gaussians.py -o <output directory path>
```
To see the rest of the optional arguments you can run:
```console
$ python train_gim_on_gaussians.py --help
```
The program will create the following directories:

\<output directory path\>/args.json - A json file specifying the arguments for the program.

\<output directory path\>/ckpts/ - The directory where all the training checkpoints are saved. 

\<output directory path\>/logs/ - The directory where all the logs are saved (if there are any).

\<output directory path\>/tb/ - The tensorboard directory

You can monitor the training progress with [tensorboard][tb] by running:
```console
$ tensorboard --logdir <output directory path>/tb/
```

### Training GIM on images
#### Datasets
To train GIM on images you first need to create an image dataset in the following directory structure:

\<dataset root\>/\<split\>/\<group\>/\<class\>/img.jpg

Where:
 
* \<dataset root\> is the root directory of the dataset.
  
* \<split\> is train, val or test.

* \<group\> is a partition of the classes to groups (E.g., in omniglot it's languages,). The existence of this 
hierarchy originated from the structure of the Omniglot and Voxceleb2 dataset.

* \<class\> is the of classes. (E.g characters in Omniglot, or identities in face images). Each class directory contains all the images of that class

You can create both train ('dev' in the raw Voxceleb2 dataset) and val ('test' in the raw Voxceleb2 dataset) Voxceleb2 datasets using: 
```console
$ python data_handling/prepare_voxceleb_dataset.py --src_vid_ds_root <path to the raw voxceleb2 video directory at .../test/mp4/ or /dev/mp4> --dst_img_ds_root <path to the new dataset>
```
For omniglot, simply divide the dataset to train and val directories (see paper for splits and augmentation).

#### Training
Once you have a dataset, you can train GIM using the following command line:
```console
$ python python train_gim_on_imgs.py -o <output dir> --dataset_root <root dir of dataset> --dataset_type <omniglot or voxceleb2>
```
To see the rest of the optional arguments and the hyper-parametrs we used in the paper for training GIM on omniglot and voxceleb2 you can run:
```console
$ python train_gim_on_imgs.py --help
```
The program will create the following directories:

\<output dir\>/args.json - A json file specifying the arguments for the program.

\<output dir\>/ckpts/ - The directory where all the training checkpoints are saved. 

\<output dir\>/logs/ - The directory where all the logs are saved (if there are any).

\<output dir\>/tb/ - The tensorboard directory

You can monitor the training progress with [tensorboard][tb] by running:
```console
$ tensorboard --logdir <output dir>/tb/
```

### Evaluating GIM on the authentication task
To evaluate GIM on the authentication task run:
```console
$ python authentication_eval/eval_gim_on_authentication.py --ds_root <dataset root> --gim_exp_dir <GIM experiment output directory>
```
This will create a .csv file with the authentication results of GIM vs. GIM, GIM vs. Replay, and GIM vs. RS, as seen in the paper.

To see the rest of the optional arguments you can run:
```console
$ python authentication_eval/eval_gim_on_authentication.py --help
```

## Citation
```
@inproceedings{
Mor2020Optimal,
title={Optimal Strategies Against Generative Attacks},
author={Roy Mor and Erez Peterfreund and Matan Gavish and Amir Globerson},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BkgzMCVtPB}
}
```
[iclr]: https://openreview.net/forum?id=BkgzMCVtPB&noteId=BkgzMCVtPB
[tb]: https://www.tensorflow.org/tensorboard