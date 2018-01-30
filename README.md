# SeqGAN_tensorflow

This code is used to reproduce the result of synthetic data experiments in "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient" (Yu et.al). It replaces the original tensor array implementation by higher level tensorflow API for easier understanding and better flexibility.

## Introduction
The baisc idea of SeqGAN is to regard sequence generator as an agent in reinforcement learning. To train this agent, it applies REINFORCE algorithm to train the generator and a discriminator is trained to provide the reward. To calculate the reward of partially generated sequence, Monte-Carlo sampling is used to rollout the unfinished sequence to get the estimated reward.

## Prerequisites
   * Python 2.7
   * Tensorflow 1.3
## Run the code
Simplt run `python train.py` will start the training process. It will first pretrain the generator and discriminator then start adversarial training.

## Results

    
