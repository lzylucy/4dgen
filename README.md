# Geometry-aware 4D Video Generation for Robot Manipulation

[[Project page]](https://robot4dgen.github.io/)
[[Paper]](https://arxiv.org/abs/2507.01099)
[[Dataset]](https://real.stanford.edu/4dgen)
[[Checkpoints]](https://real.stanford.edu/4dgen/checkpoints/)

Dataset and checkpoints can also be accessed and downloaded from huggingface now!
[[Dataset]](https://huggingface.co/datasets/Zeyi/4dgen-dataset)
[[Checkpoints]](https://huggingface.co/Zeyi/4dgen-ckpts)

<p align="center">
<img width="90%" src="assets/teaser.jpg">
</p>

[Zeyi Liu](http://lzylucy.github.io/) <sup>1</sup>,
[Shuang Li](https://shuangli59.github.io/)<sup>1</sup>,
[Eric Cousineau](https://www.eacousineau.com/)<sup>2</sup>,
[Siyuan Feng](https://www.cs.cmu.edu/~sfeng/)<sup>2</sup>,
[Benjamin Burchfiel](https://www.tri.global/about-us/dr-ben-burchfiel)<sup>2</sup>,
[Shuran Song](https://shurans.github.io/)<sup>1</sup>

<sup>1</sup>Stanford University,
<sup>2</sup>Toyota Research Institute

## Install Dependencies
To install the required dependencies, we recommend using conda / mamba environment.
```console
$ cd 4dgen
$ conda env create -f environment.yml
$ conda activate video_policy
(video_policy)$ conda install pytorch3d
```
Tested on Ubuntu 22.04, CUDA Version 12.2.

## Download checkpoints and dataset
<p align="center">
<img width="90%" src="assets/data.png">
</p>

We release 50 demonstrations each for 3 tasks StoreCerealBoxUnderShelf, PutSpatulaOnTableFromUtensilCrock, and PlaceAppleFromBowlIntoBin in the Large Behavior Model (LBM) simulation. Each demonstration includes RGB-D observations (and robot actions) from 16 different camera poses, sampled from the upper hemisphere positioned above the workstation. All data can be downloaded [here](https://real.stanford.edu/4dgen/data/).

Checkpoints for pre-trained Stable Video Diffusion (SVD) and VAE can be found [here](https://real.stanford.edu/4dgen/checkpoints/).

In addition, we release fine-tuned VAE encoders for pointmaps and RGB images on our simulation dataset [here](https://real.stanford.edu/4dgen/checkpoints/VAE/), which outputs better latent representations for the specific robotic tasks they're trained on.

Checkpoints for 4D video generation models can be found [here](https://real.stanford.edu/4dgen/checkpoints/outputs/).


## Finetune VAE
```console
(video_policy)$ CUDA_VISIBLE_DEVICES=<GPU-device-ids> HYDRA_FULL_ERROR=1 python scripts/train.py --config-name=finetune_autoencoder_workspace
```

## Train 4D generation model
```console
(video_policy)$ CUDA_VISIBLE_DEVICES=<GPU-device-ids> HYDRA_FULL_ERROR=1 python scripts/train.py --config-name=finetune_svd_lightning_workspace
```
Tested on 4 NVIDIA A6000 GPUs with 48GB memory each, with batch size 1. The training takes about 2 days to finish.

## Run inference example
```console
(video_policy)$ python notebooks/eval.py
```

## Citation
If you find this codebase useful, please consider citing our work:
```bibtex
@article{liu2025geometry,
  title={Geometry-aware 4D Video Generation for Robot Manipulation},
  author={Liu, Zeyi and Li, Shuang and Cousineau, Eric and Feng, Siyuan and Burchfiel, Benjamin and Song, Shuran},
  journal={arXiv preprint arXiv:2507.01099},
  year={2025}
}
```
