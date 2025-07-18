### Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity

### Installation

Environments:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

### Getting started

0. Wandb as the logger

- The registeration: <https://wandb.ai/home>.
- If you don't want wandb as your logger, set `CFG.LOG.WANDB` as `False` at `mdistiller/engine/cfg.py`.

1. Evaluation

- You can evaluate the performance of our models or models trained by yourself.

- Our models are at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>, please download the checkpoints to `./download_ckpts`


2. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for instance, our Angular-KD method
  CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --cfg configs/cifar100/crd_ours.yaml

  # you can also change settings at command line
  CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --cfg configs/cifar100/crd_ours.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
  ```

3. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # for instance, our Angular-KD method.
  CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --cfg configs/imagenet/r34_r18/crd.yaml
  ```



# Acknowledgement

- Thanks for mdistiller, CRD, MLKD and Review KD. This code is built on [mdistiller](https://github.com/megvii-research/mdistiller), [Multi-Level-logit-Distillation](https://github.com/Jin-Ying/Multi-Level-Logit-Distillation), [CRD](https://github.com/HobbitLong/RepDistiller) and [ReviewKD](https://github.com/dvlab-research/ReviewKD).
