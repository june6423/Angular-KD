## Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity

[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/poster/118239)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org/abs/<TODO:_ADD_LINK>)

This repository is the official PyTorch implementation for our NeurIPS 2025 paper: **Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity**.

We introduce **Angular-KD**, a simple yet effective augmentation technique that generates diverse "virtual" teacher views from a single teacher model. By creating these views via angular diversity, Angular-KD enables the student model to learn richer, more comprehensive representations during knowledge distillation.

---

## 📣 News
* **[2025.07.19]** 🚀 Our paper has been accepted to **NeurIPS 2025**!

---

## ⚙️ Installation

We recommend using `conda` for environment management.

1.  **Create a conda environment:**
    ```bash
    conda create -n angularkd python=3.6 -y
    conda activate angularkd
    ```

2.  **Install dependencies:**
    (The required versions are specified below)
    ```bash
    # Example for CUDA 11.1 (Adjust for your environment)
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 
    # Install other requirements
    pip install -r requirements.txt
    ```

3.  **Install the package:**
    ```bash
    python setup.py develop
    ```

---

## 🚀 Getting Started

### 1. (Optional) Wandb Setup

This project uses [Weights & Biases (wandb)](https://wandb.ai/home) for logging.

* To disable wandb, set `CFG.LOG.WANDB = False` in `mdistiller/engine/cfg.py`.
* To use wandb, run `wandb login` to link your account.

### 2. Dataset

The CIFAR-100 dataset will be downloaded automatically by the script. No preparation is needed.

### 3. Pretrained Teacher Models

Download the teacher checkpoints required for training on CIFAR-100.

* Download `cifar_teachers.tar` from the [mdistiller releases](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints).
* Unpack the archive and move the checkpoints to the `./download_ckpts` directory.

    ```bash
    # After downloading the .tar file
    tar xvf cifar_teachers.tar
    mv cifar_teachers ./download_ckpts
    ```

---

## 📈 Training

You can train models using the `tools/train.py` script.

**Example 1: Train Angular-KD (using CRD loss) from scratch**

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
    --cfg configs/cifar100/angularkd/crd.yaml
```

**Example 2: Two-stage training (Pretrain Teacher + Distill)**

- **Step 1: Pretrain the teacher model**
```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
    --cfg configs/cifar100/pretrain.yaml
```

- **Step 2: Run Angular-KD using the pretrained teacher.** The ```--pretrained_ckpt``` argument shold point to the checkpoint saved in Step 1(e.g. ```output/cifar100_baselines/pretrain,resnet32x4/latest```).
```
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
    --cfg configs/cifar100/angularkd/crd.yaml \
    --pretrained_ckpt output/cifar100_baselines/pretrain,resnet32x4/latest
```

**Tip: Overriding Configs**
You can change settings directly from the command line:
```
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
    --cfg configs/cifar100/angularkd/crd.yaml \
    SOLVER.BATCH_SIZE 128 \
    SOLVER.LR 0.1
```


---
## 📊 Evaluation
```
# evaluate students
python3 tools/eval.p -m resnet8x4 -c download_ckpts/dkd_resnet8x4 # dkd-resnet8x4 on cifar100
python3 tools/eval.p -m MobileNetV1 -c download_ckpts/imgnet_dkd_mv1 -d imagenet # dkd-mv1 on imagenet
python3 tools/eval.p -m model_name -c output/your_exp/student_best # your checkpoints
```

---
## 📜 Citation
If you find this work helpful, please cite our paper:


```
@inproceedings{Yu2025angular,
  title={Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity},
  author={Seonghoon Yu*, Dongjun Nam*, Dina Katabi and Jeany Son},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---
## 🙏 Acknowledgement
This code is built upon these excellent repositories. We thank their authors for open-sourcing their work.
- [mdistiller](https://github.com/megvii-research/mdistiller)
- [Multi-Level-logit-Distillation](https://github.com/Jin-Ying/Multi-Level-Logit-Distillation)
- [CRD](https://github.com/HobbitLong/RepDistiller)
-  [ReviewKD](https://github.com/dvlab-research/ReviewKD)
