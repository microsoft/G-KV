# G-KV

## Introduction

This library provides comprehensive support for various KV cache compression algorithms, including H2O, SnapKV, R-KV, StreamingLLM, and the proposed G-KV. It is compatible with a wide range of models, such as the Qwen 2 series and the Llama series (versions 1 to 3).


The library also supports post-training for KV cache compression models. It includes a complete GRPO reinforcement learning pipeline, enabling generation with KV cache compression and constructing sparse attention masks for training. Additionally, the library offers pipelines for supervised fine-tuning (SFT) and distillation training, ensuring adaptability and optimization of models under KV cache compression settings.


## Environment

python >= 3.10

```
pip install -r requirement.txt
pip install flash-attn==2.7.4.post1 --no-cache-dir
```

## Quick Start

The scripts contain detailed descriptions of parameter settings.

Inference

```
bash scripts/inference.sh
```

Train (SFT or Distillation)

```
bash scripts/sft.sh
```

Train (RL)

```
bash scripts/rl.sh
```

evaluate on LiveCodeBench

```
python datasets/lcb_precess.py

bash scripts/lcb_eval.sh
```