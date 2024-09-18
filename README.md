<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/onshape/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
</div>

# EdgeVLA: An Open-Source Vision-Language-Action Model

# Introduction
We propose training efficient VLA models based on SLMs like Qwen2 with non-autoregressive objective. Our early results shows that these models achieve similar training characteristics compared to much larger counterparts. This repository is a direct fork of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) and [OpenVLA](https://github.com/openvla/openvla). You can train from scratch, finetune or test [our pre-trained models]([https://kscale-public.s3.amazonaws.com/evla_09092024/](https://kscale-public.s3.amazonaws.com/evla_09092024/evla_09092024.tar.gz)). See [our blog](https://medium.com/@budzianowski/34baf4f434ec) or our [report](https://kscale-public.s3.amazonaws.com/evla_09092024/report.pdf) for more details about the architecture.

## Setup
```
conda create --name vla python=3.10
conda activate evla
cd evla
pip install -e .
```
Now you have to add HF TOKEN under `.hf_token` to run models like llama2/3 or qwen2.

## Training/Inference
You can either train your own model from scratch or finetune a model with your own dataset.
We recommend first running the debug mode to see if everything works.
```
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=1235 python vla-scripts/test.py \
 --vla.type "debug" \
 --data_root_dir DATA_ROOT_DIR \
 --run_root_dir RUN_ROOT_DIR
```
The full-scale training can be run with the 'evla' config from `prismatic/conf/vla.py`.


## TODO
1. Remove the hardcoded attention setup.
2. Export model to the HF format.
3. Add support for LoRA.


## Citation

```bibtex
@article{kscale2024evla,
    title={EdgeVLA: Efficient Vision-Language-Action Models},
    author={Paweł Budzianowski, Wesley Ma, Matthew Freed, Jingxiang Mo, Aaron Xie, Viraj Tipnis, Benjamin Bolte},
    year={2024}
} 
```
