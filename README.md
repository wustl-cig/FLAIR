# FLAIR
This is the official Repo for WACV2025 paper "FLAIR: A Conditional Diffusion Framework with Applications to Face Video Restoration" ([paper](https://arxiv.org/abs/2311.15445))
## Installation

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## Download Pre-trained Models

[Google Drive](https://drive.google.com/file/d/1dmF7pjN8N-T1UXdijO7kHGjqREAx0a9L/view?usp=sharing)

Place them in `./checkpoints`
## Usage

```bash
python scripts/video_sample.py gaussian-demo --device cuda

python scripts/video_sample.py x8-bicubic-demo --device cuda

python scripts/video_sample.py x16-bicubic-demo --device cuda

python scripts/video_sample.py jpeg-demo --device cuda
```
