# FLAIR

## Installation

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Usage

```bash
python scripts/video_sample.py gaussian-demo --device cuda

python scripts/video_sample.py x8-bicubic-demo --device cuda

python scripts/video_sample.py x16-bicubic-demo --device cuda

python scripts/video_sample.py jpeg-demo --device cuda
```