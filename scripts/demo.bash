source ~/.bashrc
conda activate flair

python scripts/video_sample.py gaussian-demo --device cuda

python scripts/video_sample.py x8-bicubic-demo --device cuda

python scripts/video_sample.py x16-bicubic-demo --device cuda

python scripts/video_sample.py jpeg-demo --device cuda
