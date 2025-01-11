# bash Miniconda3-latest-Linux-x86_64.sh
# source /root/.bashrc
# conda install mkl
# pip install -U openmim
# mim install mmengine
# mim install "mmcv>=2.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmpretrain>=1.0.0rc8"
mim install mmdet
pip install -v -e .
pip install ftfy