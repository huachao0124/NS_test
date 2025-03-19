# bash Miniconda3-latest-Linux-x86_64.sh
# source /root/.bashrc
# conda install mkl
# pip install -U openmim
# mim install mmengine
# mim install "mmcv>=2.0.0"
export https_proxy=http://10.7.4.2:3128
apt-get install psmisc
apt-get install libsparsehash-dev
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
mim install --no-cache-dir "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmpretrain>=1.0.0rc8"
mim install mmdet
pip install -v -e .
pip install ftfy
TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/mit-han-lab/torchsparse
python -m pip install ujson
pip install info-nce-pytorch
