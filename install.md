# 1. 环境准备
# 2. 安装依赖

```bash
conda create -n soul-speak python==3.12.9 -y
conda activate soul-speak
pip install -r requirements.txt
```


sudo apt install ffmpeg
pip3 install torch torchvision torchaudio

pip3 install -U funasr



pip install git+https://github.com/huggingface/transformers \
            torch librosa websockets numpy
accelerate 

git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd CosyVoice
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
pip install Cython

https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B


export PYTHONPATH=/home/chengzi/projects/github/CosyVoice/third_party/Matcha-TTS

python3 webui.py --port 50000 --model_dir iic/CosyVoice-300M