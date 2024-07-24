conda create -n aloha python=3.9
conda activate aloha

pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyquaternion -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install rospkg -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pexpect -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mujoco==2.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install dm_control==1.0.14 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install packaging -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ipython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers
cd detr && pip install -e . && cd ..

# git clone https://gitee.com/twilightLZL/CLIP.git
# cd CLIP
# python setup.py develop
# cd ..
