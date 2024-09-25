conda create -n VIRT python=3.8
conda activate VIRT

# install Isaac Gym
# Move to isaacgym/python/
# pip install -e .

pip install torch==2.3.1 torchvision==0.18.1
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
pip install xformers==0.0.27
pip install tensorboard
pip install tqdm
cd detr && pip install -e . && cd ..

python setup.py develop