exp_id=isaac_gripper_novae
config_name=isaac_gripper_novae

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=172.17.167.51
master_port=29513

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir ./datasets/isaac_gripper \
    --num_nodes $nnodes \
    --load_dir outputs/isaac_gripper_novae/policy_iter20000.ckpt \
    #--debug \