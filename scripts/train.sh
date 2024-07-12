exp_id=isaac_gripper_novae
config_name=isaac_gripper_novae

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=172.17.167.51
master_port=29515

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --ckpt_dir ./outputs/$exp_id \
    --data_dir ./datasets/isaac_gripper \
    --num_nodes $nnodes \
    #--debug \