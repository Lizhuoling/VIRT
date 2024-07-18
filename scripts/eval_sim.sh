exp_id=isaac_singlebox_vae_chunk300
config_name=isaac_singlebox_vae_chunk300

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=172.17.167.51
master_port=29513

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir ./datasets/isaac_singlebox \
    --num_nodes $nnodes \
    --eval \
    #--load_dir outputs/isaac_singlebox_novae/policy_latest.ckpt \
    #--save_episode \
    #--debug \