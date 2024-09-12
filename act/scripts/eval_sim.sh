exp_id=isaac_multicolorbox
config_name=isaac_multicolorbox

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29514

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir outputs/$exp_id \
    --load_dir outputs/$exp_id/policy_latest.ckpt \
    --num_nodes $nnodes \
    --eval \
    #--real_robot \
    #--save_episode \
    #--debug \