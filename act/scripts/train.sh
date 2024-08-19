exp_id=isaac_multicolorbox_pretrain
config_name=isaac_multicolorbox

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29512

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/isaac_multicolorbox \
    --num_nodes $nnodes \
    --load_pretrain outputs/droid_pretrain2/policy_latest.ckpt \
    #--debug \
    #--real_robot \
    #--load_dir outputs/droid_pretrain_dec/policy_latest.ckpt \
