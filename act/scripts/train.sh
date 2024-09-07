exp_id=aloha_pourblueberry
config_name=aloha_pourblueberry

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29514

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/aloha_pourblueberry \
    --num_nodes $nnodes \
    --load_pretrain outputs/pretrained_weight/droid_pretrain_dec_114k.ckpt \
    #--debug \
    #--real_robot \
    #--load_dir outputs/droid_pretrain_dec/policy_latest.ckpt \
