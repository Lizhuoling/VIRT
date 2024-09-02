exp_id=aloha_pourblueplate
config_name=aloha_pourblueplate

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29514

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/aloha_pourblueplate \
    --num_nodes $nnodes \
    --load_pretrain outputs/pretrained_weight/droid_pretrain_dec_114k.ckpt \
    #--debug \
    #--real_robot \
    #--load_dir outputs/droid_pretrain_dec/policy_latest.ckpt \
