exp_id=droid_pretrain2
config_name=droid_pretrain2

nnodes=1
nproc_per_node=2
node_rank=0
master_addr=172.17.167.51
master_port=29512

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/droid \
    --num_nodes $nnodes \
    #--debug \
    #--load_pretrain outputs/pretrained_weight/droid_pretrain_dec_114k.ckpt \
    #--real_robot \
    #--load_dir outputs/droid_pretrain_dec/policy_latest.ckpt \
