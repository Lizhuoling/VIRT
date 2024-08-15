exp_id=debug
config_name=aloha_singleobjgrasp

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=172.17.167.51
master_port=29512

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/aloha_singleobjgrasp \
    --num_nodes $nnodes \
    --load_pretrain outputs/droid_pretrain_dec/policy_latest.ckpt \
    #--real_robot \
    #--debug \
    #--load_dir outputs/droid_pretrain_dec/policy_latest.ckpt \
