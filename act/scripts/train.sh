exp_id=act_isaac_singlecolorbox
config_name=act_isaac_singlecolorbox

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29514

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/isaac_singlecolorbox \
    --num_nodes $nnodes \
    #--load_pretrain outputs/pretrained_weight/droid_pretrain_dec_114k.ckpt \
    #--load_dir outputs/aloha_cleantable2/policy_latest.ckpt \
    #--debug \
    #--real_robot \
    #--load_dir outputs/aloha_cleantable/policy_latest.ckpt \
