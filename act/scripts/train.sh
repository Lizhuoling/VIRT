exp_id=diffusion_isaac_singlebox
config_name=diffusion_isaac_singlebox

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29515

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name $config_name \
    --save_dir ./outputs/$exp_id \
    --data_dir datasets/isaac_singlebox \
    --num_nodes $nnodes \
    #--debug \
    #--load_pretrain outputs/pretrained_weight/droid_pretrain_dec_114k.ckpt \
    #--load_dir outputs/aloha_cleantable2/policy_latest.ckpt \
    #--real_robot \
    #--load_dir outputs/aloha_cleantable/policy_latest.ckpt \
