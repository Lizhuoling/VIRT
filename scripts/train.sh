exp_id=sim_transfer_cube_scripted2
config_name=sim_transfer_cube_scripted 

nnodes=1
nproc_per_node=1
node_rank=0
master_addr=172.17.167.131
master_port=29514

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    imitate_episodes.py \
    --config_name $config_name \
    --ckpt_dir ./outputs/$exp_id \
    --data_dir ./datasets/sim_transfer_cube_scripted \
    --num_nodes $nnodes \
    #--debug \