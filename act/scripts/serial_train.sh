nnodes=1
nproc_per_node=1
node_rank=0
master_addr=127.0.0.1
master_port=29515

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name cnnmlp_isaac_multicolorbox \
    --save_dir ./outputs/cnnmlp_isaac_multicolorbox \
    --data_dir datasets/isaac_multicolorbox \
    --num_nodes $nnodes

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name baseline_singlecolorbox \
    --save_dir ./outputs/baseline_singlecolorbox \
    --data_dir datasets/isaac_singlecolorbox \
    --num_nodes $nnodes \

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name baseline_singlecolorbox_gaze \
    --save_dir ./outputs/baseline_singlecolorbox_gaze \
    --data_dir datasets/isaac_singlecolorbox \
    --num_nodes $nnodes

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name baseline_multicolorbox \
    --save_dir ./outputs/baseline_multicolorbox \
    --data_dir datasets/isaac_multicolorbox \
    --num_nodes $nnodes \

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name baseline_multicolorbox_gaze \
    --save_dir ./outputs/baseline_multicolorbox_gaze \
    --data_dir datasets/isaac_multicolorbox \
    --num_nodes $nnodes

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name diffusion_isaac_singlecolorbox \
    --save_dir ./outputs/diffusion_isaac_singlecolorbox \
    --data_dir datasets/isaac_singlecolorbox \
    --num_nodes $nnodes \

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name diffusion_isaac_multicolorbox \
    --save_dir ./outputs/diffusion_isaac_multicolorbox \
    --data_dir datasets/isaac_multicolorbox \
    --num_nodes $nnodes

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name act_aloha_pourblueplate \
    --save_dir ./outputs/act_aloha_pourblueplate \
    --data_dir datasets/aloha_pourblueplate \
    --num_nodes $nnodes

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name act_aloha_openlid \
    --save_dir ./outputs/act_aloha_openlid \
    --data_dir datasets/aloha_openlid \
    --num_nodes $nnodes

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --master_addr=$master_addr --master_port $master_port \
    main.py \
    --config_name act_aloha_cleantable \
    --save_dir ./outputs/act_aloha_cleantable \
    --data_dir datasets/aloha_cleantable2 \
    --num_nodes $nnodes
