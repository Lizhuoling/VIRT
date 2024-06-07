exp_id=sim_transfer_cube_scripted
task_name=sim_transfer_cube_scripted 

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python3 imitate_episodes.py \
    --task_name $task_name \
    --ckpt_dir ./outputs/$exp_id \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0 \
    --eval \
    #--debug \