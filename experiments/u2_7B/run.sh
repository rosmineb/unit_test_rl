exp_name=u2_7B

ray start --head --port=6380 --node-ip-address=127.0.0.1 --temp-dir=/home/rosmine/data2/rl/${exp_name}

DATA_DIR=/home/rosmine/data2/rl/${exp_name}

echo "ray start"

OPENRLHF_DIR=/home/rosmine/projects/OpenRLHF

ray job submit --address="http://127.0.0.1:8265" \
          --runtime-env-json='{"working_dir": "'${OPENRLHF_DIR}'"}' \
          -- python3 -m openrlhf.cli.train_ppo_ray \
          --ref_num_nodes 1 \
          --ref_num_gpus_per_node 4 \
          --actor_num_nodes 1 \
          --actor_num_gpus_per_node 4 \
          --vllm_num_engines 1 \
          --vllm_tensor_parallel_size 2 \
          --vllm_gpu_memory_utilization 0.95 \
          --save_path ${DATA_DIR}/save_path \
          --ckpt_path ${DATA_DIR}/checkpoint \
          --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
          --save_steps 1 \
          --logging_steps 1 \
          --eval_steps -1 \
          --micro_train_batch_size 2 \
          --train_batch_size 16 \
          --micro_rollout_batch_size 4 \
          --rollout_batch_size 32 \
          --n_samples_per_prompt 16 \
          --rm_batch_size 1 \
          --max_epochs 2 \
          --prompt_max_len 2048 \
          --generate_max_len 2048 \
          --zero_stage 2 \
          --bf16 \
          --actor_learning_rate 3e-6 \
          --gradient_checkpointing \
          --init_kl_coef 0.0 \
          --apply_chat_template \
          --prompt_data /home/rosmine/data2/rl_codegen/datasets/pi_verifiable_no_fn_call_with_unit_tests_v2_train \
          --input_key prompt \
          --label_key verification_info \
          --max_samples 100000 \
          --normalize_reward \
          --load_checkpoint \
          --advantage_estimator group_norm \
          --remote_rm_url http://localhost:5432/get_reward \
          --use_tensorboard logs/${exp_name} \
          --vllm_sync_backend nccl \
          --enforce_eager \
          --save_hf_ckpt \
          --disable_ds_ckpt \
          --RM_CONFIG_save_threshold 60 \
          --RM_CONFIG_n_steps 100 \
          --RM_CONFIG_save_file /home/rosmine/data2/ray/${exp_name}/optimizer_code_output \
          --RM_CONFIG_aux_decay 1 \
          --RM_CONFIG_n_trials 10 \
          --RM_CONFIG_dfg_complexity_weight 0.5 \
          --RM_CONFIG_code_bleu_weight 0.5 \
          --RM_CONFIG_timeout_seconds 120 \
          --RM_CONFIG_aux_coef 0.0 \
          --RM_CONFIG_aux_coef_warmup 0 \
          --RM_CONFIG_useful_line_ratio_coef 0.0 \
          --RM_CONFIG_batch_size 128 \
          --RM_CONFIG_use_input_format_reward true \
          --RM_CONFIG_code_format "pi_verifiable" \
          --RM_CONFIG_max_time 10.0 \
          --RM_CONFIG_thinking_length_weight 0 \
          --entropy_coef 0.0 \
          --num_episodes 1 \
          --outlier_reward_filter -10.0 \
          --ring_attn_size 4 \
          --lora_alpha 128 \
          --lora_rank 64 \
          --packing_samples \
          --flash_attn \
          --adam_offload \
          --lr_warmup_ratio 0.001 \
          #--colocate_all_models \
          #--vllm_enable_sleep \
          #--deepspeed_enable_sleep \
          #--deepspeed_enable_super_sleep

ray stop
         #  --lora_alpha 128 \
         # --lora_rank 64
         #--pretrain $(pwd)/checkpoint/7B_sft_v2 \
          #--flash_attn \
          #--pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
          #--vllm_sync_with_ray \
          #--pretrain Qwen/Qwen2.5-Coder-0.5B-Instruct \
