exp_name=u3_sft_7B

DATA_DIR=/home/rosmine/data2/rl/${exp_name}

deepspeed --module openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset /home/rosmine/data2/rl_codegen/datasets/pi_verifiable_no_fn_call_with_unit_tests_v2_sft_formatted \
    --input_key prompt \
    --output_key response \
    --train_batch_size 48 \
    --apply_chat_template \
    --micro_train_batch_size 2 \
    --max_samples 500000 \
    --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
    --save_path /home/rosmine/data2/rl/${exp_name}/checkpoint \
    --save_steps 5 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --learning_rate 3e-6 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --use_tensorboard ${DATA_DIR}/logs/${exp_name} \

