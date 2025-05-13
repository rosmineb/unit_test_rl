from unsloth import FastLanguageModel
import torch
import yaml
from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer

import argparse

import os
import importlib.util
import sys
import pdb

from unsloth_train_patch import train

def load_and_call_function(file_path, function_name, *args, **kwargs):
    """
    Loads a Python file from the given path and calls the specified function.
    
    Args:
        file_path (str): Path to the Python file
        function_name (str): Name of the function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    # Get the absolute path
    abs_path = os.path.abspath(file_path)
    
    # Check if file exists
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    
    # Get the module name from the file path
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    
    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {abs_path}")
    
    # Create the module
    module = importlib.util.module_from_spec(spec)
    
    # Add the module to sys.modules
    sys.modules[module_name] = module
    
    # Execute the module
    spec.loader.exec_module(module)
    
    # Check if the function exists in the module
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {abs_path}")
    
    # Get the function
    function = getattr(module, function_name)
    
    # Call the function with the provided arguments
    return function

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Unsloth model")
    parser.add_argument("--config", '-c', type=str, help="Path to configuration file")
    return parser.parse_args()

def main(args: argparse.Namespace):


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,  # False for LoRA 16bit
        load_in_8bit=args.load_in_8bit,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.6,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    dataset = load_from_disk(args.dataset_path)

    training_args = GRPOConfig(
        learning_rate=args.learning_rate,
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Increase to 4 for smoother training
        num_generations=args.num_generations,  # Decrease if out of memory
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_seq_length - args.max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",  # Can use Weights & Biases
        output_dir=args.output_dir,
        use_vllm=True,
        beta=0.001,
        # generation_batch_size=args.generation_batch_size,
        # steps_per_generation=args.steps_per_generation,
    )


    # training_args.steps_per_generation = None

    reward_function = load_and_call_function(args.reward_function_path, args.reward_function_name)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_function,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # train(trainer)
    trainer.train()

    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)
    main(args)