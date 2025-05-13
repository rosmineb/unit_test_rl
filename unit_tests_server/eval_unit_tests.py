import argparse
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from generate_local import (prepare_dataset, load_model_async, generate_async,
                            clear_model_async, cleanup, combine_generations)
import yaml
import asyncio
import concurrent.futures
import pdb
import requests
import json
from collections import Counter
import numpy as np
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unit tests for code generation models")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoint(s) to evaluate"
    )
    parser.add_argument(
        "--test_set",
        type=str,
        required=True,
        help="Path to the test set file or directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Start step for evaluation"
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=1000000,
        help="End step for evaluation"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=100000,
        help="Step size for evaluation"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Apply chat template to the test set"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=6,
        help="Number of servers to use for evaluation"
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="Port for the vllm server"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--reward_model_url",
        type=str,
        help="URL to the reward model"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Name of the base model"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="System prompt for the model"
    )
    parser.add_argument(
        "--grammar_path",
        type=str,
        default=None,
        help="Path to the grammar file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000000000000,
        help="Number of samples to evaluate"
    )
    
    return parser.parse_args()

def make_api_call(payload, url="http://0.0.0.0:5000/get_reward"):
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        response_json = response.json() 
        print(f"API call successful. Response: {response_json}")    
        return response_json
    else:
        print(f"API call failed with status code {response.status_code}. Response:")
        print(response.text)
        return False

def call_reward_server(sample, reward_url):

    payload = {
        "query": [sample["prompt"] + sample["response"]],
        # "query": ["<|im_start|>assistant " + sample["response"]],
        "prompts": ["asdf"],
        "labels": [sample["labels"]],
        "reward_config": {
            "max_time": 15000,
            "code_format":"pi_verifiable",
            "thinking_length_weight": 0.0,
            "use_input_format_reward": False,
            "use_reference_implementation": True,
            # "return_fine_grained_results": True
        }
    }

    url = "http://0.0.0.0:5432/get_reward"
    response = make_api_call(payload, url=url)
    print(f'response: {response}')
    print(f'response["rewards"]: {response["rewards"]}')
    print(f'response["rewards"][0]: {response["rewards"][0]}')
    return response["rewards"][0]


def compute_test_stats(dataset, reward_url):
    """
    Compute test statistics by calling the reward server for each sample in the dataset.
    
    Args:
        dataset: The dataset containing samples to evaluate
        reward_url: URL of the reward server
        
    Returns:
        Dictionary with test statistics
    """
    results = []
    
    # Define a worker function for parallel processing
    def process_sample(sample):
        return call_reward_server(sample, reward_url)
    
    # Use ThreadPoolExecutor for parallel API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(process_sample, sample) for sample in dataset]
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(dataset)} samples")
            except Exception as e:
                print(f"Error processing sample: {e}")

    stats = Counter(results)
    
    # total_correct = 0
    # pass_ref_test_only = 0
    # pass_llm_test_only = 0
    # pass_ref_and_llm_tests = 0
    # pass_none = 0
    # error_count = 0

    # for result in results:
    #     if result["error"]:
    #         error_count += 1
    #     elif result["llm_code_with_llm_tests_passes"]:
    #         if result["llm_code_with_ref_tests_passes"]:
    #             pass_ref_and_llm_tests += 1
    #         else:
    #             pass_llm_test_only += 1
    #     else:
    #         if result["llm_code_with_ref_tests_passes"]:
    #             pass_ref_test_only += 1
    #         else:
    #             pass_none += 1

    # # Compute statistics from results
    # stats = {
    #     "total_samples": len(dataset),
    #     "pass_ref_test_only": pass_ref_test_only,
    #     "pass_llm_test_only": pass_llm_test_only,
    #     "pass_ref_and_llm_tests": pass_ref_and_llm_tests,
    #     "pass_none": pass_none,
    #     "error_count": error_count
    # }
    
    return stats

async def call_vllm_server(args, model_dir, prompts):
    loads = [load_model_async(args, i, model_dir) for i in range(args.num_gpus)]
    await asyncio.gather(*loads)
    if args.grammar_path is not None:
        print(f'eval unit tests using grammar: {args.grammar_path}')
    generate_times = [generate_async(args, i, prompts, return_time=True) for i in range(args.num_gpus)]
    gathered_generate_times =await asyncio.gather(*generate_times)
    clears = [clear_model_async(args, i) for i in range(args.num_gpus)]
    await asyncio.gather(*clears)
    return gathered_generate_times

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)
    args.debug = False
    args.config = os.path.join(os.getcwd(), args.config)
    
    # Validate arguments
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory {args.checkpoint_dir} does not exist")
    
    if not os.path.exists(args.test_set):
        raise ValueError(f"Test set path {args.test_set} does not exist")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Main evaluation logic would go here
    print(f"Evaluating model from {args.checkpoint_dir} on test set {args.test_set}")

    # 1. load test set
    test_set = load_from_disk(args.test_set)

    # 2. for each checkpoint, load an adapter, merge it, save it, load in the vllm server,
    # then evaluate the model on the test set

    args.dataset = args.test_set

    all_stats = []


    for step in tqdm(range(args.start_step, args.end_step, args.step_size)):
        prompts = prepare_dataset(args)
        checkpoint_dir = os.path.join(args.checkpoint_dir, f"global_step{step}_hf")
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        # model = base_model

        model = model.merge_and_unload()

        # Create a temporary directory to save the merged model
        temp_dir = os.path.join(os.getcwd(), f"temp_model")
        # Delete temp_dir if it exists
        if os.path.exists(temp_dir):
            
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the merged model and tokenizer to the temporary directory
        print(f"Saving merged model to {temp_dir}")
        # model.base_model.model.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        tokenizer.save_pretrained(temp_dir)
        
        # Free up memory
        del model
        torch.cuda.empty_cache()

        generation_times = asyncio.run(call_vllm_server(args, temp_dir, prompts))
        output_dataset = combine_generations(args.num_gpus)

        stats = compute_test_stats(output_dataset, args.reward_model_url)
        stats["generation_time"] = np.sum(generation_times)
        all_stats.append((step, stats))

        cleanup()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f)
        
        
