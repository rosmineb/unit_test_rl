import requests
import json
import pdb
from datasets import load_from_disk
from transformers import AutoTokenizer
import concurrent.futures
import pdb

def hf_reward_fn(prompts, completions, **kwargs):
    # print(prompts, completions, kwargs)
    verification_info = kwargs["verification_info"]
    rewards = []
    # Create payloads for all completions
    payloads = []
    for i in range(len(completions)):
        payload = {
            "query": [completions[i]],
            "prompts": [prompts[i]],
            "labels": [verification_info[i]],
            "step": 2,
            "extra_info": kwargs,
            "reward_config": {"n_steps": 200,
                            "target_precision": 0.01,
                            "warmup_steps": 10000,
                            "max_time": 5,
                            "code_format":"pi_verifiable",
                            "thinking_length_weight": 0.0,
                            "use_input_format_reward": True}
        }
        payloads.append(payload)
    
    url = "http://0.0.0.0:5431/get_reward"
    
    # Process requests in parallel
    def process_request(payload):
        result = make_api_call(payload, url=url)
        return result['rewards'][0]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_request, payloads))
    
    rewards.extend(results)
    print(f"returning rewards: {rewards}")
    return rewards

def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    data_source: str
    solution_str: str
    ground_truth: str
    extra_info: dict
    """
    payload = {
        "query": [solution_str],
        "prompts": [data_source],
        "labels": [ground_truth],
        "step": 2,
        "extra_info": extra_info,
        "reward_config": {"n_steps": 200,
                          "target_precision": 0.01,
                          "warmup_steps": 10000,
                          "max_time": 5,
                          "code_format":"pi_verifiable",
                          "thinking_length_weight": 0.0,
                          "use_input_format_reward": True}
    }

    url = "http://0.0.0.0:5431/get_reward"
    result = make_api_call(payload, url=url)
    return result
    

def make_api_call(payload, url="http://0.0.0.0:5000/get_reward"):
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        response_json = response.json() 
        # print(f"API call successful. Response: {response_json}")    
        return response_json
    else:
        print(f"API call failed with status code {response.status_code}. Response:")
        print(response.text)
        return False

if __name__ == "__main__":

    n_queries = 1
    n_calls = 2
    n_calls_parallel = 8

    correct_code = r"""
a = int(input())
b = int(input())
print(a + b)
"""

    incorrect_code = r"""
a = int(input())
b = int(input())
print(a * b)
"""

    unit_test = r"""1
2
"""

    unit_test_correct_output = r"""3
"""

    unit_test_incorrect_output = r"""2
"""
    
    pairs = [
        # (correct_code, unit_test, unit_test_correct_output), # 1
        # (incorrect_code, unit_test, unit_test_correct_output),  # 0
        (incorrect_code, unit_test, unit_test_incorrect_output), # -1
    ]

    labels = "{'test_cases': [{'type': 'stdin_stdout', 'input': '1\n2', 'output': '3'}]}"

    for code, unit_test, unit_test_output in pairs:
        unit_tests = f"<test_input>{unit_test}</test_input><test_output>{unit_test_output}</test_output>"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "solve plz"},
            {"role": "assistant", "content": f"<code>{code}</code>\n{unit_tests}"}
        ]

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
        inputs = tokenizer.apply_chat_template(messages, tokenize=False)

        payload = {
            "query": [inputs],
            "prompts": ["asdf"],
            "labels": [labels],
            "step": 2,
            "reward_config": {
                "n_steps": 200,
                "target_precision": 0.01,
                "warmup_steps": 10000,
                "max_time": 100000,
                "code_format":"pi_verifiable",
                "thinking_length_weight": 0.0
            }
        }

        url = "http://0.0.0.0:5432/get_reward"
        make_api_call(payload, url=url)

