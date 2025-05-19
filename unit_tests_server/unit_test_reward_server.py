import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from argparse import Namespace

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
from torch.optim import AdamW
import random

from openrlhf.utils import get_tokenizer

import os
from os.path import dirname, abspath
import pdb
import json

import logging

import os
import signal
from torch.multiprocessing import Process, Queue

import timeit
import types
import re
import pdb
import ast
import random
import threading
import psutil
import platform
import asyncio
import multiprocessing
from multiprocessing import Process, Queue
from contextlib import contextmanager
import subprocess
import math
import uuid
import atexit

# # Create a logger
logger = logging.getLogger('rm_server_logger')
logger.setLevel(logging.DEBUG)

# # Create file handler which logs even debug messages
# fh = logging.FileHandler('rm_server_logs.log')
# fh.setLevel(logging.INFO)

# # Add the handlers to the logger
# logger.addHandler(fh)

# logger.info("Logger initialized and ready to log.")


def check_answer(example):
    # global fn_call_correct
    # global std_call_correct
    # global fn_call_total
    # global std_call_total
    code = example['gold_standard_solution']
    code = '\n'.join(line for line in code.split('\n') if '```' not in line)

    import sys
    import io
    import contextlib

    test_cases = eval(example['verification_info'])['test_cases']
    # if test_cases[0]['type'] == 'function_call':
    #     fn_call_total += 1
    # else:
    #     std_call_total += 1
    if ('sys.exit' in code or "/dev/stdin" in code or "EOFError" in code or "quit()" in code or 
        "from sys" in code or "open(0)" in code or "map(int,input().split())" in code or 
        "SystemExit" in code or "exit(" in code or "abort(" in code or "map(int, input().split())" in code or
        "in input().split()" in code or "from io" in code):
        # print(f"sys.exit in code")
        return False

    # print(f"code\n{code}\n")

    for test_case in test_cases:

        if test_case['type'] == 'function_call':
            # print(f"input\n{test_case['input']}\n")
            # print(f"expectedoutput\n{test_case['output']}\n")
            # print('--------------------------------')

            # Execute the code to define the function
            local_dict = {}
            # Extract any import statements from the code
            imports = []
            code_lines = []
            for line in code.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    imports.append(line.strip())
                else:
                    code_lines.append(line)

            # Execute imports first in both globals and locals
            try:
                # First, execute all imports in both global and local dictionaries
                for import_stmt in imports:
                    exec(import_stmt, globals(), local_dict)
                    # Also make sure the imported modules are in globals()
                    exec(import_stmt, globals())
                
                # Then execute the rest of the code
                exec('\n'.join(code_lines), globals(), local_dict)
                
                # Get all functions defined in the code
                defined_functions = {name: obj for name, obj in local_dict.items() 
                                  if not name.startswith('__')}
                if not defined_functions:
                    print("No functions were defined in the code")
                    return False
            except Exception as e:
                print(f"Failed to execute code: {e}")
                return False

            if test_case['fn_name'] in defined_functions:
                fn_name = test_case['fn_name']
                # Make sure all necessary imports are available in both contexts
                try:
                    # Get the actual function object and call it directly
                    function_to_call = defined_functions[fn_name]
                    # Create a context with all defined functions
                    function_context = {**defined_functions}
                    # Add the function to globals temporarily to ensure it has access to other functions
                    for func_name, func_obj in defined_functions.items():
                        globals()[func_name] = func_obj
                    result = function_to_call(*test_case['input'])
                except Exception as e:
                    return False
            elif 'Solution' in defined_functions:
                soln_class = defined_functions['Solution']
                if isinstance(soln_class, type):  # Check if it's a class
                    # Create and add the solution instance
                    local_dict['soln_obj'] = soln_class()
                else:
                    # Add the existing instance
                    local_dict['soln_obj'] = soln_class
                
                fn_name = test_case['fn_name']
                try:
                    result = eval(f"soln_obj.{fn_name}(*{repr(test_case['input'])})", globals(), local_dict)
                except Exception as e:
                    return False
            else:
                print(f"Function {test_case['fn_name']} not found in code")
                return False
            
            if result == test_case['output'] or (type(test_case['output']) == list and (result == test_case['output'][0])):
                # sometimes the output is a list, sometimes it's a string
                pass
            else:
                return False
        else:
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["python", "-c", code],
                    input=test_case['input'],
                    capture_output=True,
                    timeout=10,
                    text=True  # This makes input and output work as strings
                )
                end_time = time.time()
                print(f"Non-docker time taken: {end_time - start_time} seconds")
                start_time = time.time()
                result = subprocess.run(
                    ["docker", "exec", "-i", CONTAINER_NAME, "python", "-c", code],
                    input=test_case["input"],              # feed the test-case input to STDIN
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                end_time = time.time()
                print(f"Docker time taken: {end_time - start_time} seconds")
            except Exception as e:
                return False

            for line1, line2 in zip(result.stdout.strip().split("\n"), test_case['output'].strip().split("\n")):
                try:
                    if not (line1.strip() == line2.strip() or math.isclose(float(line1.strip()), float(line2.strip()))):
                        return False
                except:
                    return False

    return True


def matches_input_format_reward(code_text):
    tests = {
        # 'do_not_call_sys_stdin': True,
        'do_not_call_solve': True,
        # 'call_input': False,
        "defined_solve": False,
    }

    for line in code_text.split('\n'):
        if 'def solve()' in line:
            tests['defined_solve'] = True
        # if 'input()' in line:
        #     tests['call_input'] = True
        if 'solve()' in line and 'def solve()' not in line:
            tests['do_not_call_solve'] = False
        # if 'sys.stdin' in line:
            # tests['do_not_call_sys_stdin'] = False

    correct = len([v for v in tests.values() if v])
    return correct / len(tests)


def safety_check(code_text):
    """
    checks if there are any http calls, modifications to the file system, or pdb usage
    best way to do this would be to use a sandboxed environment. 
    """
    return True
    unsafe_patterns = [
        "http", "requests", "urllib",  # HTTP calls
        "open(", "write(", "os.", "shutil",  # File system operations
        "pdb", "breakpoint",  # Debugging
        "logging"  # Printing
    ]
    
    code_text = code_text.lower()
    for pattern in unsafe_patterns:
        if pattern in code_text:
            return False
        if 'solve()' in code_text and 'def solve()' not in code_text:
            return False
    return True            


def extract_imports(code_str):
    """
    Extracts all import statements from the given code string and returns a dictionary
    suitable for use as the globals argument in exec.
    """
    # Parse the code into an AST
    tree = ast.parse(code_str)
    
    # Dictionary to hold the imports
    imports = {}
    
    # Iterate over all nodes in the AST
    for node in ast.walk(tree):
        # Check for import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Import the module and add it to the dictionary
                module = __import__(alias.name)
                imports[alias.asname or alias.name] = module
        elif isinstance(node, ast.ImportFrom):
            # Import specific names from a module
            module = __import__(node.module, fromlist=[alias.name for alias in node.names])
            for alias in node.names:
                imports[alias.asname or alias.name] = getattr(module, alias.name)
    
    return imports

def extract_tests(response):
    """
    Extract test input/output pairs from a response containing test data in the format:
    <test_input>(lines)</test_input><test_output>(lines)</test_output>
    
    Returns a list of dictionaries, each containing 'input' and 'output' keys.
    """
    test_pairs = []
    newline = "\n"
    
    # Find all input/output pairs using regex
    pattern = r'<test_input>(.*?)</test_input><test_output>(.*?)</test_output>'
    matches = re.findall(pattern, response, re.DOTALL)


    test_cases = []
    
    for input_text, output_text in matches:
        escaped_input = input_text.strip().replace('\n', '\\n')
        escaped_output = output_text.strip().replace('\n', '\\n')
        test_cases.append("{'type': 'stdin_stout', 'input': '" + escaped_input + "\\n', 'output': '" + escaped_output + "\\n'}")

    verification_info = "{'test_cases': [" + ', '.join(test_cases) + "]}"

    return verification_info, len(test_cases)

def extract_code(response, code_start_str, code_end_str, remove_print=False):
    try:

        start_tag = "```python"
        end_tag = "```"

        code_start = response.find(code_start_str) + len(code_start_str)
        code_end = response.rfind(code_end_str)
        code = response[code_start:code_end].strip() + "\n"

        # Replace <test_input>...</test_output> patterns with empty strings
        pattern = r'<test_input>.*?</test_input><test_output>.*?</test_output>'
        code = re.sub(pattern, '', code, flags=re.DOTALL)

        # Split the code by newlines and remove print statements
        if remove_print:
            code_lines = []
            for line in code.split('\n'):
                if not line.strip().startswith('print('):
                    code_lines.append(line)
            code = '\n'.join(code_lines)

        if safety_check(code):
            namespace = extract_imports(code)
            return code, namespace
        else:
            return "", {}
    except Exception as e:
        print(f"Error extracting code: {e}")
        return "", {}

def calculate_format_reward(response, response_begin_str):
    if response_begin_str and response_begin_str in response:
        response = response.split(response_begin_str)[1]
    tests = {
        "has_input": False,
        "has_output": False,
        "input_output_concat": False,
        "has_one_code_block": False,
        "unit_test_not_in_code_block": False,
    }
    # test 1, does it have <test_input> and <test_output> tags?
    # Use regex to extract test input and output tags
    import re
    test_input_pattern = re.compile(r'<test_input>(.*?)</test_input>', re.DOTALL)
    test_output_pattern = re.compile(r'<test_output>(.*?)</test_output>', re.DOTALL)
    code_block_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    end_code_block = response.rfind("```")
    unit_test_start = response.find("<test_input>")

    test_inputs = test_input_pattern.findall(response)
    test_outputs = test_output_pattern.findall(response)
    code_blocks = code_block_pattern.findall(response)

    if len(test_inputs) > 0:
        tests["has_input"] = True
    if len(test_outputs) > 0:
        tests["has_output"] = True
    if "</test_input><test_output>" in response:
        tests["input_output_concat"] = True
    if len(code_blocks) == 1:
        tests["has_one_code_block"] = True
    if unit_test_start > end_code_block:
        tests["unit_test_not_in_code_block"] = True
    score = sum(tests.values())
    return (score - len(tests) ) / (len(tests)) * 10


def timeit_high_precision(full_code, test_globals, max_time=10, target_precision=0.01, N_per_trial=10, warmup_steps=1000):
    """
    runs full_code and times it multiple times. If there is high variance in the timings, it will run N_per_trial times and take the median.
    """
    times = []
    for _ in range(warmup_steps):
        exec(full_code, test_globals)
    for _ in range(N_per_trial):
        start_time = time.time()
        exec(full_code, test_globals)
        end_time = time.time()
        times.append(end_time - start_time)
    print(f"Times: {np.median(times)} {times}")
    return np.median(times)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Simple context manager for timeout using signals - works in single-threaded contexts
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Set the timeout handler
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore previous signal handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

def calculate_reward_single(args, code_text, unit_test, label):
    if args.code_format == "humaneval":
        return calculate_reward_humaneval(args, code_text, unit_test, label)
    if args.code_format == "pi_verifiable":
        return calculate_reward_pi_verifiable(args, code_text, unit_test, label)

def calculate_reward_pi_verifiable(args, code_text, unit_test, label):

    score = 0.0

    fine_grained_results = calculate_reward_pi_verifiable_bool(args, code_text, unit_test, label)

    if args.return_fine_grained_results:
        return fine_grained_results
    
    if fine_grained_results["error"]:
        return -1.0

    if fine_grained_results["ref_code_with_llm_tests_passes"]:
        # unit test correct, but possibly weak

        if fine_grained_results["llm_code_with_ref_tests_passes"]:
            # code is correct
            score = 11.0
        else:
            # code is incorrect
            if fine_grained_results["llm_code_with_llm_tests_passes"]:
                # unit test weak, don't catch error
                score = 2.0
            else:
                # unit test caught bad code
                score = 10.0

    else:
        if fine_grained_results["llm_code_with_ref_tests_passes"]:
            score = 1.0

    return score

def calculate_reward_pi_verifiable_bool(args, code_text, unit_test, label):

    all_fail = {
        "ref_code_with_llm_tests_passes": False,
        "llm_code_with_llm_tests_passes": False,
        "llm_code_with_ref_tests_passes": False,
        "error": True,
    }

    try:
        with time_limit(int(args.max_time)):
            unit_test_info, num_tests = unit_test
            llm_code_with_ref_tests = {
                "gold_standard_solution": code_text,
                "verification_info": label.replace("\n", "\\n"),
            }

            verification_info = eval(label)



            llm_code_with_llm_tests = {
                "gold_standard_solution": code_text,
                "verification_info": unit_test_info,
            }

            # Run the full code and check if it errors out
            try:
                # Create a temporary globals dictionary for execution
                test_globals = {}

                if num_tests > 0:
                    llm_code_with_llm_tests_passes = check_answer(llm_code_with_llm_tests)

                    if args.use_reference_implementation:
                        reference_implementation = verification_info['reference_implementation']
                        ref_code_with_llm_tests = {
                            "gold_standard_solution": reference_implementation,
                            "verification_info": unit_test_info,
                        }
                        ref_code_with_llm_tests_passes = check_answer(ref_code_with_llm_tests)
                    else:
                        ref_code_with_llm_tests_passes = False
                else:
                    llm_code_with_llm_tests_passes = False
                    ref_code_with_llm_tests_passes = False

                llm_code_with_ref_tests_passes = check_answer(llm_code_with_ref_tests)
                
                # If we reach here, execution was successful
                
            except Exception as e:
                print(f"Code execution failed with error: {e}")
                return all_fail

            return {
                "ref_code_with_llm_tests_passes": ref_code_with_llm_tests_passes,
                "llm_code_with_llm_tests_passes": llm_code_with_llm_tests_passes,
                "llm_code_with_ref_tests_passes": llm_code_with_ref_tests_passes,
                "error": False,
            }

    except TimeoutException:
        print("Execution timed out")
        return all_fail




def calculate_reward_humaneval(args, code_text, unit_test, label):
    """
    Calculate reward for a single code sample with built-in timeout
    """
    input_format_reward = 0
    try:
        # Create a temporary module to execute the code
        if (not code_text) or (not label):
            return args.reward_parse_error

        full_code = code_text + "\n" + label
        temp_module = types.ModuleType('temp_module')
        temp_module = globals()

        reward = 0

        if args.use_input_format_reward:
            input_format_reward = (matches_input_format_reward(code_text) - 1)
            reward += input_format_reward

        
        # Execute the submitted code with timeout
        try:
            with time_limit(int(args.max_time)):
                temp_module = globals()
                
                # Set up test globals with access to temp module
                test_globals = {'temp_module': temp_module}
                
                unit_test_code = code_text + "\n" + unit_test
                try:
                    exec(unit_test_code, test_globals)
                except Exception as e:
                    print(f"Error executing unit test: {e}")
                    return 0.0
        except TimeoutException:
            print("Execution timed out")
            return 0.0
        # Execute the submitted code with timeout
        try:
            with time_limit(int(args.max_time)):
                temp_module = globals()
                
                # Set up test globals with access to temp module
                test_globals = {'temp_module': temp_module}
                
                try:
                    exec(full_code, test_globals)
                except Exception as e:
                    return -1.0
        except TimeoutException:
            print("Execution timed out")
            return -1.0
        
        
    except Exception as e:
        print(f"Input format reward: {input_format_reward}")
        print(f"Error executing code: {e}")
        return -1.0
    
    return 1.0

def calculate_reward_with_timeout(args, code_text, label, result_queue):
    """Run calculation in a separate process that can be terminated"""
    try:
        reward = calculate_reward_single(args, code_text, label)
        result_queue.put(reward)
    except Exception as e:
        print(f"Error in calculation process: {e}")
        result_queue.put(-2e-3)

# Create the FastAPI application.
app = FastAPI()

IMAGE          = "python:3.12-slim"
CONTAINER_NAME = f"pybox-{uuid.uuid4().hex[:8]}"
def _ensure_container():
    # Launch once; --detach keeps it alive, --rm cleans up after kill
    subprocess.run(
        [
            "docker", "run", "-d", "--rm",
            "--network", "none",          # still sandboxed
            "--name", CONTAINER_NAME,
            IMAGE, "sleep", "infinity"
        ],
        check=True
    )
    # Auto-remove on interpreter exit
    atexit.register(lambda: subprocess.run(["docker", "kill", CONTAINER_NAME],
                                        capture_output=True))

@app.on_event("startup")
async def startup_event():
    _ensure_container()

def calculate_thinking_length_reward(args, query, reward, code_text):
    if args.thinking_length_weight:
        return -1 * (len(query) - len(code_text)) * args.thinking_length_weight + 100000
    else:
        return 0

def handle_formatting(queries):
    if isinstance(queries[0], str):
        return queries
    else:
        return [messages[-1]['content'] for messages in queries]


@app.post("/get_reward")
async def get_reward(request: Request):
    start_time = time.time()
    data = await request.json()
    queries = data.get("query")

    prompts = data.get("prompts")
    labels = data.get("labels")
    step = int(data.get("step", 1000))


    reward_config = data.get("reward_config", {})
    use_input_format_reward = reward_config.get("use_input_format_reward", False)
    max_time = float(reward_config.get("max_time", 5))
    reward_max_single_run_time = float(reward_config.get("reward_max_single_run_time", -1 *max_time))
    reward_failure = float(reward_config.get("reward_failure", reward_max_single_run_time))
    reward_parse_error = float(reward_config.get("reward_parse_error", reward_max_single_run_time - 1))
    code_format = reward_config.get("code_format", "humaneval")
    max_length_score = float(reward_config.get("max_length_score", 100_000))
    thinking_length_weight = float(reward_config.get("thinking_length_weight", 1.0))
    use_reference_implementation = reward_config.get("use_reference_implementation", True)
    return_fine_grained_results = reward_config.get("return_fine_grained_results", False)
    split_query_on_response_begin = reward_config.get("split_query_on_response_begin", True)
    assert labels[0] is not None

    args = Namespace(step=step,
                     reward_config=reward_config,
                     target_precision=float(reward_config.get("target_precision", 0.1)),
                     warmup_steps=int(reward_config.get("warmup_steps", 1000)),
                     max_time=max_time,
                     use_input_format_reward=use_input_format_reward,
                     code_format=code_format,
                     reward_failure=reward_failure,
                     reward_max_single_run_time=reward_max_single_run_time,
                     reward_parse_error=reward_parse_error,
                     max_length_score=max_length_score,
                     thinking_length_weight=thinking_length_weight,
                     use_reference_implementation=use_reference_implementation,
                     return_fine_grained_results=return_fine_grained_results)

    response_begin_str = reward_config.get("response_begin_str", "<|im_start|>assistant")
    code_start_str = reward_config.get("code_start_str", "```python\n")
    code_end_str = reward_config.get("code_end_str", "```")
    
    rewards = []

    split_queries = []


    queries = handle_formatting(queries)
    prompts = handle_formatting(prompts)

    for query in queries:
        print(f"format reward: {calculate_format_reward(query, response_begin_str)} query: {query}")
        print(f'------------------')

        if split_query_on_response_begin and response_begin_str and response_begin_str in query:
            split_queries.append(query.split(response_begin_str)[-1])
        else:
            split_queries.append(query)
            
    queries = split_queries
    code_text_context_pairs = [extract_code(query, code_start_str, code_end_str) for query in queries]
    tests = [extract_tests(query) for query in queries]


    for query, (code_text, _), unit_test, label in zip(queries, code_text_context_pairs, tests, labels):
        # Direct execution with built-in timeout

        reward = calculate_reward_single(args, code_text, unit_test, label)
        try:
            if (type(reward) == float and reward > 5) or (type(reward) == dict and reward["ref_code_with_llm_tests_passes"]):
                print(f'reward: {reward} query: {query}')
        except Exception as e:
            pass
        if args.use_input_format_reward:
            format_reward = calculate_format_reward(query, response_begin_str)
            reward = reward + format_reward
        if args.thinking_length_weight:
            thinking_length_reward = calculate_thinking_length_reward(args, query, reward, code_text)
            if reward > max(args.reward_max_single_run_time, args.reward_parse_error, args.reward_max_single_run_time):
                reward += thinking_length_reward
        else:
            thinking_length_reward = 0
        rewards.append(reward)

    result = {"rewards": rewards}
    logger.info(f"{result}")
    end_time = time.time()
    print(f"Full request time taken: {end_time - start_time:.2f} seconds results {result} thinking_length_reward {thinking_length_reward}")
    return JSONResponse(result)


active_requests = 0
total_requests = 0

@app.middleware("http")
async def track_requests(request: Request, call_next):
    global active_requests
    global total_requests
    total_requests += 1
    active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests -= 1
    print(f"Total requests: {total_requests}")

def test_reward_model():
    # make minor change so match isn't 100%
    code_text = r"""

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--code_start", type=str, default="```python\n", help="Start of the code block")
    parser.add_argument("--code_end", type=str, default="```", help="End of the code block")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    # Pass the application as an import string (module:variable) when using multiple workers.
    uvicorn.run("unit_tests_server.unit_test_reward_server:app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level="info")
