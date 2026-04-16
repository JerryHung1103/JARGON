import os
import json
import sys
import datetime 
import pandas as pd
from multiprocessing import Pool, Lock
from tqdm import tqdm
import argparse
import openai
from dotenv import load_dotenv
from huggingface_hub import login
import traceback


from utils import generate, read_txt_to_string 
from jailbreak_engine import Jailbreak 
from config.config_loader import ConfigLoader 
from LLM_judge.judge import LLMJudger 
from agents.evaluator import Evaluator 
from benchmark.benchmark_interface import JailbreakBenchAdapter, MedSafetyBenchAdapter

file_lock = Lock()


load_dotenv()
hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
login(token=hf_token)

config_loader = ConfigLoader()
target_model_name = config_loader.target_model['model']
paper_name = config_loader.paper_name
context_json_path = config_loader.context_json_path


try:
    with open(context_json_path, 'r') as f:
        context = json.load(f)
except FileNotFoundError:
    print(f"File {context_json_path} not found. Generate context summary first!")

if paper_name not in context:
    raise ValueError(
        f"Paper '{paper_name}' not found in context. "
        f"Either it's not yet cached or the name is incorrect."
    )

judge_name = LLMJudger().get_name()
jailbreak_config = config_loader.jailbreak_setting
enable_safeguard = jailbreak_config['enable_safeguard']
num_workers = jailbreak_config['num_workers']
mode = jailbreak_config['mode']
if mode not in ['Distillation', 'Attack']:
    raise ValueError(f"Invalid mode '{mode}'. Must be either 'Distillation' or 'Attack'.")

early_stop = jailbreak_config['early_stop']
if mode == 'Distillation' and early_stop:
    raise ValueError("Distillation mode cannot be used with early stopping enabled.")

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out_dir = f'./{config_loader.out_dir}/{paper_name.replace("/","_")}_as_ctx'

os.makedirs(out_dir, exist_ok=True)
log_dir = f'{out_dir}/{config_loader.logs_dir}'


result_dir = out_dir+'/'+config_loader.result_dir
os.makedirs(result_dir, exist_ok=True)
output_file_path = f'{result_dir}/{target_model_name.replace('/', '_')}_{current_time}_result.json'

successfulExampleCacheDir = out_dir + '/' + config_loader.successful_example_cache

harmful_trajectory_cache_file = None
safe_trajectory_cache_file  = None
if jailbreak_config['record_trajectories']:
    os.makedirs(f"{out_dir}/harmful_trajectories", exist_ok = True)
    os.makedirs(f"{out_dir}/safe_trajectories", exist_ok = True)
    harmful_trajectory_cache_file = f"{out_dir}/harmful_trajectories/{target_model_name.replace('/','_')}_{current_time}.json"
    safe_trajectory_cache_file = f"{out_dir}/safe_trajectories/{target_model_name.replace('/','_')}_{current_time}.json"


os.makedirs(successfulExampleCacheDir, exist_ok=True)
successfulExampleCachePath = f"{successfulExampleCacheDir}/{target_model_name.replace('/','_')}_{current_time}.json"


# Reload config and objects
config_loader = ConfigLoader()
judge = LLMJudger() if mode == 'Attack' else None

from agents.safeguard import get_safeguard_response, format_target_safeguard_instruction, create_dynamic_sys_prompt

import copy
if enable_safeguard:
    def target_generate(messages):
        internal_messages = copy.deepcopy(messages)
        previous_queries = [conv['content'] for conv in internal_messages if conv['role'] == 'user']
        current_query = previous_queries.pop()
        dynamic_system_prompt = create_dynamic_sys_prompt(current_query, previous_queries)
        print('-'*50)
        print('[Safeguard] Dynamic System Prompt: ', dynamic_system_prompt)
        print('-'*50)
        if internal_messages[0]['role'] == 'system':
            internal_messages[0]['content'] += '\n'+dynamic_system_prompt
        else:
            internal_messages = [{'role':'system','content':dynamic_system_prompt}]  + internal_messages
        safeguard_res = get_safeguard_response(current_query, previous_queries)
        print('-'*50)
        print('[Safeguard] Dynamic Prompt Injection: ', safeguard_res)
        print('-'*50)
        internal_messages[-1]['content'] = format_target_safeguard_instruction(safeguard_res, current_query)
        return generate(internal_messages, client=openai.OpenAI(base_url=config_loader.target_model['base_url'],api_key=config_loader.target_model['api_key']), model=target_model_name, **config_loader.target_model['generate_kwargs'])
else:
    target_generate = lambda messages: generate(
        messages, 
        client=openai.OpenAI(
            base_url=config_loader.target_model['base_url'],
            api_key=config_loader.target_model['api_key']
        ),
        model=target_model_name, 
        **config_loader.target_model['generate_kwargs']
    )

refusal_checker = config_loader.refusal_checker['model']
check_refusal_generate = lambda messages: generate(
    messages, 
    client=openai.OpenAI(
        base_url=config_loader.refusal_checker['base_url'],
        api_key=config_loader.refusal_checker['api_key']
    ),
    model=refusal_checker, 
    **config_loader.refusal_checker['generate_kwargs']
)

summary_model=config_loader.summary_model['model']
summerizer_generate = lambda messages: generate(
    messages, 
    client=openai.OpenAI(
        base_url=config_loader.summary_model['base_url'],
        api_key=config_loader.summary_model['api_key']
    ),
    model=summary_model, 
    **config_loader.summary_model['generate_kwargs']
)

optimizer=config_loader.optimizer['model']
optimizer_genrate = lambda messages: generate(
    messages, 
    client=openai.OpenAI(
        base_url=config_loader.optimizer['base_url'],
        api_key=config_loader.optimizer['api_key']
    ),
    model=optimizer, 
    **config_loader.optimizer['generate_kwargs']
)

attacker = config_loader.attacker['model']
attacker_generate = lambda messages: generate(
    messages, 
    client=openai.OpenAI(
        base_url=config_loader.attacker['base_url'],
        api_key=config_loader.attacker['api_key']
    ),
    model=attacker, 
    **config_loader.attacker['generate_kwargs']
)

knowledge_extractor = config_loader.knowledge_extractor['model']
knowledge_extractor_generate = lambda messages: generate(
    messages, 
    client=openai.OpenAI(
        base_url=config_loader.knowledge_extractor['base_url'],
        api_key=config_loader.knowledge_extractor['api_key']
    ),
    model=knowledge_extractor, 
    **config_loader.knowledge_extractor['generate_kwargs']
)

# Same as the attacker model, adjust if needed
evaluator = Evaluator(
    base_url=config_loader.attacker['base_url'],
    api_key=config_loader.attacker['api_key'],
    model_name = config_loader.attacker['model']
)

# Initialize Jailbreaker
jailbreaker = Jailbreak(
    output_file_path = output_file_path,
    target_generate = target_generate,
    attacker_generate = attacker_generate,
    check_refusal_generate = check_refusal_generate,
    summerizer_generate = summerizer_generate,
    optimizer_genrate = optimizer_genrate,
    knowledge_extractor_generate = knowledge_extractor_generate,
    context= context[paper_name],
    config = jailbreak_config,
    evaluator = evaluator,
    judge=judge,
    file_lock=file_lock,
    early_stop = early_stop,
    harmful_trajectory_cache_file = harmful_trajectory_cache_file,
    safe_trajectory_cache_file  = safe_trajectory_cache_file
)

max_retries = jailbreak_config['max_retries']


def get_ordinal_suffix(number):
    if 10 <= number % 100 <= 20:
        return 'th'
    else:
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')



def init_worker(lock):
    tqdm.set_lock(lock)

def worker_setup_and_run(chunk_data):
    """
    Worker process function with fixed progress bar layout and file logging.
    """
    pid = os.getpid()
    samples_subset, start_index_offset, worker_idx = chunk_data
    
    total_steps = len(samples_subset) * max_retries
    
    worker_pbar = tqdm(
        total=total_steps,
        desc=f"[Worker {worker_idx}] Init",
        position=worker_idx, 
        leave=True,
        unit="task",
        file=sys.__stdout__, 
        dynamic_ncols=True   
    )

    try:
        for numberOfretry in range(1, max_retries + 1):
            
            for i, test_case in enumerate(samples_subset):
                actual_index = start_index_offset + i
                topic = test_case['query']

                
                worker_pbar.set_description(f"[Worker {worker_idx}] Retry {numberOfretry}")
                worker_pbar.set_postfix({"Idx": actual_index, "Status": "Processing"})

  
                skip_processing = False
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if topic in data and ((data[topic].get('is_success') == True and mode=='Attack') or (data[topic].get('distilled_content') is not None and mode=='Distillation')):
                            skip_processing = True
                except Exception:
                    pass
                
                if skip_processing:
                    worker_pbar.set_postfix({"Idx": actual_index, "Status": "Skipped"})
                    worker_pbar.update(1)
                    continue


                os.makedirs(f'{log_dir}/{target_model_name.replace("/","_")}/{current_time}', exist_ok=True)
                log_file = f'{log_dir}/{target_model_name.replace("/","_")}/{current_time}/retry_{numberOfretry}_topic_{actual_index}.txt'
               
                original_stdout = sys.stdout 
                f_log = None

                try:
      
                    f_log = open(log_file, 'w', buffering=1)
                    sys.stdout = f_log

                    print("="*60)
                    print(f"ATTACK START | WORKER {worker_idx} | PID {pid} | TOPIC INDEX {actual_index}")
                    print(f"Goal: {topic}")
                    print(f"Retry: {numberOfretry}")
                    print("="*60)

          
                    jailbreaker.run_Jailbreak(
                        test_case = test_case, 
                        index = actual_index,
                        successfulExampleCachePath=successfulExampleCachePath, 
                        currentNumberOfretry=numberOfretry
                    )
                    
     
                    worker_pbar.set_postfix({"Idx": actual_index, "Status": "Done"})

                except Exception as e:
          
                    sys.stdout = original_stdout
                    worker_pbar.set_postfix({"Idx": actual_index, "Status": "Error"})
                    continue
                finally:
         
                    if f_log and not f_log.closed:
                        f_log.close()
                    sys.stdout = original_stdout

        
                worker_pbar.update(1)
    
    except Exception as e:
 
        tqdm.write(f"Critical Worker Error: {e}")
    finally:
        worker_pbar.close()

def main():
    """Main function to setup chunks and the multiprocessing pool."""
    
    tqdm_lock = Lock()
  
    jailbreakBench_df = pd.read_csv(config_loader.dataset_path)
    jailbreak_adapter = JailbreakBenchAdapter(jailbreakBench_df)
    # jailbreak_samples = jailbreak_adapter.get_all_data()
    jailbreak_samples = jailbreak_adapter.get_top_n_by_category(n_samples=2)
   
    

    chunk_size = (len(jailbreak_samples) + num_workers - 1) // num_workers 

    chunks = []
    print(f"Attacking {target_model_name}")
    print(f"Total samples: {len(jailbreak_samples)}. Using {num_workers} workers.")
    print("-" * 60)

    worker_idx = 0
    for i in range(0, len(jailbreak_samples), chunk_size):
        subset = jailbreak_samples[i : i + chunk_size]
        chunks.append((subset, i, worker_idx)) 
        worker_idx += 1


    print("\n" * 1) 


    with Pool(processes=num_workers, initializer=init_worker, initargs=(tqdm_lock,)) as pool:
        pool.map(worker_setup_and_run, chunks)

    print("\nMultiprocessing finished.")

if __name__ == '__main__':
    main()