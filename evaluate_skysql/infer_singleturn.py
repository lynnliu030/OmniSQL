import argparse
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import sys
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from functools import partial
import evaluate_spider
from tqdm import tqdm

RL_TEMPLATE = """
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
{external_knowledge}

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analisying questions, 
summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.

Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
- If you have 0 turns left, you MUST directly provide the final SQL query solution inside <solution>...</solution>. 

------------------------ START OF EXAMPLE ------------------------
Question: how many pigs are in the farm? 
    Database Schema:
    Table: animals
    - id (INTEGER, PRIMARY KEY)
    - species (TEXT)
    - age (INTEGER)
    - name (TEXT)

<think>I am querying how many pigs are in the farm. Since the question asks for how many pigs, I can use a SELECT COUNT() statement to query from the animals table where species is pig.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ END OF EXAMPLE ------------------------
"""


def parse_response(response, format="rl"):
    if format == "base":
        pattern = r"```sql\s*(.*?)\s*```"
    elif format == "rl":
        last_open = response.rindex("<solution>") if "<solution>" in response else -1
        last_close = response.rindex("</solution>") if "</solution>" in response else -1
        if last_open != -1 and last_close != -1:
            return response[last_open + len("<solution>"):last_close].strip()
        else:
            return ""
    else:
        raise ValueError(f"Unknown format: {format}")
        
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        
        # Remove surrounding ```sql ... ``` block if present inside the <solution> block
        inner_sql_match = re.match(r"```sql\s*(.*?)\s*```", last_sql, re.DOTALL)
        if inner_sql_match:
            last_sql = inner_sql_match.group(1).strip()
            
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

def get_db_paths(dataset, db_path, gold_sql_path, input_dataset=None):
    if dataset == "spider":
        db_ids = [line.split("\t")[1].strip() for line in open(gold_sql_path).readlines()]
        db_files = [os.path.join(db_path, db_id, db_id + ".sqlite") for db_id in db_ids]
    elif dataset == "synsql":
        # NOTE: no gold data from generation [separate from evals]
        db_ids = [item["db_id"] for item in input_dataset]
        db_files = [os.path.join(db_path, db_id, db_id + ".sqlite") for db_id in db_ids]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return db_files


def parse_intermediate_response(response, format="new"):
    if format == "base":
        raise ValueError("No multiturn with base format.")
    elif format == "new":
        sql_pattern = r"<sql>\s*```sql\s*(.*?)\s*```\s*</sql>"
        sol_pattern = r"<solution>\s*```sql\s*(.*?)\s*```\s*</solution>"
    elif format == "rl":
        sql_pattern = r"<sql>(.*?)</sql>"
        sol_pattern = r"<solution>(.*?)</solution>"
    else:
        raise ValueError(f"Unknown format: {format}")
        
    sql_blocks = re.findall(sql_pattern, response, re.DOTALL)
    sol_blocks = re.findall(sol_pattern, response, re.DOTALL)
    
    if sol_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sol_blocks[-1].strip()
        return True, last_sql
    elif sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        
        # Remove surrounding ```sql ... ``` block if present inside the <solution> block
        inner_sql_match = re.match(r"```sql\s*(.*?)\s*```", last_sql, re.DOTALL)
        if inner_sql_match:
            last_sql = inner_sql_match.group(1).strip()
            
        return False, last_sql
    else:
        print("No SQL or Solution blocks found.")
        return False, None 

def execute_sql(data_idx, db_file, sql):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res)
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        # print(f"[DEBUG] ERROR: {error_msg}")
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, error_msg, 0
        
def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, "Function TimeOut", 0)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        res = (data_idx, db_file, sql, error_msg, 0)
    return res
    
################################## Multi-Turn SQL Execution ##################################
def execute_sqls_parallel(db_files, pred_sqls, format="rl", num_cpus=32, timeout=40, truncate=50):
    # prepare the argument tuples
    args = [
        (idx, db_file, sql, timeout)
        for idx, (db_file, sql) in enumerate(zip(db_files, pred_sqls))
    ]
    # spawn a pool and map
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.starmap(execute_sql_wrapper, args)
    
def generate(llm, tokenizer,samples, sampling_params, batch_size=256):
    generations = []
    for i in range(0, len(samples), batch_size):
        # samples_as_text = tokenizer.apply_chat_template(samples[i:i+batch_size], tokenize=False, add_generation_prompt=True)
        samples_as_text = samples[i:i+batch_size]
        generations.extend(llm.generate(samples_as_text, sampling_params))
    return generations
    
def generate_multi_turn(chat_prompts, llm, tokenizer, db_files, sampling_params, num_turns=5, format="rl"):
    turn_prompts = list(enumerate(chat_prompts))
    outputs = [None] * len(chat_prompts)
    
    for i in range(num_turns):
        next_round_prompts = []
        turn_outputs = generate(llm, tokenizer, [t[1] for t in turn_prompts], sampling_params, 512)
        turn_responses = [o.outputs[0].text for o in turn_outputs] # TODO: handle multiple sampling
        turn_sqls = [parse_intermediate_response(response, format) for response in turn_responses]
        turn_observations = execute_sqls_parallel([db_files[i] for i in [p[0] for p in turn_prompts]], [s[1] for s in turn_sqls], format)
        turn_observations = [convert_observation_to_string(obs, i, num_turns, s[1]) for obs, s in zip(turn_observations, turn_sqls)]
        for (idx, prompt), response, (finished, _), obs in zip(turn_prompts, turn_responses, turn_sqls, turn_observations):
            if not finished:
                if outputs[idx] is None:
                    outputs[idx] = response + obs
                else:
                    outputs[idx] += response + obs
                # new_prompt = prompt + [{"role": "assistant", "content": response}] + [{"role": "user", "content": obs}]
                new_prompt = prompt + response + obs
                next_round_prompts.append((idx, new_prompt))
            else:
                if outputs[idx] is None:
                    outputs[idx] = response
                else:
                    outputs[idx] += response
            
        turn_prompts = next_round_prompts
    
    # final turn
    turn_outputs = generate(llm, tokenizer, [t[1] for t in turn_prompts], sampling_params, 512)
    turn_responses = [o.outputs[0].text for o in turn_outputs] # TODO: handle multiple sampling
    turn_sqls = [parse_intermediate_response(response, format) for response in turn_responses]
    for (idx, _), response in zip(turn_prompts, turn_responses):
        if outputs[idx] is None:
            outputs[idx] = response
        else:
            outputs[idx] += response
    
    return outputs, chat_prompts

from typing import Union, Optional

def convert_observation_to_string(observation: Union[frozenset, str], turn: int, num_turns: int, sql: Optional[str] = None):
    current_turn = turn
    reminder_text = f'<reminder>You have {num_turns-current_turn} turns left to complete the task.</reminder>'
    df = None
    if isinstance(observation, frozenset):
        df = pd.DataFrame(observation)
        res_str = df.to_string(index=False)
    else:
        res_str =  str(observation)
    
    if sql is None:
        err_msg = (
            "Your previous action is invalid. "
            "If you want to call sql tool, you should put the query between <sql> and </sql>. "
            "If you want to give the final solution, you should put the solution between <solution> and </solution>. Try again. You are encouraged to give the final solution if this is the last turn."
        )
        res_str = err_msg

    append_obs_str = f'\n\n<observation>{res_str}\n{reminder_text}</observation>\n\n'
    if len(append_obs_str) > 9000:
        print(f"[DEBUG-WARNING] OBSERVATION TOO LONG BEFORE TOKENIZATION — LEN = {len(append_obs_str)} chars, EST TOKENS ~ {len(append_obs_str)//4}")
        
        # just truncate
        truncated_df = df.head(50)
        res_str = truncated_df.to_string(index=False)  # or index=True if you want row numbers
        
        append_obs_str = f'\n\n<observation>Truncated to 50 lines since returned response too long: {res_str}\n{reminder_text}</observation>\n\n'
    
    return append_obs_str

##################### OpenAI Generation #########################

def parse_intermediate_response_openai(response, format="rl"):
    """
    Parses OpenAI-style response without guaranteed stop strings.
    Determines whether the last block was <solution> or <sql>.
    Ensures the stop tag is appended.
    """
    if format not in {"rl", "new"}:
        raise ValueError(f"Unsupported format for OpenAI: {format}")

    sql_match = list(re.finditer(r"<sql>(.*?)($|</sql>)", response, re.DOTALL))
    sol_match = list(re.finditer(r"<solution>(.*?)($|</solution>)", response, re.DOTALL))

    if not sql_match and not sol_match:
        return False, ""

    last_sql = sql_match[-1] if sql_match else None
    last_sol = sol_match[-1] if sol_match else None

    if last_sol and (not last_sql or last_sol.start() > last_sql.start()):
        sql_text = last_sol.group(1).strip()
        if not last_sol.group(2):  # missing </solution>
            response += "\n</solution>"
        return True, sql_text
    else:
        sql_text = last_sql.group(1).strip()
        if not last_sql.group(2):  # missing </sql>
            response += "\n</sql>"
        return False, sql_text


def stringify(res, truncate=50):
    if isinstance(res, frozenset):
        df = pd.DataFrame(res)
        if truncate: 
            df = df.head(truncate)
        return df.to_string(index=False)
    else:
        return str(res)

from typing import Literal


def is_reasoning_model(model_name):
    return model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")

def get_completion(client, msgs, sampling_params, model_name):
    if is_reasoning_model(model_name):
        return client.chat.completions.create(
                model=model_name,
                messages=msgs,
                n=sampling_params.n,
                # max_completion_tokens=sampling_params.max_tokens,
                reasoning_effort="high", # stop token is not supported
        )
                
    return client.chat.completions.create(
                        model=model_name,
                        messages=msgs,
                        temperature=sampling_params.temperature,
                        n=sampling_params.n,
                        max_tokens=sampling_params.max_tokens,
                        stop=sampling_params.stop
        )


def openai_generate_multi_turn(model_name, local_port, chat_prompts, db_files, sampling_params, model_type: Literal["local", "openai"],num_turns=5, format="rl", on_finish=None,):
    if model_type == "local":
        # local host model
        client = OpenAI(
            base_url=f"http://localhost:{local_port}/v1",
            api_key="token-abc123",
        )
    else:
        client = OpenAI()
        
    outputs = [None] * len(chat_prompts)
    finished_flags = [False] * len(chat_prompts)


    for turn in range(num_turns):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    lambda idx, msgs: (idx, get_completion(client, msgs, sampling_params, model_name)),
                    idx, chat_prompts[idx]
                )
                for idx in range(len(chat_prompts)) if not finished_flags[idx]
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Turn {turn+1}"):
                idx, response = future.result()
                text = response.choices[0].message.content
                finished, sql = parse_intermediate_response_openai(text, format)

                _, _, _, observation, _ = execute_sql_wrapper(idx, db_files[idx], sql, 40)
                # hack, fix later
                obs_str = convert_observation_to_string(observation, turn+1, num_turns)

                patched_text = patch_openai_stop_string(text)
                should_append_obs = (turn < num_turns-1) and not finished
                if should_append_obs:
                    full_response = patched_text + "\n" + obs_str
                else:
                    full_response = patched_text

                if outputs[idx] is None:
                    outputs[idx] = full_response
                else:
                    outputs[idx] += "\n" + full_response

                if finished:
                    finished_flags[idx] = True
                    if on_finish:
                        print(f"[INFO] Finished and saving idx={idx}")  # Debug log
                        on_finish(idx, chat_prompts[idx], outputs[idx])

                else:
                    chat_prompts[idx].append({"role": "assistant", "content": patched_text})
                    if should_append_obs:
                            chat_prompts[idx].append({"role": "user", "content": obs_str})

    for idx in range(len(finished_flags)):
        # reached max turns
        if not finished_flags[idx]:
            print(f"Reached max turns for: {idx}")
            on_finish(idx, chat_prompts[idx], outputs[idx])
    return outputs, chat_prompts

from threading import Lock
save_lock = Lock()

def save_result(idx, prompt, output, output_dir):
    pass 


def save_result_full(outputs, input_dataset, chat_prompts, output_file, n: int = 1):
    original_len = len(input_dataset) // n
    results = [None for _ in range(original_len)]
    for idx in range(len(input_dataset)):
        prompt = chat_prompts[idx]
        output = outputs[idx]
        data = input_dataset[idx].copy()
        sql = parse_response(output, opt.prompt_format)
        data = input_dataset[idx].copy()

        data["input_seq"] = prompt
        data["responses"] = [output]
        data["pred_sqls"] = [sql]

        _orig_idx= input_dataset[idx]["_unique_id"]
        data.pop("schema", None)
        data.pop("external_knowledge", None)

        if results[_orig_idx] is None:
            results[_orig_idx] = data
        else:
            results[_orig_idx]["pred_sqls"].append(sql)
            results[_orig_idx]["responses"].append(output)

        
        if isinstance(prompt, list):
            num_turns_taken = sum(1 for msg in prompt if msg["role"] == "assistant")
            data["num_turns"] = num_turns_taken
        else:
            data["num_turns"] = "N/A"

    with save_lock:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False))  # no indent for compact JSONL
                f.write("\n")
# LLM = 
def patch_openai_stop_string(response: str) -> str:
    """
    Ensures response ends with </sql> or </solution> if missing.
    """
    if response.strip().endswith("</solution>") or response.strip().endswith("</sql>"):
        return response

    last_sql = response.rfind("<sql>")
    last_sol = response.rfind("<solution>")

    if last_sol > last_sql:
        return response + "\n</solution>"
    elif last_sql > last_sol:
        return response + "\n</sql>"
    return response

import random


def kill_process(process):
    if process.poll() is None:  # If process is still running
        try:
            process.terminate()  # Try graceful termination first
            try:
                process.wait(timeout=10)  # Give it 10 seconds to terminate
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if not terminated after timeout
                process.wait()
            print(f"vllm process (PID {process.pid}) terminated")
        except Exception as e:
            print(f"Error terminating process: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type = str, default = "/fs/fast/u2021000902/previous_nvme/xxx")
    parser.add_argument("--input_file", type = str, help = "the input file path (prompts)")
    parser.add_argument("--output_dir", type = str, help = "the output file path (results)")
    parser.add_argument("--num_turns", type = int, help = "the number of turns to evaluate", default = 5)
    parser.add_argument("--tensor_parallel_size", type = int, help = "the number of used GPUs", default = 4)
    parser.add_argument("--n", type = int, help = "the number of generated responses", default = 1)
    parser.add_argument("--temperature", type = float, help = "temperature of llm's sampling", default = 1.0)
    parser.add_argument("--prompt_format", type = str, help = "format to use (base, rl, or new), defaults to base", default = "rl")
    parser.add_argument("--dataset", type=str, default="synsql", help="dataset name")
    parser.add_argument("--model_type", choices=["local", "openai"])
    parser.add_argument("--db_path", type=str, default="", help="database path")
    parser.add_argument("--gold_file", type = str, default="", help="gold sql path")
    parser.add_argument("--as_test", action="store_true", help="Run a test eval on 10 samples")
    parser.add_argument("--ts_db_path", type = str, default = "", help = "test suite database path (required by Spider)")
    parser.add_argument("--eval", action="store_true", help="Run evaluation as well")
    parser.add_argument("--infer", action="store_true", help="Run inference ")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previous run")
    parser.add_argument("--gold_result_dir", type=str, default="", help="Gold result dir for Spider 2.0")
    parser.add_argument("--eval_standard", type=str, default="", help="Gold result dir for Spider 2.0")
    parser.add_argument("--local_port", type=int, default=8000, help="local port")

    opt = parser.parse_args()
    print(opt)
    from pathlib import Path 
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)
    input_dataset = json.load(open(opt.input_file))
    if opt.as_test:
        input_dataset = input_dataset[:10]
            
    if opt.model_type == "openai":
        opt.num_turns += 1
        print(f"Adding another turn for openai models to be consistent with local vs openai. Total: {opt.num_turns}")

    max_model_len = 32768 # used to allocate KV cache memory in advance
    max_input_len = 24576  # or something like 24k
    max_output_len = 8192  # make sure sum <= max_model_len

    llm = None  # placeholder
    import time 

    model_name = opt.pretrained_model_name_or_path
    if len(opt.pretrained_model_name_or_path.split("/")) > 1:
        model_name = opt.pretrained_model_name_or_path.split("/", 1)[1]

    # Run OpenAI generate multi-turn
    db_files = get_db_paths(opt.dataset, opt.db_path, opt.gold_file, input_dataset)
    

    for i in range(len(input_dataset)):
        input_dataset[i]["_unique_id"] = i
        
    if opt.n > 1:
        input_dataset = input_dataset * opt.n
        db_files = db_files * opt.n

        
    if opt.model_type == "local":
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            # NOTE: For multi-turn generation, we set n=1 and repeat input dataset n times
            n = 1,
            stop=["</sql>", "</solution>"],
            include_stop_str_in_output=True,
            stop_token_ids=[151645] # NOTE: only for Qwen model, can adapt this 
        )
            
        llm = LLM(
            model = opt.pretrained_model_name_or_path,
            dtype = "bfloat16", 
            tensor_parallel_size = opt.tensor_parallel_size,
            max_model_len = max_model_len,
            gpu_memory_utilization = 0.92,
            swap_space = 42,
            enforce_eager = True,
            disable_custom_all_reduce = True,
            trust_remote_code = True,
        )
    else:
        # OpenAI models
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            n = 1,
            stop=["</sql>", "</solution>"],
            max_tokens = max_output_len,
        )

    print(f"OPT Prompt Format: {opt.prompt_format}")
    
    ############# Logic of Using OpenAI calls #################
    ek_key = "external_knowledge"
    system_prompt = RL_TEMPLATE
    chat_prompts = [
        [   {
                "role": "user" if is_reasoning_model(opt.pretrained_model_name_or_path) else "system",
                "content": RL_TEMPLATE+f"/n/n You are given limited {opt.num_turns} turns to address the problem."
            },
            {
                "role": "user",
                "content": "{db_details}:"+ data_entry["schema"] + ";\n\n {external_knowledge}: " + data_entry[ek_key] + ";\n\n {question}: " + data_entry["question"],
            }
        ] 
    for data_entry in input_dataset]


    if is_reasoning_model(model_name):
        print("Evaluating a REASONING model: ", model_name)
    
    # clear output file 
    if opt.infer:
        output_file = os.path.join(opt.output_dir, "results.json")
        if opt.overwrite:
            with open(output_file, "w") as f: 
                pass
        else:
            if os.path.exists(output_file):
                raise ValueError(f"File already exists: {output_file}. To overwrite, pass --overwrite")
        
        if opt.model_type == "local": 
            tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path)
            chat_prompts = tokenizer.apply_chat_template(chat_prompts, tokenize=False, add_generation_prompt=True)
            outputs, chat_prompts = generate_multi_turn(chat_prompts, llm, tokenizer, db_files, sampling_params, num_turns=opt.num_turns)
        else:
            outputs, chat_prompts = openai_generate_multi_turn(
                model_name, opt.local_port, chat_prompts, db_files, sampling_params,
                model_type=opt.model_type,
            num_turns=opt.num_turns, format=opt.prompt_format, on_finish=partial(save_result, output_dir=opt.output_dir)
        )
        save_result_full(outputs, input_dataset, chat_prompts, output_file=output_file, n=opt.n)
    # save summary json
    
    # evaluate greedy search
    if opt.eval: 
        gs_pred_file = os.path.join(opt.output_dir, "results.json")
        eval_type = "greedy_search" if opt.n == 1 else "major_voting"
        if opt.dataset == "spider": # for "spider"
            # warm up
            evaluate_spider.run_spider_eval(opt.gold_file, gs_pred_file, opt.db_path, 
                                        opt.ts_db_path, eval_type, True, as_test=opt.as_test)
            # record evaluation results
            ex_score, ts_score = evaluate_spider.run_spider_eval(opt.gold_file, gs_pred_file, opt.db_path, 
                                        opt.ts_db_path, eval_type, True, as_test=opt.as_test)
            if ts_score is None:
                gs_acc = ex_score
            else:
                gs_acc = [ex_score, ts_score]
        else:
            raise NotImplementedError("Not implemented for other datasets")

        acc_dict  = {"accuracy": gs_acc}
        eval_path = os.path.join(opt.output_dir, f"{eval_type}.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(acc_dict, indent=2, ensure_ascii=False))
    else: 
        print(f"Skipping evaluation since `eval` is not passed. Generations have been saved in {opt.output_dir}")