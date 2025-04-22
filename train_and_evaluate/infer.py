import argparse
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from data.process_to_new_format import rl_input_from_base, new_input_from_base
import os
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import sys
import pandas as pd
import multiprocessing as mp

def parse_response(response, format="new"):
    if format == "base":
        pattern = r"```sql\s*(.*?)\s*```"
    elif format == "new":
        pattern = r"<solution>\s*```sql\s*(.*?)\s*```\s*</solution>"
    elif format == "rl":
        pattern = r"<solution>\s*(.*?)\s*</solution>"
    else:
        raise ValueError(f"Unknown format: {format}")
        
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

def get_db_paths(dataset, db_path, gold_sql_path):
    if dataset == "spider":
        db_ids = [line.split("\t")[1].strip() for line in open(gold_sql_path).readlines()]
        db_files = [os.path.join(db_path, db_id, db_id + ".sqlite") for db_id in db_ids]
    elif dataset == "bird":
        gold = json.load(open(gold_sql_path))
        db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]
    elif dataset == "spider2.0":
        db_files = []
        gold = json.load(open(gold_sql_path))
        for gold_data in gold:
            db_file = os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite")
            db_files.append(db_file)
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
        return False, last_sql
    else:
        # print("No SQL blocks found.")
        return False, ""

def execute_sql(data_idx, db_file, sql):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
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
def execute_sqls_parallel(db_files, pred_sqls, format="new", num_cpus=64, timeout=40):
    # prepare the argument tuples
    args = [
        (idx, db_file, sql, timeout)
        for idx, (db_file, sql) in enumerate(zip(db_files, pred_sqls))
    ]
    # spawn a pool and map
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.starmap(execute_sql_wrapper, args)
    
    def stringify(res):
        if isinstance(res, frozenset):
            df = pd.DataFrame(res)
            df = df.head(5)
            return df.to_string(index=False)
        else:
            return str(res)

    return [f"\n<observation>\n{stringify(r[3])}\n</observation>\n" for r in results]

def generate_multi_turn(chat_prompts, db_files, sampling_params, format="new"):
    turn_prompts = list(enumerate(chat_prompts))
    outputs = [None] * len(chat_prompts)
    
    for i in range(3):
        next_round_prompts = []
        turn_outputs = llm.generate([t[1] for t in turn_prompts], sampling_params)
        turn_responses = [o.outputs[0].text for o in turn_outputs] # TODO: handle multiple sampling
        turn_sqls = [parse_intermediate_response(response, format) for response in turn_responses]
        turn_observations = execute_sqls_parallel([db_files[i] for i in [p[0] for p in turn_prompts]], [s[1] for s in turn_sqls], format)
        for (idx, prompt), response, (finished, _), obs in zip(turn_prompts, turn_responses, turn_sqls, turn_observations):
            if not finished:
                if outputs[idx] is None:
                    outputs[idx] = response + obs
                else:
                    outputs[idx] += response + obs
                next_round_prompts.append((idx, prompt + outputs[idx]))
            else:
                if outputs[idx] is None:
                    outputs[idx] = response
                else:
                    outputs[idx] += response
            
        turn_prompts = next_round_prompts
    
    # final turn
    turn_outputs = llm.generate([t[1] for t in turn_prompts], sampling_params)
    turn_responses = [o.outputs[0].text for o in turn_outputs] # TODO: handle multiple sampling
    turn_sqls = [parse_intermediate_response(response, format) for response in turn_responses]
    for (idx, _), response in zip(turn_prompts, turn_responses):
        if outputs[idx] is None:
            outputs[idx] = response
        else:
            outputs[idx] += response
    
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type = str, default = "/fs/fast/u2021000902/previous_nvme/xxx")
    parser.add_argument("--input_file", type = str, help = "the input file path (prompts)")
    parser.add_argument("--output_file", type = str, help = "the output file path (results)")
    parser.add_argument("--tensor_parallel_size", type = int, help = "the number of used GPUs", default = 4)
    parser.add_argument("--n", type = int, help = "the number of generated responses", default = 4)
    parser.add_argument("--temperature", type = float, help = "temperature of llm's sampling", default = 1.0)
    parser.add_argument("--prompt_format", type = str, help = "format to use (base, rl, or new), defaults to base", default = "base")
    parser.add_argument("--use_multiturn", action="store_true", help = "does multi-turn generation")
    parser.add_argument("--dataset", type=str, default="spider", help="dataset name")
    parser.add_argument("--db_path", type=str, default="", help="database path")
    parser.add_argument("--gold_file", type = str, default="", help="gold sql path")

    opt = parser.parse_args()
    print(opt)

    input_dataset = json.load(open(opt.input_file))
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path, trust_remote_code=True)
    
    if "Qwen2.5-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [151645] # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
    elif "deepseek-coder-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [32021]
    elif "DeepSeek-Coder-V2" in opt.pretrained_model_name_or_path:
        stop_token_ids = [100001]
    elif "OpenCoder-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [96539]
    elif "Meta-Llama-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [128009, 128001]
    elif "granite-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [0] # <|end_of_text|> is the end token of granite-3.1 and granite-code
    elif "starcoder2-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [0] # <|end_of_text|> is the end token of starcoder2
    elif "Codestral-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [2]
    elif "Mixtral-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [2]
    elif "OmniSQL-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [151645] # OmniSQL uses the same tokenizer as Qwen2.5
    else:
        print("Use Qwen2.5's stop tokens by default.")
        stop_token_ids = [151645]

    print("stop_token_ids:", stop_token_ids)
    
    max_model_len = 8192 # used to allocate KV cache memory in advance
    max_input_len = 6144
    max_output_len = 2048 # (max_input_len + max_output_len) must <= max_model_len
    
    print("max_model_len:", max_model_len)
    print("temperature:", opt.temperature)
    if opt.use_multiturn:
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            n = opt.n,
            stop_token_ids = stop_token_ids,
            stop=["</sql>", "</solution>"],
            include_stop_str_in_output=True,
        )
    else:
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            n = opt.n,
            stop_token_ids = stop_token_ids
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
        trust_remote_code = True
    )
    
    if opt.prompt_format == "base":
        chat_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": data["input_seq"]}],
            add_generation_prompt = True, tokenize = False
        ) for data in input_dataset]
    elif opt.prompt_format == "rl":
        chat_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": rl_input_from_base(data["input_seq"])}],
            add_generation_prompt = True, tokenize = False
        ) for data in input_dataset]
    elif opt.prompt_format == "new":
        if opt.use_multiturn:
            chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": new_input_from_base(data["input_seq"], multiturn = True)}],
                add_generation_prompt = True, tokenize = False
            ) for data in input_dataset]
        else:
            chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": new_input_from_base(data["input_seq"], multiturn = False)}],
                add_generation_prompt = True, tokenize = False
            ) for data in input_dataset]
    else:
        raise ValueError(f"Unknown prompt format: {opt.prompt_format}")

    if opt.use_multiturn:
        db_files = get_db_paths(opt.dataset, opt.db_path, opt.gold_file)
        outputs = generate_multi_turn(chat_prompts, db_files, sampling_params, opt.prompt_format)
        
        results = []
        for data, output in zip(input_dataset, outputs):
            sql  = parse_response(output, opt.prompt_format)
            
            data["responses"] = [output]
            data["pred_sqls"] = [sql]
            results.append(data)
    else:
        outputs = llm.generate(chat_prompts, sampling_params)
    
        results = []
        for data, output in zip(input_dataset, outputs):
            responses = [o.text for o in output.outputs]
            sqls  = [parse_response(response, opt.prompt_format) for response in responses]
            
            data["responses"] = responses
            data["pred_sqls"] = sqls
            results.append(data)

    with open(opt.output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(results, indent = 2, ensure_ascii = False))
