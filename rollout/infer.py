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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

THINKING_SYSTEM = """
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, use SQL tools to verify your partial solution or get information about the database, and generate the solution.

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
- Even if the database is empty or missing relevant entries, you must still generate a valid SQL query that would correctly answer the question if data were present.
- Do not stop early just because you observe no results from a query. Instead, continue reasoning using the schema.
- Only output the information that is asked in the question. If the question asks for a specific column, include only that column in the SELECT clause.
- Think through the steps of how to write the query, use the SQL tool to verify partial steps or schema understanding. You can explore, refine, and iterate logically based on observations.

Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
- You can use SQL tool written within a single <sql>your sql</sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>. Based on this observation, you can think again and refine.
- The returned dataframe will be truncated in 100 rows if observation is too long. 
- If you find no further exploration is needed or reaches max turns, you MUST directly provide the final SQL query solution inside <solution>...</solution>. 

Example: 
<think>I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.</think>
<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>
<observation>
+----------+
| COUNT(*) |
+----------+
|   12     |
+----------+
</observation>
<think>The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
"""


NO_SCHEMA_THINKING_SYSTEM = """
Task Overview:
You are a data science expert. Below, you are provided with a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question. You should breakdown the problem, draft your reasoning process, use SQL tools to verify your partial solution or get information about the database, and generate the solution.

Database Engine:
SQLite

External Knowledge:
{external_knowledge}

Question:
{question}

Instructions:
- Only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more. Be sure to return all of the information asked in the question without any missing or extra information.
- Think through the steps of how to write the query, use the SQL tool to verify as needed. It can include analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.

Format:
- Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
- After reasoning, if you find you lack some knowledge or confidence, you can use SQL tool written within a single <sql> your test sql query </sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>. Do not put ```sql``` inside <sql></sql>, just put raw sql.
- If the SQL tool output is too long, the returned dataframe will be truncated in 100 rows. If possible, try to use LIMIT clause to keep verification SQL tool / queries lightweight. 
- Based on this observation, you can think again and refine your query, and potentially test again.
- If you find no further external information is needed, directly provide the final SQL query solution inside <solution>...</solution>. You are given limited turns to think, interact, and answer the question.
"""

def parse_response(response, format="rl"):
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

def get_db_paths(dataset, db_path, gold_sql_path, input_dataset=None):
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
    elif dataset == "synsql":
        # NOTE: no gold data from generation [separate from evals]
        db_ids = [item["db_id"] for item in input_dataset]
        db_files = [os.path.join(db_path, db_id, db_id + ".sqlite") for db_id in db_ids]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return db_files


def parse_intermediate_response(response, format="rl"):
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
        print(f"[DEBUG] ERROR: {error_msg}")
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
def execute_sqls_parallel(db_files, pred_sqls, format="rl", num_cpus=64, timeout=40, truncate=50):
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
            df = df.head(truncate)
            return df.to_string(index=False)
        else:
            return str(res)

    return [f"\n<observation>\n{stringify(r[3])}\n</observation>\n" for r in results]

    
def generate_multi_turn(chat_prompts, db_files, sampling_params, num_turns=5, format="rl"):
    turn_prompts = list(enumerate(chat_prompts))
    outputs = [None] * len(chat_prompts)
    
    for i in range(num_turns):
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

##################### OpenAI Generation #########################

def parse_intermediate_response_openai(response, format="rl"):
    """
    Parses OpenAI-style response without guaranteed stop strings.
    Determines whether the last block was <solution> or <sql>.
    Ensures the stop tag is appended.
    """
    if format not in {"rl", "new"}:
        raise ValueError(f"Unsupported format for OpenAI: {format}")

    sql_match = list(re.finditer(r"<sql>(.*?)(</sql>)?", response, re.DOTALL))
    sol_match = list(re.finditer(r"<solution>(.*?)(</solution>)?", response, re.DOTALL))

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
        df = df.head(truncate)
        return df.to_string(index=False)
    else:
        return str(res)
    
def openai_generate_multi_turn(model_name, chat_prompts, db_files, sampling_params, num_turns=5, format="rl"):
    client = OpenAI()
    outputs = [None] * len(chat_prompts)
    finished_flags = [False] * len(chat_prompts)

    for turn in range(num_turns):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    lambda idx, msgs: (idx, client.chat.completions.create(
                        model=model_name,
                        messages=msgs,
                        temperature=sampling_params.temperature,
                        n=sampling_params.n,
                        max_tokens=sampling_params.max_tokens,
                        stop=sampling_params.stop
                    )),
                    idx, chat_prompts[idx]
                )
                for idx in range(len(chat_prompts)) if not finished_flags[idx]
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Turn {turn+1}"):
                idx, response = future.result()
                text = response.choices[0].message.content
                finished, sql = parse_intermediate_response_openai(text, format)

                if finished:
                    finished_flags[idx] = True

                _, _, _, observation, _ = execute_sql(idx, db_files[idx], sql)
                obs_str = f"<observation>\n{stringify(observation)}\n</observation>"

                patched_text = patch_openai_stop_string(text)
                full_response = patched_text if finished else patched_text + "\n" + obs_str

                if outputs[idx] is None:
                    outputs[idx] = full_response
                else:
                    outputs[idx] += "\n" + full_response

                if not finished:
                    chat_prompts[idx].append({"role": "assistant", "content": patched_text})
                    chat_prompts[idx].append({"role": "user", "content": obs_str})
                else:
                    chat_prompts[idx].append({"role": "assistant", "content": patched_text})

    return outputs

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
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type = str, default = "/fs/fast/u2021000902/previous_nvme/xxx")
    parser.add_argument("--input_file", type = str, help = "the input file path (prompts)")
    parser.add_argument("--output_file", type = str, help = "the output file path (results)")
    parser.add_argument("--num_turns", type = int, help = "the number of turns to evaluate", default = 5)
    parser.add_argument("--tensor_parallel_size", type = int, help = "the number of used GPUs", default = 4)
    parser.add_argument("--n", type = int, help = "the number of generated responses", default = 4)
    parser.add_argument("--temperature", type = float, help = "temperature of llm's sampling", default = 1.0)
    parser.add_argument("--prompt_format", type = str, help = "format to use (base, rl, or new), defaults to base", default = "rl")
    parser.add_argument("--use_multiturn", action="store_true", help = "does multi-turn generation")
    parser.add_argument("--dataset", type=str, default="synsql", help="dataset name")
    parser.add_argument("--db_path", type=str, default="", help="database path")
    parser.add_argument("--gold_file", type = str, default="", help="gold sql path")

    opt = parser.parse_args()
    print(opt)

    input_dataset = json.load(open(opt.input_file))
    

    max_model_len = 8192 # used to allocate KV cache memory in advance
    max_input_len = 6144
    max_output_len = 2048 # (max_input_len + max_output_len) must <= max_model_len
    
    print("max_model_len:", max_model_len)
    print("temperature:", opt.temperature)

    if opt.pretrained_model_name_or_path.startswith("openai/"):
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            n = opt.n,
            stop=["</sql>", "</solution>"],
        )
        
        use_openai = True
        llm = None  # placeholder
    else:
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            n = opt.n,
            # stop_token_ids = stop_token_ids, # NOTE(shu): not sure if we need this 
            stop=["</sql>", "</solution>"],
            include_stop_str_in_output=True,
        )
            
        use_openai = False
        llm = LLM(
            model = opt.pretrained_model_name_or_path,
            tensor_parallel_size = opt.tensor_parallel_size,
            max_model_len = max_model_len,
            trust_remote_code = True
        )
    
    print(f"OPT Prompt Format: {opt.prompt_format}")
    
    if use_openai:
        ############# Logic of Using OpenAI calls #################
        system_prompt = THINKING_SYSTEM
        chat_prompts = [
            [   {
                    "role": "system",
                    "content": THINKING_SYSTEM+f"/n/n You are given limited {opt.num_turns} turns to address the problem."
                },
                {
                    "role": "user",
                    "content": "{db_details}:"+ data_entry["schema"] + ";\n\n {external_knowledge}: " + data_entry["external_knowledge"] + ";\n\n {question}: " + data_entry["question"],
                }
            ] 
        for data_entry in input_dataset]
        
        # NOTE(shu): no schema does not work, it does not even know how to go into the table, and will hallucinate 
        # system_prompt = NO_SCHEMA_THINKING_SYSTEM
        # chat_prompts = [
        #     [   {
        #             "role": "system",
        #             "content": NO_SCHEMA_THINKING_SYSTEM+f"/n/n You are given limited {opt.num_turns} to address the problem."
        #         },
        #         {
        #             "role": "user",
        #             "content": ";\n\n {external_knowledge}: " + data_entry["external_knowledge"] + ";\n\n {question}: " + data_entry["question"],
        #         }
        #     ] 
        # for data_entry in input_dataset]
        
        # Run OpenAI generate multi-turn
        db_files = get_db_paths(opt.dataset, opt.db_path, opt.gold_file, input_dataset)
        model_name = opt.pretrained_model_name_or_path.split("/")[1]
        outputs = openai_generate_multi_turn(model_name, chat_prompts, db_files, sampling_params, opt.num_turns, opt.prompt_format)
        
        # Append results 
        results = []
        for data, prompt, output in zip(input_dataset, chat_prompts, outputs):
            sql  = parse_response(output, opt.prompt_format)
            
            data["input_seq"] = prompt
            data["responses"] = [output]
            data["pred_sqls"] = [sql]
            
            # Remove unwanted fields
            data.pop("schema", None)
            data.pop("external_knowledge", None)
            
            # TODO: print number of turns 
            num_turns_taken = sum(1 for msg in data["input_seq"] if msg["role"] == "assistant")
            data["num_turns"] = num_turns_taken
            
            results.append(data)
        
        with open(opt.output_file, "w", encoding = "utf-8") as f:
             f.write(json.dumps(results, indent = 2, ensure_ascii = False))
    else: 
        ############# Logic of Not Using OpenAI calls #################
        # REMOVE ALL RL or template like that 
        tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path, trust_remote_code=True)
        
        chat_prompts = []
        for i, data_entry in enumerate(input_dataset):
            prompt = {
                "role": "system",
                "content": THINKING_SYSTEM + f"\n\nYou are given limited {opt.num_turns} turns to address the problem."
            }
            user_input = {
                "role": "user",
                "content": "{db_details}:" + data_entry["schema"] + ";\n\n" +
                        "{external_knowledge}: " + data_entry["external_knowledge"] + ";\n\n" +
                        "{question}: " + data_entry["question"]
            }
            chat = [prompt, user_input]
            rendered = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
            chat_prompts.append(rendered)

        db_files = get_db_paths(opt.dataset, opt.db_path, opt.gold_file, input_dataset)
        outputs = generate_multi_turn(chat_prompts, db_files, sampling_params, opt.num_turns, opt.prompt_format)
        
        results = []
        for data, prompt, output in zip(input_dataset, chat_prompts, outputs):
            sql  = parse_response(output, opt.prompt_format)
            
            data["input_seq"] = prompt
            data["responses"] = [output]
            data["pred_sqls"] = [sql]
            
            # Remove unwanted fields
            data.pop("schema", None)
            data.pop("external_knowledge", None)
            
            results.append(data)

        with open(opt.output_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, indent = 2, ensure_ascii = False))
