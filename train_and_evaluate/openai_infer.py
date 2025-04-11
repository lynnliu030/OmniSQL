import argparse
import json
import re
# from vllm import LLM, SamplingParams
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "gpt-4o-mini")
    parser.add_argument("--input_file", type = str, help = "the input file path (prompts)")
    parser.add_argument("--output_file", type = str, help = "the output file path (results)")
    parser.add_argument("--n", type = int, help = "the number of generated responses", default = 4)
    parser.add_argument("--reasoning_effort", type = str, help = "reasoning effort to use", default = "medium")

    opt = parser.parse_args()
    print(opt)

    input_dataset = json.load(open(opt.input_file))

    # max_completion_len = 8000
    # print("max_completion_len:", max_completion_len)
    print("reasoning_effort:", opt.reasoning_effort)

    # openai client
    client = OpenAI()
    
    chat_messages = [[{"role": "user", "content": data["input_seq"]}] for data in input_dataset]

    def run_generation(messages, idx):
        response = client.chat.completions.create(
            model=opt.model,
            messages=messages,
            reasoning_effort=opt.reasoning_effort,
            # max_completion_tokens=max_completion_len,
            n=opt.n,
        )
        return response.choices, idx

    pbar = tqdm(total=len(chat_messages), desc=f"Infer")
    outputs = [None] * len(chat_messages)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(run_generation, prompt, idx)
            for idx, prompt in enumerate(chat_messages)
        ]
        
        for future in as_completed(futures):
            try:
                output, idx = future.result()
                outputs[idx] = output
            except Exception as e:
                print(f"Error generating chat: {e}")
            finally:
                pbar.update(1)
    
    results = []
    for data, output in zip(input_dataset, outputs):
        if output is None: # skip failed generations
            data["responses"] = []
            data["pred_sqls"] = []
        else:
            responses = [o.message.content for o in output]
            sqls  = [parse_response(response) for response in responses]
            
            data["responses"] = responses
            data["pred_sqls"] = sqls
        results.append(data)

    os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
    with open(opt.output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(results, indent = 2, ensure_ascii = False))
