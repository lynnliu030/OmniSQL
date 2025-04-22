#!/usr/bin/env python3
import argparse
import orjson
import re
from pathlib import Path
from tqdm import tqdm


RL_TEMPLATE = (
    """
    Your response format MUST follow the template below. 
    - <think> Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution.</think>
    - After reasoning, if you find you lack some knowledge or confidence, you can use SQL tool written within a single <sql> your test sql query </sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>.
    - <solution>Final SQL query solution presented to the user.</solution>
    """
)

NEW_TEMPLATE = (
"""
Your response format MUST follow the template below.
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
<solution>
```sql
Final SQL query solution presented to the user.
```
</solution>
"""
)

NEW_MULTITURN_TEMPLATE = (
"""
Your response format MUST follow the template below.
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
<sql>
```sql
After reasoning, if you find you lack some knowledge or confidence, you can use SQL written here to explore or verify.
Output from your SQL code will be shown as a dataframe inside <observation>...</observation>.
You must then continue reasoning on the output with <think>...</think>.
You may use this SQL tool up to 3 times before providing your final solution.
```
</sql>
<solution>
```sql
Final SQL query solution presented to the user.
```
</solution>
"""
)

def rl_input_from_base(text: str) -> str:
    pattern = re.compile(
        r"Output Format:[\s\S]*?Take a deep breath and think step by step to "
        r"find the correct SQL query\.\s*$",
        flags=re.MULTILINE,
    )
    return pattern.sub(RL_TEMPLATE, text)

def rl_output_from_base(text: str) -> str:
    m = re.search(r"```sql\s*([\s\S]+?)```", text, re.MULTILINE)
    if not m:
        return text
    sql = m.group(1).strip()
    before = text[: m.start()].strip()
    after = text[m.end():].strip()
    think_parts = "\n\n".join(p for p in (before, after) if p)
    return f"<think>{think_parts}</think>\n<solution>{sql}</solution>"

def new_input_from_base(text: str, multiturn: bool = False) -> str:
    pattern = re.compile(
        r"Output Format:[\s\S]*?Take a deep breath and think step by step to "
        r"find the correct SQL query\.\s*$",
        flags=re.MULTILINE,
    )
    if not multiturn:
        return pattern.sub(NEW_TEMPLATE, text)
    else:
        return pattern.sub(NEW_MULTITURN_TEMPLATE, text)

def new_output_from_base(text: str) -> str:
    m = re.search(r"```sql\s*([\s\S]+?)```", text, re.MULTILINE)
    if not m:
        return text
    sql = m.group(1).strip()
    before = text[: m.start()].strip()
    after = text[m.end():].strip()
    think_parts = "\n\n".join(p for p in (before, after) if p)
    return f"<think>\n{think_parts}\n</think>\n<solution>\n```sql\n{sql}\n```\n</solution>"

def transform_to_rl_from_base(item: dict) -> dict:
    item["input_seq"] = rl_input_from_base(item["input_seq"]).rstrip()
    item["output_seq"] = rl_output_from_base(item["output_seq"])
    return item

def transform_to_new_from_base(item: dict) -> dict:
    item["input_seq"] = new_input_from_base(item["input_seq"]).rstrip()
    item["output_seq"] = new_output_from_base(item["output_seq"])
    return item

def main(input: Path, output: Path, format: str):
    if format == "rl":
        transform_fn = transform_to_rl_from_base
    elif format == "new":
        transform_fn = transform_to_new_from_base
    else:
        raise ValueError(f"Unknown format: {format}")
    data = orjson.loads(input.read_bytes())
    new_data = [transform_fn(obj.copy()) for obj in tqdm(data, desc="Processing")]
    output.write_bytes(
        orjson.dumps(new_data, option=orjson.OPT_INDENT_2)
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--format", type=str, choices=["rl", "new"], default="rl")
    main(**vars(ap.parse_args()))
