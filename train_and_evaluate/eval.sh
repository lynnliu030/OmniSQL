# python3 evaluate.py \
#     --source spider2.0 \
#     --eval_name Llama-3.1-8B-Instruct_test_spider2_sqlite \
#     --gold_file ./data/spider2_sqlite/test.json \
#     --db_path ./data/spider2_sqlite/databases/ \
#     --gold_result_dir ./data/spider2_sqlite/gold_exec_result/ \
#     --eval_standard ./data/spider2_sqlite/spider2_sqlite_eval.jsonl

python3 evaluate.py \
    --source bird \
    --eval_name OmniSQL-7B_dev_bird \
    --gold_file ./data/bird/dev_20240627/dev.json \
    --db_path ./data/bird/dev_20240627/dev_databases
