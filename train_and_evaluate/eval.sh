# python3 evaluate.py \
#     --source spider2.0 \
#     --eval_name Llama-3.1-8B-Instruct_test_spider2_sqlite \
#     --gold_file ./data/spider2_sqlite/test.json \
#     --db_path ./data/spider2_sqlite/databases/ \
#     --gold_result_dir ./data/spider2_sqlite/gold_exec_result/ \
#     --eval_standard ./data/spider2_sqlite/spider2_sqlite_eval.jsonl

# python3 evaluate.py \
#     --source bird \
#     --eval_name o3-mini-high_dev-bird \
#     --gold_file ./data/bird/dev_20240627/dev.json \
#     --db_path ./data/bird/dev_20240627/dev_databases \
#     > ./evaluation_results/o3-mini-high_dev-bird/eval_log.txt

# python3 evaluate.py \
#     --source spider \
#     --eval_name o3-mini-high_dev-spider \
#     --gold_file ./data/spider/dev_gold.sql \
#     --db_path ./data/spider/database \
#     --ts_db_path ./test_suite_sql_eval/test_suite_database \
#     > ./evaluation_results/o3-mini-high_dev-spider/eval_log.txt

# python3 evaluate.py \
#     --source spider \
#     --eval_name o3-mini-high_test-spider \
#     --gold_file ./data/spider/test_gold.sql \
#     --db_path ./data/spider/test_database \
#     > ./evaluation_results/o3-mini-high_test-spider/eval_log.txt

python3 evaluate.py \
    --source spider2.0 \
    --eval_name o3-mini-high_test-spider2 \
    --gold_file ./data/spider2_sqlite/test.json \
    --db_path ./data/spider2_sqlite/databases/ \
    --gold_result_dir ./data/spider2_sqlite/gold_exec_result/ \
    --eval_standard ./data/spider2_sqlite/spider2_sqlite_eval.jsonl \
    --log_file ./evaluation_results/o3-mini-high_test-spider2/eval_log_v2.txt
