#!/bin/bash

model="o4-mini"
eval_dir="./evaluation_results"
data_dir="/shared/dcli/lshu/BIRD/OmniSQL/train_and_evaluate/data"
log_dir=$eval_dir

# Spider test evaluation

# only create folder if it doesn't exist
if [ ! -d "${log_dir}/${model}-high_test-spider" ]; then
    mkdir -p ${log_dir}/${model}-high_test-spider
fi
python3 evaluate.py \
    --source spider \
    --eval_name ${model}-high_test-spider \
    --gold_file ${data_dir}/spider/test_gold.sql \
    --db_path ${data_dir}/spider/test_database \
    --ts_db_path /shared/dcli/lshu/BIRD/openai/OmniSQL/train_and_evaluate/test_suite_sql_eval/test_suite_database \
    > ${log_dir}/${model}-high_test-spider/eval_log.txt

# Spider dev evaluation
# if [ ! -d "${log_dir}/${model}-high_dev-spider" ]; then
#     mkdir -p ${log_dir}/${model}-high_dev-spider
# fi
# python3 evaluate.py \
#     --source spider \
#     --eval_name ${model}-high_dev-spider \
#     --gold_file ${data_dir}/spider/dev_gold.sql \
#     --db_path ${data_dir}/spider/database \
#     --ts_db_path ./test_suite_sql_eval/test_suite_database \
#     > ${eval_dir}/${model}-high_dev-spider/eval_log.txt


# # Spider 2.0 test evaluation
# only create folder if it doesn't exist
# if [ ! -d "${log_dir}/${model}-high_test-spider2" ]; then
#     mkdir -p ${log_dir}/${model}-high_test-spider2
# python3 evaluate.py \
#     --source spider2.0 \
#     --eval_name ${model}-high_test-spider2 \
#     --gold_file ${data_dir}/spider2_sqlite/test.json \
#     --db_path ${data_dir}/spider2_sqlite/databases \
#     --gold_result_dir ${data_dir}/spider2_sqlite/gold_exec_result \
#     --eval_standard ${data_dir}/spider2_sqlite/spider2_sqlite_eval.jsonl \
#     --log_file ${eval_dir}/${model}-high_test-spider2/eval_log_v2.txt

# BIRD dev evaluation
# only create folder if it doesn't exist
# if [ ! -d "${log_dir}/${model}-high_dev-bird" ]; then
#     mkdir -p ${log_dir}/${model}-high_dev-bird
# python3 evaluate.py \
#     --source bird \
#     --eval_name ${model}-high_dev-bird \
#     --gold_file ${data_dir}/bird/dev_20240627/dev.json \
#     --db_path ${data_dir}/bird/dev_20240627/dev_databases \
#     > ${eval_dir}/${model}-high_dev-bird/eval_log.txt

