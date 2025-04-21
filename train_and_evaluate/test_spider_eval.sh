#!/bin/bash

model="o4-mini"
eval_dir="./evaluation_results"
data_dir="./data"
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
    --ts_db_path ./test_suite_sql_eval/test_suite_database \
    > ${log_dir}/${model}-high_test-spider/eval_log.txt