#!/bin/bash

# export OPENAI_API_KEY=xx

path_to_data="/shared/dcli/lshu/BIRD/OmniSQL/train_and_evaluate/data"
model="o4-mini"

python3 openai_infer.py \
    --model $model \
    --input_file "$path_to_data/test_spider.json" \
    --output_file "results/$model-high_test-spider/greedy_search.json" \
    --reasoning_effort high \
    --n 1

python3 openai_infer.py \
    --model $model \
    --input_file "$path_to_data/dev_spider.json" \
    --output_file "results/$model-high_dev-spider/greedy_search.json" \
    --reasoning_effort high \
    --n 1

python3 openai_infer.py \
    --model $model \
    --input_file "$path_to_data/test_spider2_sqlite.json" \
    --output_file "results/$model-high_test-spider2/greedy_search.json" \
    --reasoning_effort high \
    --n 1

python3 openai_infer.py \
    --model $model \
    --input_file "$path_to_data/dev_bird.json" \
    --output_file "results/$model-high_dev-bird/greedy_search.json" \
    --reasoning_effort high \
    --n 1


##########
# python3 infer.py \
#     --pretrained_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --input_file ./data/test_spider2_sqlite.json \
#     --output_file results/Llama-3.1-8B-Instruct_test_spider2_sqlite/sampling.json \
#     --tensor_parallel_size 2 \
#     --n 4 \
#     --temperature 0.8


# python3 infer.py \
#     --pretrained_model_name_or_path seeklhy/OmniSQL-7B \
#     --input_file ./data/dev_bird.json \
#     --output_file results/OmniSQL-7B_dev_bird/sampling.json \
#     --tensor_parallel_size 2 \
#     --n 4 \
#     --temperature 0.8