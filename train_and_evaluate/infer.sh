# python3 infer.py \
#     --pretrained_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --input_file ./data/test_spider2_sqlite.json \
#     --output_file results/Llama-3.1-8B-Instruct_test_spider2_sqlite/greedy_search.json \
#     --tensor_parallel_size 2 \
#     --n 1 \
#     --temperature 0.0

# python3 infer.py \
#     --pretrained_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --input_file ./data/test_spider2_sqlite.json \
#     --output_file results/Llama-3.1-8B-Instruct_test_spider2_sqlite/sampling.json \
#     --tensor_parallel_size 2 \
#     --n 4 \
#     --temperature 0.8

python3 infer.py \
    --pretrained_model_name_or_path seeklhy/OmniSQL-7B \
    --input_file ./data/dev_bird.json \
    --output_file results/OmniSQL-7B_dev_bird/greedy_search.json \
    --tensor_parallel_size 2 \
    --n 1 \
    --temperature 0.0

python3 infer.py \
    --pretrained_model_name_or_path seeklhy/OmniSQL-7B \
    --input_file ./data/dev_bird.json \
    --output_file results/OmniSQL-7B_dev_bird/sampling.json \
    --tensor_parallel_size 2 \
    --n 4 \
    --temperature 0.8
