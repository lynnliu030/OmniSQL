set -e

MODEL_PATH="/shared/dcli/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7"
INPUT_PATH="/shared/dcli/lshu/BIRD/OmniSQL/rollout/data/RL_format_10_from130k.json"

PATH_TO_RLEF="/shared/dcli/lshu/BIRD/multi-turn-RL-code/bird-table/data"
DB_PATH="$PATH_TO_RLEF/SynSQL-2.5M/databases"
TP=4
TURNS=5
TEMP=0.5

MODEL_NAME="Qwen2.5-Coder-32B-Instruct"
OUTPUT_PATH="./rollouts/${MODEL_NAME}-${TURNS}-turns-test.json"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 infer.py \
--pretrained_model_name_or_path $MODEL_PATH \
--input_file $INPUT_PATH \
--output_file $OUTPUT_PATH \
--tensor_parallel_size $TP \
--n 1 \
--num_turns $TURNS \
--temperature $TEMP \
--use_multiturn \
--dataset synsql \
--db_path $DB_PATH 