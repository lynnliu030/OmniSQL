set -e

MODEL_PATH='openai/gpt-4o-mini'
INPUT_PATH="/shared/dcli/lshu/BIRD/OmniSQL/rollout/data/RL_format_10_from130k.json"

PATH_TO_RLEF="/shared/dcli/lshu/BIRD/multi-turn-RL-code/bird-table/data"
DB_PATH="$PATH_TO_RLEF/SynSQL-2.5M/databases"
TP=4
TURNS=5
TEMP=0.5

MODEL_NAME="${MODEL_PATH#openai/}"
OUTPUT_PATH="./rollouts/${MODEL_NAME}-${TURNS}-turns-test.json"


# No need prompt format, auto multiturn 
python3 infer.py \
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