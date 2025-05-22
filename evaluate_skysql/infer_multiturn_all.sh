DB_ROOT=$1
TEST_SUITE_ROOT=$2
SKYRL_EVAL=$3

MODEL_PATHS=(
    "NovaSky-AI/SkyRL-SQL-7B"
    "Qwen/Qwen2.5-Coder-7B-Instruct" 
)
SAVE_NAMES=(
    "MULTITURN/SkyRL-SQL-7B"  
    "MULTITURN/Qwen2.5-Coder-7B-Instruct"
)

MODEL_TYPES=("local" "local")
N_SAMPLES=(1 1)
TENSOR_PARALLEL_SIZES=(4 4)
TEMPERATURE=0.0


if [ "${#MODEL_PATHS[@]}" -ne "${#SAVE_NAMES[@]}" ]; then
  echo "❌ arrays must be same length" >&2
  exit 1
fi


for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    SAVE_NAME=${SAVE_NAMES[$i]}
    MODEL_TYPE=${MODEL_TYPES[$i]}
    N_SAMPLE=${N_SAMPLES[$i]}
    TP=${TENSOR_PARALLEL_SIZES[$i]}
    echo "Running inference for ${SAVE_NAME} with model ${MODEL_PATH}"

  # Test spider
  SAVE_DIR="./evaluation_results/${SAVE_NAME}/test_spider"
  python3 infer_multiturn.py \
      --pretrained_model_name_or_path $MODEL_PATH \
      --dataset spider \
      --input_file $SKYRL_EVAL/test_spider.json \
      --output_dir $SAVE_DIR \
      --num_turns 5 \
      --model_type $MODEL_TYPE \
      --n $N_SAMPLES \
      --temperature 0.0 \
      --gold_file $DB_ROOT/spider/test_gold.sql \
      --db_path $DB_ROOT/spider/test_database \
      --prompt_format rl \
      --eval \
      --infer \
      --tensor_parallel_size $TP \
      --overwrite

  # dev spider
  SAVE_DIR="./evaluation_results/${SAVE_NAME}/dev_spider"
  python3 infer_multiturn.py \
      --pretrained_model_name_or_path $MODEL_PATH \
      --dataset spider \
      --input_file $SKYRL_EVAL/dev_spider.json \
      --output_dir $SAVE_DIR \
      --num_turns 5 \
      --model_type $MODEL_TYPE \
      --n $N_SAMPLES \
      --temperature 0.0 \
      --gold_file $DB_ROOT/spider/dev_gold.sql \
      --db_path $DB_ROOT/spider/database \
      --ts_db_path $TEST_SUITE_ROOT/test_suite_database \
      --prompt_format rl \
      --eval \
      --infer \
      --tensor_parallel_size $TP \
      --overwrite

  # Dev spider dk 
  SAVE_DIR="./evaluation_results/${SAVE_NAME}/dev_spider_dk"
  python3 infer_multiturn.py \
      --pretrained_model_name_or_path $MODEL_PATH \
      --dataset spider \
      --input_file $SKYRL_EVAL/dev_spider_dk.json \
      --output_dir $SAVE_DIR \
      --num_turns 5 \
      --model_type $MODEL_TYPE \
      --n $N_SAMPLES \
      --temperature 0.0 \
      --gold_file $DB_ROOT/Spider-DK/spider_dk_gold.sql \
      --db_path $DB_ROOT/Spider-DK/database \
      --prompt_format rl \
      --eval \
      --infer \
      --tensor_parallel_size $TP \
      --overwrite

  # dev spider realistic
  SAVE_DIR="./evaluation_results/${SAVE_NAME}/dev_spider_realistic"
  python3 infer_multiturn.py \
      --pretrained_model_name_or_path $MODEL_PATH \
      --dataset spider \
      --input_file $SKYRL_EVAL/dev_spider_realistic.json  \
      --output_dir $SAVE_DIR \
      --num_turns 5 \
      --model_type $MODEL_TYPE \
      --n $N_SAMPLES \
      --temperature 0.0 \
      --gold_file $DB_ROOT/spider-realistic/spider_realistic_gold.sql \
      --db_path $DB_ROOT/spider/database \
      --ts_db_path $TEST_SUITE_ROOT/test_suite_database \
      --prompt_format rl \
      --eval \
      --infer \
      --tensor_parallel_size $TP \
      --overwrite

  # dev spider synthetic
  SAVE_DIR="./evaluation_results/${SAVE_NAME}/dev_spider_syn"
  python3 infer_multiturn.py \
      --pretrained_model_name_or_path $MODEL_PATH \
      --dataset spider \
      --input_file $SKYRL_EVAL/dev_spider_syn.json  \
      --output_dir $SAVE_DIR \
      --num_turns 5 \
      --model_type $MODEL_TYPE \
      --n $N_SAMPLES \
      --temperature 0.0 \
      --gold_file $DB_ROOT/Spider-Syn/spider_syn_gold.sql \
      --db_path $DB_ROOT/spider/database \
      --ts_db_path $TEST_SUITE_ROOT/test_suite_database \
      --prompt_format rl \
      --eval \
      --infer \
      --tensor_parallel_size $TP \
      --overwrite
      
  done
