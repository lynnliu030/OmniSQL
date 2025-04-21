set -e

LR=2e-5
EPOCHS=2
CONFIG_FILE="./accelerate_config_7b.yaml"
PER_DEVICE_TRAIN_BATCH_SIZE=1
MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
CKPT_NUM=10

BASE_NAME="qwencoder_7b_instruct_lr${LR}_epochs${EPOCHS}_filter_15k_in_domain_spider2_highly_complex"
DATASET_DIR="./data/train_synsql_downsampled_15k_in_domain_spider2_highly_complex.json"


# Downsample from 210k only highly complex --> 55k
# BASE_NAME="qwencoder_7b_instruct_lr${LR}_epochs${EPOCHS}_filter_in_domain_spider2_highly_complex_55k"
# DATASET_DIR="./data/train_synsql_filter_in_domain_spider2_highly_complex_55769.json"

# Simple downsample from 130k (random sample) --> 30k
# BASE_NAME="qwencoder_7b_instruct_lr${LR}_epochs${EPOCHS}_filter_domain_spider2_non_simple_moderate_downsample_70k"
# DATASET_DIR="./data/train_synsql_filter_in_domain_spider2_non_simple_moderate_downsample_70k.json"

# Downsample from number of queries per database 
# BASE_NAME="qwencoder_7b_instruct_lr${LR}_epochs${EPOCHS}_filter_domain_spider2_non_simple_moderate_20pdb_downsample_28k"
# DATASET_DIR="./data/train_synsql_domain_spider2_non_simple_moderate_20pdb_downsample_28k.json"

CKPT_DIR="./ckpts/$BASE_NAME"
LOG_DIR="./train_logs/$BASE_NAME"
# DATASET_DIR="./data/train_bird.json"
# DATASET_DIR="./data/train_spider.json"



accelerate launch --main_process_port 10000 --config_file $CONFIG_FILE train.py \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --block_size 8192 \
    --seed 42 \
    --pretrained_model_name_or_path $MODEL_PATH \
    --epochs $EPOCHS \
    --lr $LR \
    --ckpt_num $CKPT_NUM \
    --tensorboard_log_dir $LOG_DIR \
    --output_ckpt_dir $CKPT_DIR \
    --sft_data_dir $DATASET_DIR \
    --mode sft