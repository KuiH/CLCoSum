TASK="summarize"
SUB_TASK="tlcodesum_clean+delchar+tofunc_tokenlen_411"
SCHEDULER_FUN="S5"
LANG="java"
MODEL_TAG="unix-base"
GPU="5,6,7"
LR=5
EPOCH=10
WEIGHT_DECAY=0
WARMUP_RATE=1
MODEL_DIR="saved_models"
SUMMARY_DIR="tensorboard"

if [ "$SCHEDULER_FUN" != "std" ]; then
  if [[ $SUB_TASK == *"tlcodesum"* ]]; then
      FIRST_EPOCH_NUM=53597
  elif [[ $SUB_TASK == *"pcsd"* ]]; then
      FIRST_EPOCH_NUM=57849      
  fi
else
    FIRST_EPOCH_NUM=-1
fi

mkdir -p results
RES_FN="results/${TASK}_${MODEL_TAG}.txt"

if [[ $SUB_TASK == *"tlcodesum"* || $SUB_TASK == *"pcsd"* ]]; then
    DATA_DIR="../dataset/${TASK}/${SUB_TASK}"
    SRC_LEN=384
    TRG_LEN=32
    TRAIN_BS=32
    EVAL_BS=24
fi


TRAIN_FILE="${DATA_DIR}/train.jsonl"
DEV_FILE="${DATA_DIR}/val.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl"

FULL_MODEL_TAG=${MODEL_TAG}_${SCHEDULER_FUN}_${LANG}_lr${LR}_wd${WEIGHT_DECAY}_bs${TRAIN_BS}_src${SRC_LEN}_trg${TRG_LEN}_fen${FIRST_EPOCH_NUM}_e${EPOCH}

OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}

# only for test
# OUTPUT_DIR=/home/xxx/CLCoSum/CLCoSum4UniXCoder/saved_models/summarize/pcsd_clean_repchar/from_pcsd_clean_raw


RES_DIR=${OUTPUT_DIR}/prediction
TRAIN_LOG=${OUTPUT_DIR}/train.log
TEST_LOG=${OUTPUT_DIR}/test.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == unix-base ]]; then
  MODEL_PATH=microsoft/unixcoder-base
fi

CUDA_VISIBLE_DEVICES=${GPU} \
  python run_clcosum.py --do_train --do_eval --task $TASK --sub_task $SUB_TASK \
  --model_name_or_path $MODEL_PATH \
  --train_filename $TRAIN_FILE --dev_filename $DEV_FILE --output_dir $OUTPUT_DIR --summary_dir ${SUMMARY_DIR} \
  --res_dir ${RES_DIR} --res_fn ${RES_FN} --max_source_length $SRC_LEN --max_target_length $TRG_LEN \
  --warmup_rate ${WARMUP_RATE}e-1 --weight_decay ${WEIGHT_DECAY}e-3 --learning_rate ${LR}e-5 --num_train_epochs $EPOCH \
  --beam_size 5 --train_batch_size $TRAIN_BS --eval_batch_size $EVAL_BS \
  --scheduler_fun ${SCHEDULER_FUN} --first_epoch_num ${FIRST_EPOCH_NUM} --lang ${LANG} \
  2>&1| tee ${TRAIN_LOG}


CUDA_VISIBLE_DEVICES=${GPU} \
  python run_clcosum.py --do_test --task $TASK --sub_task $SUB_TASK \
  --model_name_or_path $MODEL_PATH \
  --test_filename $TEST_FILE --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --output_dir $OUTPUT_DIR --max_source_length 768 --max_target_length 64 \
  --beam_size 5 --eval_batch_size $EVAL_BS --lang ${LANG} \
  2>&1| tee $TEST_LOG