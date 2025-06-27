WORKDIR="/home/xxx/CLCoSum/CLCoSum4CodeT5"
DATADIR_ROOT="/home/xxx/CLCoSum"
export PYTHONPATH=$WORKDIR

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
TRAIN_BS=${6}
EVAL_BS=${7}
LR=${8}
SRC_LEN=${9}
TRG_LEN=${10}
PATIENCE=${11}
EPOCH=${12}
WARMUP=${13}
WEIGHT_DECAY=${14}
MODEL_DIR=${15}
SUMMARY_DIR=${16}
RES_FN=${17}
FIRST_EPOCH_NUM=${18}
SCHEDULER_FUN=${19}
LANG=${20}

if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
  EPOCH=1
fi

FULL_MODEL_TAG=${MODEL_TAG}_${SCHEDULER_FUN}_${LANG}_lr${LR}_wd${WEIGHT_DECAY}_bs${TRAIN_BS}_src${SRC_LEN}_trg${TRG_LEN}_fen${FIRST_EPOCH_NUM}_e${EPOCH}


if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi


CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
TRAIN_LOG=${OUTPUT_DIR}/train.log
TEST_LOG=${OUTPUT_DIR}/test.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}


if [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
fi

RUN_FN=${WORKDIR}/run_cl.py


CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN} ${MULTI_TASK_AUG} \
  --do_train --do_eval --do_eval_bleu \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM} \
  --num_train_epochs ${EPOCH} --warmup_rate ${WARMUP}e-1 --weight_decay ${WEIGHT_DECAY}e-3 --learning_rate ${LR}e-5 \
  --patience ${PATIENCE} --tokenizer_name=${TOKENIZER} --model_name_or_path=${MODEL_PATH} --data_dir ${DATADIR_ROOT}/dataset \
  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${TRAIN_BS} --eval_batch_size ${EVAL_BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  --scheduler_fun ${SCHEDULER_FUN} --first_epoch_num ${FIRST_EPOCH_NUM} --lang ${LANG} \
  2>&1 | tee ${TRAIN_LOG}

CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN} ${MULTI_TASK_AUG} \
  --do_test \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM} \
  --tokenizer_name=${TOKENIZER} --model_name_or_path=${MODEL_PATH} --data_dir ${DATADIR_ROOT}/dataset \
  --cache_path ${CACHE_DIR} --output_dir ${OUTPUT_DIR} --summary_dir ${SUMMARY_DIR} \
  --res_dir ${RES_DIR} --res_fn ${RES_FN} --eval_batch_size ${EVAL_BS} \
  --max_source_length 768 --max_target_length ${TRG_LEN} --lang ${LANG} \
  2>&1 | tee ${TEST_LOG}
