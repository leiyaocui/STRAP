#!/bin/bash

# You should select a split mode of CAD120 dataset. (object or actor)
SPLIT_MODE="object"
# You need to assign your dataset's root path which is preprocessed by what mentioned above.
DATASET_ROOT_PATH="../dataset/cad120"
# You can choose where to store the output of training.
OUTPUT_PATH_NAME="outputs"


# PREPARATION FOR CONFIGS
DATASET_SPLIT_PATH="${DATASET_ROOT_PATH}/${SPLIT_MODE}"
CURRENT_PATH="$(pwd)"
TIME=$(date "+%Y%m%d_%H%M%S")

OUTPUT_PATH="${CURRENT_PATH}/${OUTPUT_PATH_NAME}/${TIME}"
TEMPLATE_CONFIG_PATH="${CURRENT_PATH}/configs/train_cad120_${SPLIT_MODE}_template.yaml"

if [ ! -d "${OUTPUT_PATH}" ]; then
  mkdir -p "${OUTPUT_PATH}"
else
  echo "Output path: ${OUTPUT_PATH} is existed."
  echo "All processes will exit."
  exit 1
fi

FIRST_STAGE_PATH="${OUTPUT_PATH}/first_stage"

mkdir -p "${FIRST_STAGE_PATH}"
cp -r "${TEMPLATE_CONFIG_PATH}" "${FIRST_STAGE_PATH}/first.yaml"

sed -i "s,data_dir: \".*\",data_dir: \"${DATASET_SPLIT_PATH}\",g" \
       "${FIRST_STAGE_PATH}/first.yaml"
sed -i "s,save_dir: \".*\",save_dir: \"${FIRST_STAGE_PATH}\",g" \
       "${FIRST_STAGE_PATH}/first.yaml"
sed -i "s,image: 0.1,image: 0.01,g" \
       "${FIRST_STAGE_PATH}/first.yaml"
sed -i "s,resume: \".*\",resume: \"\"," \
       "${FIRST_STAGE_PATH}/first.yaml"
sed -i "s,restart: true,restart: false," \
       "${FIRST_STAGE_PATH}/first.yaml"

SECOND_STAGE_PATH="${OUTPUT_PATH}/second_stage"

mkdir -p "${SECOND_STAGE_PATH}"
cp -r "${TEMPLATE_CONFIG_PATH}" "${SECOND_STAGE_PATH}/second.yaml"

sed -i "s,data_dir: \".*\",data_dir: \"${DATASET_SPLIT_PATH}\"," \
       "${SECOND_STAGE_PATH}/second.yaml"
sed -i "s,save_dir: \".*\",save_dir: \"${SECOND_STAGE_PATH}\"," \
       "${SECOND_STAGE_PATH}/second.yaml"
sed -i "s,resume: \".*\",resume: \"${FIRST_STAGE_PATH}/model/checkpoint_latest.pth\"," \
       "${SECOND_STAGE_PATH}/second.yaml"
sed -i "s,restart: false,restart: true," \
       "${SECOND_STAGE_PATH}/second.yaml"

EM_STAGE_PATH="${OUTPUT_PATH}/em_stage"

mkdir -p "${EM_STAGE_PATH}"
cp -r "${TEMPLATE_CONFIG_PATH}" "${EM_STAGE_PATH}/em.yaml"

sed -i "s,data_dir: \".*\",data_dir: \"${DATASET_SPLIT_PATH}\"," \
       "${EM_STAGE_PATH}/em.yaml"
sed -i "s,save_dir: \".*\",save_dir: \"${EM_STAGE_PATH}\"," \
       "${EM_STAGE_PATH}/em.yaml"
sed -i "s,resume: \".*\",resume: \"${SECOND_STAGE_PATH}/model/checkpoint_latest.pth\"," \
       "${EM_STAGE_PATH}/em.yaml"
sed -i "s,restart: false,restart: true," \
       "${EM_STAGE_PATH}/em.yaml"


# THE FIRST STAGE
python "${PWD}/first_stage.py" --config "${FIRST_STAGE_PATH}/first.yaml"
if [ $? -ne 0 ]; then
  echo "first_stage.py throws some exceptions."
  echo "All processes will exit."
  exit 1
fi


# THE SECOND STAGE
python "${PWD}/second_stage.py" --config "${SECOND_STAGE_PATH}/second.yaml"
if [ $? -ne 0 ]; then
  echo "second_stage.py throws some exceptions."
  echo "All processes will exit."
  exit 1
fi


# THE EM STAGE
python "${PWD}/em_stage.py" --config "${EM_STAGE_PATH}/em.yaml"
if [ $? -ne 0 ]; then
  echo "em_stage.py throws some exceptions."
  echo "All processes will exit."
  exit 1
fi