# STRAP: Structured Object Affordance Segmentation with Point Supervision

This is an official implementation of STRAP: Structured Object Affordance Segmentation with Point Supervision.

## Installation

### Requirments

We have verified our codebase by pytorch == 1.12.1 with CUDA == 11.6 in python == 3.8.10, the following are requirements.

```
numpy==1.21.5
pillow==9.2.0
pytorch==1.12.1
pyyaml==6.0
scikit-image==0.19.2
scipy==1.7.3
tensorboard==2.9.0
timm==0.6.7
tqdm==4.64.0
```

### Data Preprocessing

#### Step 1

Download CAD120 affordance dataset from [here](https://zenodo.org/record/495570) and point annotations are stored in `./data_preprocess/CAD120/keypoints.txt`.

#### Step 2

Use `./datasets/CAD120/generate.py` to preprocess the dataset. In the meanwhile, modify the script to customize your own path.

The dataset after preprocessing is similar to the following.

```
cad120
├── actor
│   ├── images
│   ├── labels
│   ├── train_affordance_keypoint.yaml
│   ├── train_affordance.txt
│   └── val_affordance.txt
└── object
    ├── images
    ├── labels
    ├── train_affordance_keypoint.yaml
    ├── train_affordance.txt
    └── val_affordance.txt
```

## Training

### Step 1

Before starting your training, you need to modify some variables in `train.sh`. The following are some details.

```
# You should select a split mode of CAD120 dataset. (object or actor)
SPLIT_MODE="object"
# You should assign your dataset's root path which is preprocessed by what mentioned above.
DATASET_ROOT_PATH="../dataset/cad120"
 # You can choose where to store the output of training.
OUTPUT_PATH_NAME="outputs"
```

### Step 2

Use `sh train.sh` in terminal to start your training.

## Evaluation

We provide a jupyter notebook `visualize.ipynb` to get a visualized results.

The variables which you need to customize are shown as follows.

```
# You should select a split mode of CAD120 dataset. (object or actor)
split_mode = "object"
# You should assign your dataset's root path which is preprocessed by what mentioned above.
dataset_root_path = "../dataset/cad120"
# You should assign the path of your pre-trained model.
resume = "./model.pth"
```

## Pre-trained Models

| Split  | mIoU | URL |
| ------ | ---- | --- |
| Object |      |     |
| Actor  |      |     |

## Acknowledgments

The point annotations of CAD120 dataset are duplicated from [keypoints.txt](https://github.com/ykztawas/Weakly-Supervised-Affordance-Detection/blob/master/weakly_supervised_affordance_detection_master/expectation_step/keypoints.txt).

Parts of code are on based on the [Cerberus](https://github.com/OPEN-AIR-SUN/Cerberus) and [BINN](https://github.com/daveboat/structured_label_inference).
