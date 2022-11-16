# STRAP: Structured Object Affordance Segmentation with Point Supervision

This is an official implementation of STRAP: Structured Object Affordance Segmentation with Point Supervision.

## Requirements

### Environments

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

### Dataset

#### CAD120

- Download CAD120 affordance dataset from [here](https://zenodo.org/record/495570) and point annotations are stored in `./data_preprocess/CAD120/keypoints.txt`.

#### Data Preprocessing

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

You need to modify the `train_cad120_object.yaml` or `train_cad120_actor.yaml` to customize your dataset path by configuring the `data_dir`.

### Step 2

You can use

```
python <script> --config <configuration>
```

to start to training. The options of `<script>` include `main.py`, `main_hc.py`, ~
