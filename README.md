## HairFlash: Realistic and Robust Hair Transfer with a Fast Encoder-Based Approach

This repository is the source code of my 3rd year course work in the HSE University: 
  
  
## Prerequisites
You need following hardware and python version to run the method.
- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.10
- PyTorch 1.13.1+

## Installation

* Clone this repo:
```bash
git clone git@github.com:TmBoris/HairFlash.git
cd HairFlash
```

* Download all pretrained models:

Download method weights from [here](https://disk.yandex.ru/d/orPsLRnDpC0m7A) into ./pretrained_models 

Download loss weights from [here](https://disk.yandex.ru/d/Tds5J3nJScEEgw) into ./losses/lpips/weights

* Setting the environment

```bash
pip install -r requirements.txt
```

## Inference
You can use `main.py` to run the method, either for a single run or for a batch of experiments.

* An example of running a single experiment:

```
python main.py --face_path=6.png --shape_path=7.png --color_path=8.png \
    --input_dir=input --result_path=output/result.png
```

* To run the batch version, first create an image triples file (face/shape/color):
```
cat > example.txt << EOF
6.png 7.png 8.png
8.png 4.jpg 5.jpg
EOF
```

And now you can run the method:
```
python main.py --file_path=example.txt --input_dir=input --output_dir=output
```

## Scripts

There is a list of scripts below.

| Path                                    | Description <img width=200>
|:----------------------------------------| :---
| scripts/align_face.py                   | Processing of raw photos for inference
| scripts/fid_metric.py                   | Metrics calculation
| scripts/rotate_gen.py                   | Dataset generation for rotate training
| scripts/blending_gen.py                 | Dataset generation for blending training
| scripts/pp_gen.py                       | Dataset generation for post processing training
| scripts/rotate_train.py                 | Rotate training
| scripts/blending_train.py               | Blending training
| scripts/pp_train.py                     | Post processing training


## Training
For training, you need to generate a dataset and then run the scripts for training. See the scripts section above.

We use [Weights & Biases](https://wandb.ai/home) to track experiments. Before training, you should put your W&B API key into the `WANDB_KEY` environment variable.


## Repository structure

    .
    ├── 📂 datasets                   # Implementation of torch datasets for inference
    ├── 📂 models                     # Folder containting all the models
    │   ├── ...
    │   ├── 📄 Embedding.py               # Implementation of Embedding module
    │   ├── 📄 Alignment.py               # Implementation of Alignment module
    │   ├── 📄 Blending.py                # Implementation of Blending module
    │   ├── 📄 Encoders.py                # Implementation of encoder architectures
    │   └── 📄 Net.py                     # Implementation of basic models
    │
    ├── 📂 losses                     # Folder containing various loss criterias for training
    ├── 📂 scripts                    # Folder with various scripts
    ├── 📂 utils                      # Folder with utility functions
    │
    ├── 📜 requirements.txt           # Lists required Python packages.
    ├── 📄 hair_swap.py               # Implementation of the HairFast main class
    └── 📄 main.py                    # Script for inference
