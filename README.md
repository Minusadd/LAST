# Language-guided Skill Learning with Temporal Variational Inference (LAST)

_ICML 2024_

We present an algorithm for skill discovery from expert demonstrations. The algorithm first utilizes Large Language Models (LLMs) to propose an initial segmentation of the trajectories. Following that, a hierarchical variational inference framework incorporates the LLM-generated segmentation information to discover reusable skills by merging trajectory segments. To further control the trade-off between compression and reusability, we introduce a novel auxiliary objective based on the Minimum Description Length principle that helps guide this skill discovery process. 

**TO DO**: online hierarchical RL with the learned skills

## Setup
Clone repo:
```bash
$ git clone https://github.com/alexpashevich/E.T..git LAST
```

Install requirements:
```bash
$ virtualenv -p $(which python3.9) last
$ source last/bin/activate
$ cd LAST
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Downloading data and checkpoints
Install [ALFRED](https://github.com/askforalfred/alfred) and download the dataset.
```bash
$ git clone https://github.com/askforalfred/alfred.git alfred
$ cd alfred/data
$ sh download_data.sh json_feat
$ cd ../..
```


(Optional) Download the preprocessed features & LLM-generated data from a [google drive](https://drive.google.com/drive/folders/1CEjAzaY0rpEYzlZ-vnKEDIHOHv9gYbu1?usp=sharing).

## LLM-generated initial segmentation
Setup openai api key
```bash
$ export OPENAI_API_KEY='your api key'
$ export OPENAI_API_BASE='your api base'
$ export OPENAI_API_TYPE='your api type'
```
Generate trajectory data using gpt-4
```bash
$ mkdir data_gpt4
$ python alfred_steps.py --data_dir ./alfred/data/json_2.1.0 --output_dir data_gpt4/ --n_workers 4
```
Note 1: The .jpeg images from the `full` dataset are different from the images rendered during evaluation due to the JPG compression. Thus we generated images for all the trajectories on our own. We are still trying to figure out how to share this but you can generate it on your own with the code provided in [ET](https://github.com/alexpashevich/E.T.).  

Note 2: **You can directly download the gpt4-generated dataset we used from the [google drive](https://drive.google.com/file/d/1lu6Xb5wwaFCDJPV3AcUIKLL4ms3dWZ2B/view?usp=sharing) and skip this step.**

## Preprocess the data
Process the image and language data given the initial segmentation results
```bash
$ python process_data.py
````
Note: **You can directly download the processed data we used from the google drive and skip this step:**
[FasterRCNN](https://drive.google.com/file/d/1vwP7Av2XUGkRNkYobxcVczGZYMwCehQ3/view?usp=sharing)
, [MaskRCNN](https://drive.google.com/file/d/12ABvTURhSRn_NXFWK8M9M8rAIcugh9Wy/view?usp=sharing)
, [Image features](https://drive.google.com/file/d/1mBJr08hHOYohD4DDgyU1f3bJdbq6-QiH/view?usp=sharing)
, [Language features](https://drive.google.com/file/d/12E1icG6T7QxsqEKSyPUt8QRtyQZLE6jj/view?usp=sharing)
, [Goal features](https://drive.google.com/file/d/1qxWWpl0poI_zdNAz3AVpzpUHWveJ3aZ0/view?usp=sharing)
, [Masks](https://drive.google.com/file/d/171cB5XD06l2WdagHcbH3VaHZ7W_fzkUp/view?usp=sharing)
, [Action sequences](https://drive.google.com/file/d/1-H0478OAGzVnIk3ne5aOJmBXPdeVpPV8/view?usp=sharing)
, [Switching points](https://drive.google.com/file/d/18j8qE93IbuulUpIrthi_pbKT7J05PG0M/view?usp=sharing)
, [Processed trajectory data (gpt4)](https://drive.google.com/file/d/117po4UBq-LHptzSPei5bq2tWvOmD4ryU/view?usp=sharing)
You will need to put all the downloaded files into the data/ folder.
```bash
$ mkdir data
````

## Skill discovery with temporal variational inference

Train a LAST agent:
```bash
$ python algorithm.py --name train_last --train 1 --include_goal 1 --ent_weight 0.1 --kl_weight 0.0001
```
Evaluate the agent on the dataset:
```bash
$ python algorithm.py --name test_last --train 0 --include_goal 1 --ent_weight 0.1 --kl_weight 0.0001 --model saved_nets/Model_epoch70
```

## Citation

If you find this repository useful, please cite our work:
```
@inproceedings{fu2024languageskill,
  title     = {{Language-guided Skill Learning with Temporal Variational Inference}},
  author    = {Haotian Fu and Pratyusha Sharma and Elias Stengel-Eskin and George Konidaris and Nicolas Le Roux and Marc-Alexandre Côté and Xingdi Yuan},
  booktitle = {ICML},
  year      = {2024},
}
```
