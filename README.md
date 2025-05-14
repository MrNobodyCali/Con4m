# [NeurIPS 2024] Con4m: Context-aware Consistency Learning Framework for Segmented Time Series Classification

<div align="center">

### [<a href="https://openreview.net/pdf?id=jCPufQaHvb" target="_blank">Paper</a>] [<a href="https://neurips.cc/media/neurips-2024/Slides/93973.pdf" target="_blank">Slides</a>] [<a href="https://recorder-v3.slideslive.com/#/share?share=94277&s=6e9b8303-8878-4d38-95d7-74706c75117a" target="_blank">Video</a>] [<a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202024/93973.png?t=1732180648.1711104" target="_blank">Poster</a>]

![case](https://github.com/user-attachments/assets/47fc6336-5735-490a-99e2-5902f985fa52)

_**[Junru Chen](https://mrnobodycali.github.io/), [Tianyu Cao](http://tiyacao.com/), Jing Xu, [Jiahe Li](https://erikaqvq.github.io/), Zhilong Chen, Tao Xiao, [Yang Yang<sup>†</sup>](http://yangy.org/)**_

Zhejiang University

</div>

## 📖 Introduction

**TL;DR:** Con4m is a consistency learning framework, which effectively utilizes contextual information more conducive 
to discriminating consecutive segments in segmented TSC tasks, while harmonizing inconsistent boundary labels for training.  <br>

![model](https://github.com/user-attachments/assets/2c44d1b7-8bd5-4dc4-8d9b-89b82a055e50)

## 🔥 Updates
- __[2025.05.14]__: Update the detailed README.
- __[2024.12.12]__: Release the [project codes](https://github.com/MrNobodyCali/Con4m).
- __[2024.11.21]__: Release the [poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/93973.png?t=1732180648.1711104).
- __[2024.11.12]__: Release the [slides](https://neurips.cc/media/neurips-2024/Slides/93973.pdf) and [short video](https://recorder-v3.slideslive.com/#/share?share=94277&s=6e9b8303-8878-4d38-95d7-74706c75117a).
- __[2024.09.26]__: Con4m is accepted by NeurIPS 2024 as a poster presentation.

## 🏁 Getting Started

### 1. Environment
You can install the necessary packages according to `requirements.txt`. Conda is suggested for the version compatibility and
other basic packages.

### 2. Public Datasets
Assume that the root path of the raw public datasets is `/data/eeggroup/public_dataset/`.
- [fNIRS](https://tufts.app.box.com/s/1e0831syu1evlmk9zx2pukpl3i32md6r/folder/144901078723) is downloaded in `/data/eeggroup/public_dataset/Tufts_fNIRS_data/`.
- [HHAR](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) is downloaded in `/data/eeggroup/public_dataset/hhar/`. 
Then use the [code](https://github.com/jc-audet/WOODS/blob/main/woods/scripts/fetch_and_preprocess.py) from WOODS to generate the .h5 file.
- [Sleep](https://physionet.org/content/sleep-edfx/1.0.0/) is downloaded in `/data/eeggroup/public_dataset/SleepEdf_Dataset/physionet.org/files/sleep-edfx/1.0.0/`.
- SEEG is a private dataset and unable to download.

You can also use your own datasets to run our model. The input shape of each raw data file should be `Timestamps x Features`, 
with corresponding `Timestamps` labels.
Also, you should write your own dataset generation file referring to the files in step 3.
Then add necessary information into `pipeline/ca_database_api.py`.
If you meet any questions, feel free to touch with me.

### 3. Dataset Generation (in `pipeline` folder)
Run the codes `b_fNIRS_database_generate.py`/`b_HHAR_database_generate.py`/`b_Sleep_database_generate.py` respectively to
generate random datasets for further training and testing.
- --load_dir: The path of raw datasets you download in.
- --save_dir: The path of generated datasets to save in.
- --noise_ratio: The disturbance ratio of raw data files.

The first run of these files may cost more time, because the codes of `a_data_file_load.py` will process the raw data files
to generate new data files for dataset generation and faster repeated runs.
The shape of generated datasets for each noise ratio, each group and each level is `SampleNum x Length x Features`.

### 4. Run the model
You can refer to the shell file `run_main.sh` to run the complete experiments of Con4m.
- --database_save_dir: The path of generated datasets in step 3.
- --path_checkpoint (`a_train.py`)/--load_path (`b_test.py`): The path of model checkpoint to save in/load from.
- --summary (`b_test.py`): Whether to summary all experiments of cross validation in an Excel file.

The summary Excel file is saved in the `--path_checkpoint/Con4m/` with corresponding `--noise_ratio` folders for convenient check and copy.

## 🌟 Citation
Please leave us a star 🌟 and cite our paper if you find our work helpful.
```
@inproceedings{NEURIPS2024_c6b96bf6,
    author = {Chen, Junru and Cao, Tianyu and Xu, Jing and Li, Jiahe and Chen, Zhilong and Xiao, Tao and Yang, Yang},
    booktitle = {Advances in Neural Information Processing Systems},
    pages = {109980--110009},
    title = {Con4m: Context-aware Consistency Learning Framework for Segmented Time Series Classification},
    volume = {37},
    year = {2024}
}
```
