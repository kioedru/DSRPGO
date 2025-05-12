# **Enhancing Multimodal Protein Function Prediction through Dual-Branch Dynamic Selection with Reconstructive Pre-training**

Multimodal protein features play a crucial role in protein function prediction. However, these features encompass a wide range of information, ranging from structural data and sequence features to protein attributes and interaction networks, making it challenging to decipher their complex interconnections. In this work, we propose a multimodal protein function prediction method (DSRPGO) by utilizing dynamic selection and reconstructive pre-training mechanisms. To acquire complex protein information, we introduce reconstructive pre-training to mine more fine-grained information with low semantic levels. Moreover, we put forward the Bidirectional Interaction Module (BInM) to facilitate interactive learning among multimodal features. Additionally, to address the difficulty of hierarchical multi-label classification in this task, a Dynamic Selection Module (DSM) is designed to select the feature representation that is most conducive to current protein function prediction. Our proposed DSRPGO model significantly enhances Fmax by at least 4.3%, 7.6%, and 23.5% for BPO, MFO, and CCO on human datasets, thereby outperforming other benchmark models. The public code can be found in the supplementary material.

![main](https://raw.githubusercontent.com/kioedru/typora/master/img/main.jpeg)

DSRPGO is a two-step protein function annotation prediction model, which is jointly constructed by pretrain and finetune phase. The superior performance of DSRPGO is demonstrated with comparative benchmarks.

This repository contains script which were used to build and train the DSRPGO model.



## Installation

### Dependencies
* The code was developed and tested using python 3.8.18.
* All the dependencies are listed in `requirements.txt` file.

### Install from github
The source code for DSRPGO can be obtained from the official GitHub repository.

To clone the repository, use the following command:

```bash
# clone project
git clone https://github.com/kioedru/DSRPGO
# install dependencies
conda install -r requirements.txt
```

Alternatively, you can manually install the necessary dependencies, such as `torch` via `pip`.
```bash
pip install torch
```



## Explanation of files and folders

### Folders
| Name                        | Description                                                  | File Type |
| :-------------------------- | :----------------------------------------------------------- | --------: |
| `data`                      | Preprocessed datasets for pretraining (`/pretrain`) and fine-tuning (`/finetune`) |    `.pkl` |
| `finetune/DSRPGO`           | Core code of DSRPGO                                          |    `.pkl` |
| `finetune/MSLB`             | Core code of MSLB                                            |     `.py` |
| `mamba`                     | Code package for bimamba                                     |     `.py` |
| `model`                     | Core implementation of models and loss functions             |     `.py` |
| `pretrain/bimamba`          | Core code of PSSI Encoder                                    |     `.py` |
| `pretrain/one_feature_only` | Core code of PSI Encoder                                     |     `.py` |
| `utils`                     | Utility code such as data loading and metric calculation     |     `.py` |

## Command Line Usage

For preprocessing data related to PPI, subcellular location, and domain information, please refer to [CFAGO](http://bliulab.net/CFAGO/).

For preprocessing protein sequence data, please refer to [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50).

### Pretraining

- To train the **PSeI Encoder**, run:

```bash
python codespace/pretrain/one_feature_only/pretrain.py
```

- To train the **PSI Encoder**, run:

```
python codespace/pretrain/bimamba/pretrain.py
```

### Finetune

**Note:** You can choose to save the best-performing model during training.

#### Step1: Train MSLB Branch

- train **MSLB Branch Model** to predict BPO/MFO/CCO terms run sh:

```bash
python codespace/finetune/MSLB/finetune.py --aspect P --num_class 45
python codespace/finetune/MSLB/finetune.py --aspect F --num_class 38
python codespace/finetune/MSLB/finetune.py --aspect C --num_class 35
```
- train **DSRPGO Model** to predict BPO/MFO/CCO terms run sh:
```bash
python codespace/finetune/DSRPGO/finetune.py --aspect P --num_class 45
python codespace/finetune/DSRPGO/finetune.py --aspect F --num_class 38
python codespace/finetune/DSRPGO/finetune.py --aspect C --num_class 35
```
During model training, test results for each epoch (including epoch number, runtime, Fmax, AUPR, etc.) are logged in the corresponding `.csv` files.



## Citation

If you use DSRPGO for your research, or incorporate our learning algorithms in your work, please cite:

```

```

