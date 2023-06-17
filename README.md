# PALM: Pre-trained Antibody generative large Language Model
![image](https://github.com/TencentAILabHealthcare/PALM/blob/main/1.png)
## Hardware requirements
`PALM` package requires only a standard computer with enough RAM and a NVIDIA GPU to support operations.
We ran the demo using the following specs:

+ CPU: 10 cores, 2.5 GHz/core
+ RAM: 40 GB
+ GPU: NVIDIA TESLA V100
## System requirements
This tool is supported for Linux. The tool has been tested on the following system:

+ CentOS Linux release 8.2.2.2004

## Installation
To install the required packages for running PALM, please use the following command:
```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
pip install -r requirements.txt
```
### Time cost
Typical install time on a "normal" desktop computer is about 30 minutes.

## How to train and use PALM
The training of PALM and A2binder consists of three steps: first, we pre-train two language models on unpaired antibody heavy and light chain sequences, respectively. Then we construct A2binder, and fine-tune it using paired affinity data. Finally, we construct PALM by Roformer and ESM2 using paired data for designing and evaluating the AI-generated CDRH3. The details of each training are in the `Code/config` folder. Note that all the commands are run in the `Code` folder.


### 1. Pre-train on unpaired sequences
The MAA task is used for the self-training of HeavyRoformer and LightRoformer. 

Due to the space limitation, we present demo unpaired data in the folder `ProcessedData`.

The training command for HeavyRoformer is:
```bash
python bert_pretrain_maa_main.py --config ./config/common/bert_pretrain_maa_common_heavy_covid.json
```
The training command for LightRoformer is:
```bash
python bert_pretrain_maa_main.py --config ./config/common/bert_pretrain_maa_common_light_covid.json
```
After the training, the pre-trained HeavyRoformer and LightRoformer will be saved in the `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX` and `../Result_covid_light/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX` folder, where `XXXX_XXXXXX` is the timestamp of the training.

### Time cost
Expected run time for demo on a "normal" desktop computer is about 2 hours.

### 2. Training A2binder on paired affinity datasets

Before running the affinity predicition task, please copy the **absolute path** of pre-trained HeavyRoformer (`../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`) and LightRoformer (`../Result_covid_light/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`) to replace the corresponding file path in the config file `bert_finetuning_er_common_Cov_abdab.json`. In detail: please replace the "heavy_dir" using `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX` ; replace the "light_dir"  using `../Result_covid_light/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. Besides, you should also replace the "antibody_tokenizer_dir" to the path `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. The above path needs to be an absolute path.

The training command for A2binder is:
```bash
python bert_finetuning_er_main.py --config ./config/common/bert_finetuning_er_common_Cov_abdab.json
```
After the training, the trained A2binder will be saved in the `../Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/XXXX_XXXXXX` folder.
Due to the use of ESM2 model parameters as the antigen model, there may be network errors when downloading ESM2  model parameters. Please check the network settings or try again later.

### 3. Training PALM on seq2seq task
Before running the seq2seq task, please copy the **absolute path** of A2binder (`../Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/XXXX_XXXXXX/heavymodel`) and (`../Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/XXXX_XXXXXX/antigenmodel`) to replace the corresponding file path in the config file `bert_finetuning_er_seq2seq_common.json`. In detail: please replace the "AntibodyBert_dir" using `../Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/XXXX_XXXXXX/heavymodel` ; replace the "AntigenBert_dir"  using `../Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/XXXX_XXXXXX/antigenmodel`. Besides, you should also replace the "antibody_tokenizer_dir" and "antigen_tokenizer_dir" to `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. The above path needs to be an absolute path.

The training command for PALM is:
```bash
python bert_finetuning_seq2seq_main.py --config ./config/common/bert_finetuning_er_seq2seq_common.json
```
After the training, the trained PALM will be saved in the `../Result_seq2seq/checkpoints/ABAG-Finetuning-Seq2seq-Common/XXXX_XXXXXX/` folder.

### 4. Generate artificial antibodies
Before running the generation task, please copy the **absolute path** of PLAM `../Result_seq2seq/checkpoints/ABAG-Finetuning-Seq2seq-Common/XXXX_XXXXXX/` to "resume", copy the absolute path of `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`to "antibody_tokenizer_dir" and "antigen_tokenizer_dir". And set "origin_seq", "origin_light", "cdrh3_begin", "cdrh3_end" and "use_antigen" in the config file `seq2seq_generate.json`. The above path needs to be an absolute path.

The generation command for PALM is:
```bash
python generate_antibody.py --config ./config/common/seq2seq_generate.json
```
### Expected output
After the running, the artificial antibody will be saved in the `../Result_seq2seq_gen/datasplit/CoV_AbDab-Seq2seq-Evaluate-Common/XXXX_XXXXXX/result.csv`.

### 5. Evaluate artificial antibodies
After generating antibodies, A2binder can be used to evaluate the affinity probability or affinity of the generated antibodies. Before evaluating, please copy the **absolute path** of A2binder `../Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/XXXX_XXXXXX` to "discriminator_resume", replace the "heavy_dir" using `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`, replace the "light_dir"  using `../Result_covid_light/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`, and replace "antibody_tokenizer_dir" to `../Result_covid_heavy/checkpoints/BERT-Beta-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`  in the `bert_eval_generation.json` and change the "data_dir" to `../Result_seq2seq_gen/datasplit/CoV_AbDab-Seq2seq-Evaluate-Common/XXXX_XXXXXX/result.csv`. The above path needs to be an absolute path.

The evalation command for PALM is:
```bash
python eval_generate_seq.py --config ./config/common/bert_eval_generation.json
```
### Expected output
After the running, the evalation result will be saved in the `../Result_eval/datasplit/Eval-genetation/XXXX_XXXXXX/test_result.csv` 

## Model availability
PALM and A2binder on all the three tasks (Pre-training, Affinity predicition, and Seq2Seq) on the comprehensive training dataset are available on Zenodo: https://doi.org/10.5281/zenodo.7794583. And you can fine-tuning it on your own dataset and downstream tasks.
## Data availability
Due to the space limitation, we present part of data used in this project in the folder `ProcessedData`. Full pre-training data are available from https://opig.stats.ox.ac.uk/webapps/oas/.
## Contact
If you have any questions, please contact us via email: 
- [Haohuai He](mailto:hehh8@mail2.sysu.edu.cn)
- [Bing He](mailto:hebinghb@gmail.com)
