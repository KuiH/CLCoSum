## CLCoSum: Curriculum Learning-based Code Summarization for Code Language Models

### Requirements

We use `python3.8.8` , and the following packages need to be installed:

```
tree-sitter==0.20.4
pytorch==1.13.0
torchvision==0.14.0
torchaudio==0.13.0
transformers==4.39.3
evaluate==0.4.1
tensorboard==2.13.0
numpy==1.24.3
nltk==3.9.1
rouge-score==0.1.2
```



### Dataset Preparation

- Download the dataset [here](https://drive.google.com/drive/folders/1_1uhRNjLENUDNHtHNO2N03TIr7DvupJR?usp=sharing).
- Place the dataset in the following location:

```
CLCoSum
├─CLCoSum4CodeT5
├─CLCoSum4GraphCodeBERT
├─CLCoSum4UniXCoder
└─dataset
    └─summarize
        ├─pcsd_clean
        ├─pcsd_clean+delchar+tofunc_tokenlen_411
        ├─tlcodesum_clean
        └─tlcodesum_clean+delchar+tofunc_tokenlen_411
```



### Run CLCoSum for CodeT5

- Modify the `WORKDIR` variable on line 1 of `CLCoSum4CodeT5/sh/exp_with_args.sh` to the absolute path of the current `CLCoSum4CodeT5` folder.
- Modify the `DATADIR_ROOT` variable on line 2 of `CLCoSum4CodeT5/sh/exp_with_args.sh` to the absolute path of the current `CLCoSum` folder.
- First run `cd CLCoSum4CodeT5/sh`, then run the following commands based on your needs:

**TLC using CLCoSum: **

```
python run_clcosum.py --model_tag codet5_base --task summarize --sub_task tlcodesum_clean+delchar+tofunc_tokenlen_411 --lang java --epoch 15 --scheduler_fun S5 --gpus=1,2,3
```

**PCSD using CLCoSum: **

```
python run_clcosum.py --model_tag codet5_base --task summarize --sub_task pcsd_clean+delchar+tofunc_tokenlen_411 --lang python --epoch 15 --scheduler_fun S5 --gpus=1,2,3
```

**Note:**

- You can modify the `gpus` parameter to change the GPUs used for training.
- If using a local model, modify the `TOKENIZER` and `MODEL_PATH` variables on lines 54 and 55 of `exp_with_args.sh` to the local CodeT5 model path.



### Run CLCoSum for GraphCodeBERT

- Modify the `WORKDIR` variable on line 1 of `CLCoSum4GraphCodeBERT/run_clcosum.sh` to the absolute path of the current `CLCoSum4GraphCodeBERT` folder.
- Modify the variables `SUB_TASK`, `SCHEDULER_FUN`, and `LANG` on lines 5 to 7 of `CLCoSum4GraphCodeBERT/run_clcosum.sh` based on your needs:

**TLC using CLCoSum: **

```shell
SUB_TASK="tlcodesum_clean+delchar+tofunc_tokenlen_411"
SCHEDULER_FUN="S5"
LANG="java"
```

**PCSD using CLCoSum: **

```shell
SUB_TASK="pcsd_clean+delchar+tofunc_tokenlen_411"
SCHEDULER_FUN="S5"
LANG="python"
```

**Note:**

- The `GPU` parameter on line 9 can be modified to specify the GPU(s) to use for multi-GPU training.
- If using a local model, modify the `TOKENIZER` and `MODEL_PATH` on lines 58 and 59 to the local GraphCodeBERT model path.



Finally, run the following commands:

```
cd CLCoSum4GraphCodeBERT
bash run_clcosum.sh
```



### Run CLCoSum for UniXCoder

Modify the variables `SUB_TASK`, `SCHEDULER_FUN`, and `LANG` on lines 2 to 4 of `CLCoSum4UniXCoder/run_clcosum.sh` based on your needs:

**TLC using CLCoSum: **

```shell
SUB_TASK="tlcodesum_clean+delchar+tofunc_tokenlen_411"
SCHEDULER_FUN="S5"
LANG="java"
```

**PCSD using CLCoSum: **

```shell
SUB_TASK="pcsd_clean+delchar+tofunc_tokenlen_411"
SCHEDULER_FUN="S5"
LANG="python"
```

**Note:**

- The `GPU` parameter on line 4 can be modified to specify the GPU(s) to use for multi-GPU training.
- If using a local model, modify the `MODEL_PATH` on line 55 to the local UniXCoder model path.



Finally, run the following commands:

```
cd CLCoSum4UniXCoder
bash run_clcosum.sh
```











