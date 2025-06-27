# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import time
import math
import torch
import json
import random
import logging
import argparse
import numpy as np
import evaluate
from model import Seq2Seq, get_model_size
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import (AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)

from scheduler_fun import (SCHEDULER_FUN)
from utils import get_elapse_time, metric_from_file, get_warmup_steps

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples = []

    with open(filename) as f1:
        for line in f1:
            example = json.loads(line.strip())
            examples.append(
                Example(
                    idx=example["index"],
                    source=example["src"],
                    target=example["trg"],
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids     
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-5]
        source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
   
        if example_index < 2:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
            )
        )
    return features



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(args.seed)


def check_args(args):
    if args.do_train:
        assert args.scheduler_fun in SCHEDULER_FUN

        if args.scheduler_fun != "std":
            assert args.first_epoch_num > 0
    
        if args.scheduler_fun == "std":
            assert args.first_epoch_num == -1

    task, sub_task = args.task, args.sub_task
    if "tlcodesum" in sub_task:
        assert args.lang == "java"

    if "pscd" in sub_task:
        assert args.lang == "python"

    assert task == "summarize"
    assert sub_task in [
                        'tlcodesum_clean','tlcodesum_clean+delchar+tofunc_tokenlen_411',
                        'pcsd_clean', 'pcsd_clean+delchar+tofunc_tokenlen_411',
                        ]
    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", default=None, type=str, required=True)
    parser.add_argument("--sub_task", default=None, type=str, required=True)   
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=5, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_rate", default=0.1, type=float)
    parser.add_argument("--log_times", default=250, type=int,
                        help="Number of train loss logging times.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--summary_dir", type=str, default='tensorboard', 
                        help='directory to save tensorboard summary')
    parser.add_argument("--res_dir", type=str, default='results', 
                        help='directory to save fine-tuning results')
    parser.add_argument("--res_fn", type=str, default='',
                        help='file path to save model metric results')
    parser.add_argument("--rouge_path", default="rouge", type=str,
                        help="Path to rouge metric.")
    parser.add_argument("--meteor_path", default="meteor", type=str,
                        help="Path to meteor metric")
    # parser.add_argument("--rouge_path", default="/home/usr/evaluate/metrics/rouge", type=str,
    #                     help="Path to rouge metric")
    # parser.add_argument("--meteor_path", default="/home/usr/evaluate/metrics/meteor", type=str,
    #                     help="Path to meteor metric")


    parser.add_argument("--lang", default="", type=str, choices=['python','java'])

    ## Curriculum learning parameters
    parser.add_argument("--scheduler_fun", default="", type=str)
    parser.add_argument("--first_epoch_num", type=int, default=-1, help='number used in first epoch')
    
    # print arguments
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args)
    check_args(args)
    t0 = time.time()

    # make dir if output_dir or res_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config) 

    model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
    
    logger.info("Training/evaluation parameters %s", args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)
    model.to(device)   
    rouge_calculator = evaluate.load(args.rouge_path)
    meteor_calculator = evaluate.load(args.meteor_path)

    if args.n_gpu > 1 and args.do_train:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0]:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long) 
        train_data = TensorDataset(all_source_ids,all_target_ids)
        train_dataset = [v for v in train_data]
        initial = args.first_epoch_num / len(train_dataset) if args.scheduler_fun != "std" else 1
        train_data = train_dataset[0:int(initial * len(train_dataset))]
        # cl: Curriculum Learning
        train_sampler_cl = RandomSampler(train_data)
        train_dataloader_cl = DataLoader(train_data, sampler=train_sampler_cl, batch_size=args.train_batch_size,
                                         num_workers=4)
        # full dataset
        train_sampler_all = RandomSampler(train_dataset)
        train_dataloader_all = DataLoader(train_dataset, sampler=train_sampler_all, batch_size=args.train_batch_size,
                                           num_workers=4)
        num_train_optimization_steps =  args.num_train_epochs * len(train_dataloader_all)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        warmup_steps = get_warmup_steps(initial, args.warmup_rate, len(train_dataloader_all), args.num_train_epochs, SCHEDULER_FUN[args.scheduler_fun])
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        #Start training
        log_steps = (warmup_steps / args.warmup_rate)//args.log_times
        train_example_num = len(train_dataset)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs) 
        logger.info(f"  Log steps: {log_steps}")     
        current_train_begin_num, current_train_end_num = 0, len(train_data)
        current_train_num = current_train_end_num - current_train_begin_num
        model.train()

        dev_dataset={}
        global_step, best_bleu, best_loss = 0, 0, 1e6 
        for epoch in range(args.num_train_epochs):
            logger.info(f"  {current_train_num} samples({round(current_train_num/train_example_num*100, 2)}%) used.")
            logger.info(f"  being at sample {current_train_begin_num}, end at sample {current_train_end_num}")
            
            bar = tqdm(train_dataloader_cl,total=len(train_dataloader_cl), desc="Training")
            nb_tr_steps, tr_loss = 0, 0
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, target_ids = batch
                loss,_,_ = model(source_ids=source_ids,target_ids=target_ids)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_loss += loss.item()
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                    if global_step % log_steps == 0:
                        tb_writer.add_scalar('train/loss', train_loss, global_step)
                    bar.set_description("[{}] Train loss {}".format(epoch, round(train_loss, 3)))


            if args.do_eval:
                #Eval model with dev dataset                   
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)   
                    eval_data = TensorDataset(all_source_ids,all_target_ids)   
                    dev_dataset['dev_loss' ]= eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                             num_workers=4, pin_memory=True)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,target_ids = batch                  

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,target_ids=target_ids)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'epoch': epoch, 'global_step': global_step, 'eval_ppl': round(np.exp(eval_loss),5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                tb_writer.add_scalar('eval/dev_ppl', result["eval_ppl"], epoch)

                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    fa.write("[%d] Best loss changed into %.4f\n" % (epoch, eval_loss))
                    best_loss=eval_loss

                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(8000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long) 
                    eval_data = TensorDataset(all_source_ids)   
                    dev_dataset['dev_bleu'] = eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for eval set"):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]                  
                    with torch.no_grad():
                        preds = model(source_ids) 
                        # convert ids to text
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()

                output_fn = os.path.join(args.res_dir, "dev_e{}.output".format(epoch))
                gold_fn = os.path.join(args.res_dir, "dev_e{}.gold".format(epoch))
                with open(output_fn,'w') as f, open(gold_fn,'w') as f1:
                    for pred_nl, gold in zip(p, eval_examples):
                        f.write(str(gold.idx) + '\t' + pred_nl.replace('\n',' ').strip() + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target.replace('\n',' ').strip() + '\n')

                result = metric_from_file(gold_fn, output_fn, rouge_calculator, meteor_calculator)
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(round(result[key], 4)))
                logger.info("  "+"*"*20)
                dev_bleu, dev_rougeL, dev_meteor = result['bleu-4'], result["rougeL"], result["meteor"]
                tb_writer.add_scalar('eval/dev_bleu', dev_bleu, epoch)
                tb_writer.add_scalar('eval/rougeL', dev_rougeL, epoch)
                tb_writer.add_scalar('eval/meteor', dev_meteor, epoch)
                if dev_bleu > best_bleu:
                    logger.info("  [%d] Best bleu: %.2f", epoch, dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    fa.write("[%d] Best bleu+em changed into %.2f\n" % (epoch, best_bleu))
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best bleu model into %s", output_model_file)

            next_train_begin_ratio, next_train_end_ratio = SCHEDULER_FUN[args.scheduler_fun](epoch+1, args.num_train_epochs-1, initial)
            next_train_begin_ratio, next_train_end_ratio = min(1, next_train_begin_ratio), min(1, next_train_end_ratio)
            # logger.info(f"next_train_begin_ratio: {next_train_begin_ratio} | next_train_end_ratio: {next_train_end_ratio}")
            begin_num, end_num = int(next_train_begin_ratio * len(train_dataset)), int(next_train_end_ratio * len(train_dataset))
            if begin_num != current_train_begin_num or end_num != current_train_end_num:
                current_train_begin_num, current_train_end_num = begin_num, end_num
                current_train_num = current_train_end_num - current_train_begin_num
                train_data = train_dataset[begin_num : end_num]
                train_sampler_cl = RandomSampler(train_data)
                train_dataloader_cl = DataLoader(train_data, sampler=train_sampler_cl, batch_size=args.train_batch_size, num_workers=4)

        tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))  

    if args.do_test:
        test_data_file = args.test_filename
        logger.info("  ***** Running bleu evaluation on test data*****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Test file: {}".format(test_data_file))         

        test_examples = read_examples(test_data_file)
        test_features = convert_examples_to_features(test_examples, tokenizer, args,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_source_ids)   
        logger.info("  Num examples = %d", len(test_examples))

        # Calculate bleu
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)

        for criteria in ['best-bleu']:
            model_file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.exists(model_file):
                logger.info(f"No ckpt for criteria `{criteria}`")
                continue
            logger.info(" Reload model from {}".format(model_file))
            model.load_state_dict(torch.load(model_file))
            model.eval() 
            p=[]
            for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Eval bleu for test set"):
                batch = tuple(t.to(device) for t in batch)
                source_ids = batch[0]                  
                with torch.no_grad():
                    preds = model(source_ids)   
                    # convert ids to text
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text.replace('\n',' '))
                        
            output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
            gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
            with open(output_fn,'w') as f, open(gold_fn,'w') as f1:
                for pred_nl, gold in zip(p, test_examples):
                    f.write(str(gold.idx) + '\t' + pred_nl.replace('\n',' ').strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.replace('\n',' ').strip() + '\n')

            result = metric_from_file(gold_fn, output_fn, rouge_calculator, meteor_calculator)
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            result_str = "[{}] bleu-1: {}, bleu-2: {}, bleu-4: {}, rougeL: {}, meteor: {}\n".format(
                    criteria, result["bleu-1"], result["bleu-2"], result["bleu-4"], result["rougeL"], result["meteor"]
                )
            fa.write(result_str)
            with open(args.res_fn, 'a+') as f:
                f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), model_file))
                f.write(result_str)  

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()

if __name__ == "__main__":
    main()


