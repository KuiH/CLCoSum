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
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
import evaluate

from utils import (get_filenames, get_elapse_time, load_and_cache_gen_data, is_model_dataparallel, 
                   rouge_from_file, meteor_from_file, get_warmup_steps)
from configs import add_args, set_seed, set_dist
from scheduler_fun import (SCHEDULER_FUN)

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch[0], batch[1]
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_metric_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria, 
                      rouge_calculator, meteor_calculator):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                if is_model_dataparallel(model):
                    preds = model.module(source_ids=source_ids, source_mask=source_mask)
                else:
                    preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]     
            else:
                if is_model_dataparallel(model):
                    preds = model.module.generate(source_ids,
                                        attention_mask=source_mask,
                                        use_cache=True,
                                        num_beams=args.beam_size,
                                        early_stopping=args.task == 'summarize',
                                        max_length=64)
                else:
                    preds = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        use_cache=True,
                                        num_beams=args.beam_size,
                                        early_stopping=args.task == 'summarize',
                                        max_length=64)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    bleu1, bleu2, bleu4 = 0.0, 0.0, 0.0
    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu-1': 0, 'bleu-2': 0, 'bleu-4': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                # for smooth-bleu4 evaluation
                predictions.append(str(gold.idx) + '\t' + pred_nl.replace('\n',' ').strip())
                f.write(str(gold.idx) + '\t' + pred_nl.replace('\n',' ').strip() + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target.replace('\n',' ').strip() + '\n')

        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        bleu1, bleu2, bleu4 = smooth_bleu.bleuFromMaps(goldMap, predictionMap)
        bleu1, bleu2, bleu4 = round(bleu1, 3), round(bleu2, 3), round(bleu4, 3)
        _ , _ , rougeL = rouge_from_file(ref_file_path=gold_fn, predic_file_path=output_fn, rouge_calculator=rouge_calculator)
        rougeL = round(rougeL, 3)
        meteor = round(meteor_from_file(ref_file_path=gold_fn, predic_file_path=output_fn, meteor_calculator=meteor_calculator),3)

        result = {'bleu-1': bleu1, 'bleu-2':bleu2, 
                  'bleu-4': bleu4, 'rougeL': rougeL, 'meteor': meteor}


    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


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
    
    args = add_args(parser)
    check_args(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    logger.info(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    rouge_calculator = evaluate.load(args.rouge_path)
    meteor_calculator = evaluate.load(args.meteor_path)
    if args.n_gpu > 1 and args.do_train:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task, args.lang)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_dataset = [v for v in train_data]
        initial = args.first_epoch_num / len(train_dataset) if args.scheduler_fun != "std" else 1
        train_data = train_dataset[0:int(initial * len(train_dataset))]
        # cl: Curriculum Learning
        train_sampler_cl = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader_cl = DataLoader(train_data, sampler=train_sampler_cl, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # full dataset
        train_sampler_all = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader_all = DataLoader(train_dataset, sampler=train_sampler_all, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader_all)
        warmup_steps = get_warmup_steps(initial, args.warmup_rate, len(train_dataloader_all), args.num_train_epochs, SCHEDULER_FUN[args.scheduler_fun])
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        logger.info(f"  Warmup steps: {warmup_steps}")

        # Start training
        log_steps = (warmup_steps / args.warmup_rate)//args.log_times
        train_example_num = len(train_dataset)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        logger.info(f"  Log steps: {log_steps}")

        current_train_num = len(train_data)
        progress = 0
        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6

        for cur_epoch in range(args.num_train_epochs):
            logger.info(f"  {current_train_num} samples({round(current_train_num/train_example_num*100, 2)}%) used.")
            bar = tqdm(train_dataloader_cl, total=len(train_dataloader_cl), desc="Training")
            nb_tr_steps, tr_loss = 0, 0
            model.train()

            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                
                progress += 1
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                    if global_step % log_steps == 0:
                        tb_writer.add_scalar('train/loss', train_loss, global_step)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('eval/dev_ppl', eval_ppl, cur_epoch)

                if eval_ppl < best_ppl:
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_metric_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch, 
                                               rouge_calculator=rouge_calculator, meteor_calculator=meteor_calculator)
                    dev_bleu, dev_em, dev_rougeL, dev_meteor = result['bleu-4'], result['em'], result["rougeL"], result["meteor"]
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('eval/dev_bleu_em', dev_bleu_em, cur_epoch)
                        tb_writer.add_scalar('eval/rougeL', dev_rougeL, cur_epoch)
                        tb_writer.add_scalar('eval/meteor', dev_meteor, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)

            next_train_ratio = min(1, SCHEDULER_FUN[args.scheduler_fun](cur_epoch+1, args.num_train_epochs-1, initial))
            next_train_num = int(next_train_ratio * len(train_dataset))
            if next_train_num != current_train_num:
                current_train_num = next_train_num
                train_data = train_dataset[0:current_train_num]
                train_sampler_cl = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
                train_dataloader_cl = DataLoader(train_data, sampler=train_sampler_cl, batch_size=args.train_batch_size,
                                            num_workers=4, pin_memory=True)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Test file: {}".format(args.test_filename))

        for criteria in ['best-bleu']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.exists(file):
                logger.info(f"No ckpt for criteria `{criteria}`")
                continue
            logger.info(" Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            test_examples, test_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_metric_epoch(args, test_data, test_examples, model, tokenizer, 'test', criteria,
                                       rouge_calculator=rouge_calculator, meteor_calculator=meteor_calculator)
            test_bleu1, test_bleu2, test_bleu4 = result['bleu-1'], result['bleu-2'], result['bleu-4'] 
            test_rougeL, test_meteor = result["rougeL"], result["meteor"]
            result_str = "[{}] bleu-1: {}, bleu-2: {}, bleu-4: {}, rougeL: {}, meteor: {}\n".format(
                    criteria, test_bleu1, test_bleu2, test_bleu4, test_rougeL, test_meteor
                )
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
