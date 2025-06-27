from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from typing import *

from _utils import *

logger = logging.getLogger(__name__)


class IOTool:
    @staticmethod
    def load_jsonl(file_path)->List:
        json_objects = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                json_objects.append(json.loads(line.strip()))
        return json_objects
    
    @staticmethod
    def dump_jsonl(obj_list:List[object], out_path:str):
        assert isinstance(obj_list, list)
        with open(out_path, 'w', encoding='utf8') as f:
            for obj in obj_list:
                json_str = json.dumps(obj, ensure_ascii=False)
                f.write(json_str + '\n')

    @staticmethod
    def load_json(file_path:str):
        with open(file_path, 'r', encoding='utf8') as f:
            obj = json.load(f)
        return obj

    @staticmethod
    def dump_json(obj:object, out_path:str):
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    

def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task)
    # logger.debug(f"First example:\n{examples[0].source}\n***********\n{examples[0].target}")
    if is_sample:
        examples = random.sample(examples, min(8000, len(examples)))

    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 8k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)

        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else: # train
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        # if args.local_rank in [-1, 0] and not is_sample:
        #     torch.save(data, cache_fn)
    return examples, data



def get_filenames(data_root, task, sub_task, language='', split=''):
    if task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/val.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)

    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))

    examples_num = len(examples)
    if is_tokenize:
        avg_src_len_tokenize = sorted(avg_src_len_tokenize)
        avg_trg_len_tokenize = sorted(avg_trg_len_tokenize)
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    examples_num, np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
        logger.info("[TOKENIZE] src len 90%|95%: {}|{}, trg len 90%|95%: {}|{}".format(
                    avg_src_len_tokenize[int(examples_num*0.9)], avg_src_len_tokenize[int(examples_num*0.95)],
                    avg_trg_len_tokenize[int(examples_num*0.9)], avg_trg_len_tokenize[int(examples_num*0.95)]))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    examples_num, np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def is_model_dataparallel(model):
    return hasattr(model, 'module') and isinstance(model.module, torch.nn.Module)


def _get_meteor(prediction: str, reference: str, meteor=None):
    # if len(prediction) == 0 or len(reference) == 0:
    #     return 0.0
    results = meteor.compute(predictions=[prediction], references=[reference])
    return results["meteor"]*100

def _get_corpus_meteor(predictions: List[str], references: List[str], meteor=None):
    # if len(prediction) == 0 or len(reference) == 0:
    #     return 0.0
    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"]*100


def _get_rouge(prediction: str, reference: str, tokenizer=None, rouge=None):
    # if len(prediction) == 0 or len(reference) == 0:
    #     return 0.0, 0.0, 0.0
    if tokenizer is None:
        results = rouge.compute(predictions=[prediction], references=[reference])
    else:
        results = rouge.compute(predictions=[prediction], references=[reference], tokenizer=tokenizer)
    # print(results)
    return results['rouge1']*100, results['rouge2']*100, results['rougeL']*100


def _get_corpus_rouge(predictions: List[str], references: List[str], tokenizer=None, rouge=None):
    # if len(prediction) == 0 or len(reference) == 0:
    #     return 0.0, 0.0, 0.0
    if tokenizer is None:
        results = rouge.compute(predictions=predictions, references=references)
    else:
        results = rouge.compute(predictions=predictions, references=references, tokenizer=tokenizer)
    # print(results)
    return results['rouge1']*100, results['rouge2']*100, results['rougeL']*100


def meteor_from_file(ref_file_path:str, predic_file_path:str, meteor_calculator):
    with open(ref_file_path) as f1, open(predic_file_path) as f2:
        reference_list, predic_list = f1.readlines(), f2.readlines()
    assert len(reference_list) == len(predic_list)
    for i in range(len(reference_list)):
        reference_list[i] = reference_list[i].strip().split('\t')[1].lower()
        predic_list[i] = predic_list[i].strip().split('\t')[1].lower()
    return _get_corpus_meteor(predictions=predic_list, references=reference_list, meteor=meteor_calculator)


def rouge_from_file(ref_file_path:str, predic_file_path:str, rouge_calculator):
    with open(ref_file_path) as f1, open(predic_file_path) as f2:
        reference_list, predic_list = f1.readlines(), f2.readlines()
    assert len(reference_list) == len(predic_list)
    for i in range(len(reference_list)):
        reference_list[i] = reference_list[i].strip().split('\t')[1].lower()
        predic_list[i] = predic_list[i].strip().split('\t')[1].lower()
    return _get_corpus_rouge(predictions=predic_list, references=reference_list, rouge=rouge_calculator)


def get_warmup_steps(initial_rate:float, warmup_rate:float, all_data_one_epoch_steps:int, train_epochs:int, scheduler_fun):
    all_train_ratio = initial_rate
    for e in range(train_epochs-1):
        next_train_ratio = min(1, scheduler_fun(e+1, train_epochs-1, initial_rate))
        all_train_ratio += next_train_ratio
    warmup_steps = int(all_data_one_epoch_steps * all_train_ratio * warmup_rate)
    return warmup_steps







    



