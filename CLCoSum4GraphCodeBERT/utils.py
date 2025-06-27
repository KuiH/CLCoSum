from typing import *
import time
import json
import smooth_bleu


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def get_warmup_steps(initial_rate:float, warmup_rate:float, all_data_one_epoch_steps:int, train_epochs:int, scheduler_fun):
    all_train_ratio = initial_rate
    for e in range(train_epochs-1):
        next_train_ratio = min(1, scheduler_fun(e+1, train_epochs-1, initial_rate))
        all_train_ratio += next_train_ratio
    warmup_steps = int(all_data_one_epoch_steps * all_train_ratio * warmup_rate)
    return warmup_steps


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


def metric_from_file(gold_fn:str, output_fn:str, rouge_calculator, meteor_calculator):
    predictions = []
    with open(output_fn) as f:
        for line in f:
            predictions.append(line.strip())
    (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
    bleu1, bleu2, bleu4 = smooth_bleu.bleuFromMaps(goldMap, predictionMap)
    bleu1, bleu2, bleu4 = round(bleu1, 3), round(bleu2, 3), round(bleu4, 3)
    _ , _ , rougeL = rouge_from_file(ref_file_path=gold_fn, predic_file_path=output_fn, rouge_calculator=rouge_calculator)
    rougeL = round(rougeL, 3)
    meteor = round(meteor_from_file(ref_file_path=gold_fn, predic_file_path=output_fn, meteor_calculator=meteor_calculator),3)
    result = {'bleu-1': bleu1, 'bleu-2':bleu2, 'bleu-4': bleu4, 'rougeL': rougeL, 'meteor': meteor}
    return result


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
