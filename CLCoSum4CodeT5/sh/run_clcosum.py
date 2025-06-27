#!/usr/bin/env python
import os
import argparse

def get_cmd(task, sub_task, model_tag, gpu, data_num, train_bs, eval_bs, lr, source_length, target_length, patience, epoch, warmup, weight_decay,
            model_dir, summary_dir, res_fn, first_epoch_num, scheduler_fun, language, max_steps=None, save_steps=None, log_steps=None):
    if max_steps is None:
        cmd_str = 'bash exp_with_args.sh %s %s %s %s %d %d %d %d %d %d %d %d %d %d %s %s %s %d %s %s' % \
                  (task, sub_task, model_tag, gpu, data_num, train_bs, eval_bs, lr, source_length, target_length, patience, epoch,
                   warmup, weight_decay, model_dir, summary_dir, res_fn, first_epoch_num, scheduler_fun, language)
    else:
        cmd_str = 'bash exp_with_args.sh %s %s %s %s %d %d %d %d %d %d %d %d %d %d %s %s %s %d %d %d %d %s %s' % \
                  (task, sub_task, model_tag, gpu, data_num, train_bs, eval_bs, lr, source_length, target_length, patience, epoch,
                   warmup, weight_decay, model_dir, summary_dir, res_fn, max_steps, save_steps, log_steps, first_epoch_num, scheduler_fun, language)
    return cmd_str


def get_args_by_task_model(args):
    task, sub_task, model_tag, lang = args.task, args.sub_task, args.model_tag, args.lang

    if "tlcodesum" in sub_task or "pcsd" in sub_task:
        src_len, trg_len = 448, 32
    else:
        raise ValueError(f"illegal subtask and language: `{sub_task}` and `{lang}`")


    if 'codet5_small' in model_tag:
        train_bs, eval_bs = 16, 16
    if 'codet5_base' in model_tag:
        train_bs, eval_bs = 12, 12
    elif 'codet5_large' in model_tag:
        train_bs, eval_bs = 2, 8
    else:
        train_bs, eval_bs = 4, 32
    lr = 2
    patience = -1 # Negligible
    return train_bs, eval_bs, lr, src_len, trg_len, patience


def run_one_exp(args):
    train_bs, eval_bs, lr, src_len, trg_len, patience = get_args_by_task_model(args)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpus,
                      data_num=args.data_num, train_bs=train_bs, eval_bs=eval_bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=args.epoch, warmup=1, weight_decay=0,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag),
                      first_epoch_num=args.first_epoch_num, scheduler_fun=args.scheduler_fun,
                      language=args.lang)
    print('%s\n' % cmd_str)
    os.system(cmd_str)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_base')
    parser.add_argument("--task", type=str, default='summarize')
    parser.add_argument("--sub_task", type=str, default='small')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpus", type=str, default="0", help='index of the gpu to use in a cluster')
    parser.add_argument("--scheduler_fun", default="", type=str)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--lang", default="", type=str, choices=['python','java'])
    
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    if args.scheduler_fun != "std":
        if "tlcodesum" in args.sub_task: 
            args.first_epoch_num = 53597
        elif "pcsd" in args.sub_task: 
            args.first_epoch_num = 57849
    else:
        args.first_epoch_num = -1

    run_one_exp(args)
