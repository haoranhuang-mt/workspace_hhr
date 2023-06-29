# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: finetuning_freeze
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/4 17:55
"""
    文件说明：
            
"""
import sys
from modeling_chatglm_for_musa import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
import torch
import torch_musa
import argparse
from torch.utils.data import RandomSampler, DataLoader
from data_set import Seq2SeqDataSet, coll_fn
import os
from shutil import copy

from transformers import get_scheduler, AutoConfig, AutoTokenizer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../ChatGLM-Finetuning/data/spo_test.json', type=str, help='')
    parser.add_argument('--model_dir', default="./", type=str, help='')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='')
    parser.add_argument('--train_batch_size', default=3, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_freeze/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='')
    return parser.parse_args()


def main():
    args = set_args()
    glm_config = AutoConfig.from_pretrained('./', trust_remote_code=True)

    model = ChatGLMForConditionalGeneration(glm_config, empty_init=False)
    print(model)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print('tken done')
    model = model.float().to('musa')
    print('musa init')
    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": args.log_steps
            }

    # for name, param in model.named_parameters():
    #     if not any(nd in name for nd in ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]):
    #         param.requires_grad = False

    # print_trainable_parameters(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)

    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=True,
                                  num_workers=0)

    # model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
    #                                                      model=model,
    #                                                      model_parameters=model.parameters())
    # model_engine.train()

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.001,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0 * args.gradient_accumulation_steps,
        num_training_steps=100 * args.gradient_accumulation_steps,
    )

    global_step = 0
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./glm_prof'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            for step, batch in enumerate(train_iter):
                batch['input_ids'] = batch['input_ids'].to('musa')
                # batch['attention_mask'] = batch['attention_mask'].cuda()
                batch['labels'] = batch['labels'].to('musa')
                outputs = model(**batch)
                loss = outputs.loss
                # total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # outputs = model_engine.forward(input_ids=input_ids, labels=labels)
                # loss = outputs[0]
                if conf["gradient_accumulation_steps"] > 1:
                    loss = loss / conf["gradient_accumulation_steps"]
                # model_engine.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                #     model_engine.step()
                global_step += 1
                if global_step % args.log_steps == 0:
                    print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
                prof.step()
            save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        # model_engine.save_pretrained(save_dir)
        # copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
        # copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=3 deepspeed --master_port 6666 finetuning_freeze.py
