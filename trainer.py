import argparse
import pconfig
import prepare_data
import torch
import numpy as np

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, AutoConfig

import wandb

from dataclasses import dataclass

from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from transformers.trainer_utils import SchedulerType

import math

# Customization of learning rate schedules:
@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    min_learning_rate: float = float(0)
    lr_decay_steps: int = None

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, lr_decay_steps: int, min_factor: float=0):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step > lr_decay_steps:
        return min_factor
    return max(0.0, min_factor + (1 - min_factor) * float(lr_decay_steps - current_step) / float(max(1, lr_decay_steps - num_warmup_steps)))

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, lr_decay_steps: int, min_factor: float=0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step > lr_decay_steps:
        return min_factor
    progress = float(current_step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
    return max(0.0, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

def get_schedule_with_warmup(optimizer, name, num_warmup_steps, lr_decay_steps, min_factor, last_epoch=-1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda if name==SchedulerType.COSINE else _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_factor=min_factor,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class CustomTrainer(Seq2SeqTrainer):
    def log(self, logs):
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        name = SchedulerType(self.args.lr_scheduler_type)

        if not name in (SchedulerType.LINEAR, SchedulerType.COSINE):
            return super().create_scheduler(num_training_steps, optimizer)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                name=name,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                lr_decay_steps=num_training_steps if not args.lr_decay_steps else args.lr_decay_steps,
                min_factor=args.min_learning_rate/args.learning_rate,
            )
        self._created_lr_scheduler = True
        return self.lr_scheduler

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    acc = np.mean(np.sum((preds==labels) & (labels!=-100), axis=-1) / np.sum(labels != -100, axis=-1))
    return {"acc": acc}

def make_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    enc_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=100,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **cfg.model.encoder.__dict__,
    )
    dec_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=prepare_data.TOK_TOKS+2,
        n_ctx=302,
        bos_token_id=prepare_data.TOK_BOS,
        eos_token_id=prepare_data.TOK_EOS,
        **cfg.model.decoder.__dict__,
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = prepare_data.TOK_BOS
    model.config.pad_token_id = prepare_data.TOK_EOS
    return model

if __name__=='__main__':
    print("Starting training")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'append', type = str, help = 'Configuration', required=True)
    parser.add_argument('--yconfig', action = 'append', type = str, help = 'Inline yaml config, useful for config overriding')
    parser.add_argument('--with-model', action = 'store', type = str, help = 'Pretrained model')
    args = parser.parse_args()
    cfg = pconfig.load_config(args.config, args.yconfig)

    if hasattr(cfg, "wandb"):
        wandb.login(key=cfg.wandb.api_key)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update({"pconfig": pconfig.namespace_to_dict(cfg)})
        report_to = "wandb"
    else:
        report_to = "none"

    torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache
    ds = prepare_data.lj_speech_dataset(cfg.prepare_data)

    if not args.with_model:
        model = make_model(cfg)
    else:
        model = EncoderDecoderModel.from_pretrained(args.with_model)
    print("Number of parameters in model:", model.num_parameters())

    args = CustomTrainingArguments(
        pconfig.model_path(cfg.model.name, None),
        report_to=report_to,
        # predict_with_generate=True,
        **cfg.training.args.__dict__,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(pconfig.model_path(cfg.model.name, None))

    print(trainer.evaluate(max_length=cfg.model.decoder.seq_length+1))