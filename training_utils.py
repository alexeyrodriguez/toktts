import torch
import numpy as np

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

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

class CustomTrainer(Seq2SeqTrainer):
    # def log(self, logs):
    #     logs["learning_rate"] = self._get_learning_rate()
    #     super().log(logs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        name = SchedulerType(self.args.lr_scheduler_type)

        if not name in (SchedulerType.LINEAR, SchedulerType.COSINE):
            return super().create_scheduler(num_training_steps, optimizer)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                name=name,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                lr_decay_steps=num_training_steps if not self.args.lr_decay_steps else self.args.lr_decay_steps,
                min_factor=self.args.min_learning_rate/self.args.learning_rate,
            )
        self._created_lr_scheduler = True
        return self.lr_scheduler

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

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    acc = np.mean(np.sum((preds==labels) & (labels!=-100), axis=-1) / np.sum(labels != -100, axis=-1))
    return {"acc": acc}
