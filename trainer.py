import argparse
import pconfig
import prepare_data
import torch

from transformers import DataCollatorForSeq2Seq
from transformers import EncoderDecoderModel, AutoTokenizer

import wandb

from dataclasses import dataclass

from model import make_model
from training_utils import compute_metrics, CustomTrainer, CustomTrainingArguments

if __name__=='__main__':
    print("Starting training")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'append', type = str, help = 'Configuration', required=True)
    parser.add_argument('--yconfig', action = 'append', type = str, help = 'Inline yaml config, useful for config overriding')
    parser.add_argument('--with-model', action = 'store', type = str, help = 'Pretrained model')
    parser.add_argument('--with-checkpoint', action = 'store', type = str, help = 'Start from checkpoint')
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

    if args.with_model:
        model = EncoderDecoderModel.from_pretrained(args.with_model)
    elif args.with_checkpoint:
        model = EncoderDecoderModel.from_pretrained(args.with_checkpoint)
    else:
        model = make_model(cfg)

    print("Number of parameters in model:", model.num_parameters())

    targs = CustomTrainingArguments(
        pconfig.model_path(cfg.model.name, None),
        report_to=report_to,
        **cfg.training.args.__dict__,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = CustomTrainer(
        model,
        targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    if not args.with_checkpoint:
        trainer.train()
    else:
        trainer.train(args.with_checkpoint)

    trainer.save_model(pconfig.model_path(cfg.model.name, None))

    print(trainer.evaluate(max_length=cfg.model.decoder.seq_length+1))