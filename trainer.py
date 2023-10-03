import argparse
import pconfig
import prepare_data
import torch
import numpy as np

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, AutoConfig

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    acc = np.mean(np.sum((preds==labels) & (labels!=-100), axis=-1) / np.sum(labels != -100, axis=-1))
    return {"acc": acc}

if __name__=='__main__':
    print("Starting training")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'store', type = str, help = 'Configuration', required=True)
    args = parser.parse_args()

    cfg = pconfig.load_config(args.config)
    print(cfg)

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
        vocab_size=prepare_data.TOK_TOKS,
        n_ctx=302,
        bos_token_id=prepare_data.TOK_BOS,
        eos_token_id=prepare_data.TOK_EOS,
        **cfg.model.decoder.__dict__,
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = prepare_data.TOK_BOS
    model.config.pad_token_id = prepare_data.TOK_EOS

    torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache
    ds = prepare_data.lj_speech_dataset(cfg)

    args = Seq2SeqTrainingArguments(
        f"Something",
        predict_with_generate=True,
        **cfg.training.args.__dict__,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(trainer.evaluate(max_length=cfg.model.decoder.seq_length+1))

    trainer.train()
    trainer.save_model(pconfig.model_path(cfg.model.name, None))

    print(trainer.evaluate(max_length=cfg.model.decoder.seq_length+1))