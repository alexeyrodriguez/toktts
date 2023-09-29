import argparse
import pconfig
import prepare_data
from datasets import load_dataset
import torch
import numpy as np

if __name__=='__main__':
    print("Starting training")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'store', type = str, help = 'Configuration', required=True)
    args = parser.parse_args()

    cfg = pconfig.load_config(args.config)
    print(cfg)

    from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, AutoConfig

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
        vocab_size=1024,
        n_ctx=302,
        bos_token_id=prepare_data.TOK_BOS,
        eos_token_id=prepare_data.TOK_EOS,
        **cfg.model.decoder.__dict__,
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = prepare_data.TOK_BOS
    model.config.pad_token_id = prepare_data.TOK_EOS

    ds = load_dataset("lj_speech", split="train")
    torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache
    ds = prepare_data.tokenize_speech(cfg.prepare_data, ds)
    ds = ds.train_test_split(train_size=0.9, seed=20)
    ds["validation"] = ds.pop("test")

    from transformers import Seq2SeqTrainingArguments

    args = Seq2SeqTrainingArguments(
        f"Something",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # eval_accumulation_steps=4, # Ok only to move from GPU to CPU
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=False, # True,
        push_to_hub=False,
    )

    from transformers import Seq2SeqTrainer
    from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # print(preds.shape, labels.shape) # (80, 512)
        # print(preds[:4, :10], '\n')
        # print(labels[:4, :10])
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        # print((preds==labels)[:8, :20], '\n')
        # print((labels != -100)[:8, :20], '\n')
        # print(labels[0, :100])
        # print(np.sum(preds==labels, axis=-1), np.sum(labels != -100, axis=-1))
        acc = np.mean(np.sum(preds==labels, axis=-1) / np.sum(labels != -100, axis=-1))

        return {"acc": acc}


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(trainer.evaluate(max_length=300))

    trainer.train()

    print(trainer.evaluate(max_length=300))