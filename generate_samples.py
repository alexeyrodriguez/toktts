import argparse
import torch
import numpy as np
import soundfile as sf
import os

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, AutoConfig
from datasets import Dataset

import pconfig
import prepare_data
import trainer

def cut_rows(ds, n_rows):
    n = min(len(ds), n_rows)
    return ds.select(range(n))

def generate_samples(trainer, cfg, ds, subdir="validation"):
    res = trainer.predict(ds, max_length=cfg.model.decoder.seq_length+1, num_beams=4, do_sample=True)

    # Decode predictions to audio and save
    os.makedirs(f"samples/{cfg.model.name}/{subdir}", exist_ok=True)
    decoder, sampling_rate = prepare_data.make_token_decoder()
    for ix, ex in enumerate(ds):
        f_name = f"samples/{cfg.model.name}/{subdir}/{ex['id']}"
        with open(f_name + ".txt", "w") as f:
            f.write(ex["normalized_text"])

        prediction = list(res.predictions[ix])
        audio_values = decoder(prediction)
        sf.write(f_name + ".wav", audio_values, sampling_rate)

        prediction = [1024] + ex['labels'][:-1] # testing
        audio_values = decoder(prediction)
        sf.write(f_name + "_ref.wav", audio_values, sampling_rate)

if __name__=='__main__':
    print("Starting sample generation")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'append', type = str, help = 'Configuration', required=True)
    parser.add_argument('--text', action = 'store', type = str, help = 'Text to speak')
    parser.add_argument('--yconfig', action = 'append', type = str, help = 'Inline yaml config, useful for config overriding')
    args = parser.parse_args()

    cfg = pconfig.load_config(args.config, args.yconfig)

    model = EncoderDecoderModel.from_pretrained(pconfig.model_path(cfg.model.name, None))
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    if not args.text:
        torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache
        ds = prepare_data.lj_speech_dataset(cfg.prepare_data)

        args = Seq2SeqTrainingArguments(
            f"Something",
            predict_with_generate=True,
            **cfg.training.args.__dict__,
            report_to = "none",
        )

        # Generate validation predictions
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=data_collator,
            compute_metrics=trainer.compute_metrics,
        )


        sds = cut_rows(ds["train"], cfg.generate_samples.limit_rows)
        print("Train metrics", trainer.evaluate(sds, max_length=cfg.model.decoder.seq_length+1))
        generate_samples(trainer, cfg, sds, "train")

        sds = cut_rows(ds["validation"], cfg.generate_samples.limit_rows)
        print("Validation metrics", trainer.evaluate(sds, max_length=cfg.model.decoder.seq_length+1))
        generate_samples(trainer, cfg, sds, "validation")

    else:
        # todo: Refactor the below into a function that goes into prepare_data
        #   also the text normalization should happen in prepare_data rather than used from the dataset

        id = "".join([c for c in args.text if c.isalpha()][:20])
        id = "test" if not id else id
        ds = Dataset.from_dict({"id": [id], "normalized_text": [args.text]})

        def add_toks(example):
            return tokenizer(example["normalized_text"])

        ds = ds.map(add_toks)


        args = Seq2SeqTrainingArguments(
            f"Something",
            predict_with_generate=True,
            **cfg.training.args.__dict__,
            report_to = "none",
        )

        # Generate validation predictions
        trainer = Seq2SeqTrainer(
            model,
            args,
            data_collator=data_collator,
        )

        generate_samples(trainer, cfg, ds, "custom")
