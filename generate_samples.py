import argparse
import torch
import numpy as np
import soundfile as sf
import os

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, AutoConfig

import pconfig
import prepare_data
import trainer


if __name__=='__main__':
    print("Starting training")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'store', type = str, help = 'Configuration', required=True)
    args = parser.parse_args()

    cfg = pconfig.load_config(args.config)
    print(cfg)

    torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache
    ds = prepare_data.lj_speech_dataset(cfg)
    model = EncoderDecoderModel.from_pretrained(pconfig.model_path(cfg.model.name, None))

    args = Seq2SeqTrainingArguments(
        f"Something",
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        fp16=False, # True,
        push_to_hub=False,
        **cfg.training.args.__dict__,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=trainer.compute_metrics,
    )


    decoder, sampling_rate = prepare_data.make_token_decoder()

    os.makedirs(f"samples/{cfg.model.name}", exist_ok=True)

    res = trainer.predict(ds["validation"], max_length=cfg.model.decoder.seq_length+1)

    for ix, ex in enumerate(ds["validation"]):
        prediction = res.predictions[ix]
        # prediction = [1024] + ex['labels'][:-1] # testing
        audio_values = decoder(prediction)
        f_name = f"samples/{cfg.model.name}/{ex['id']}.wav"
        sf.write(f_name, audio_values, sampling_rate)
