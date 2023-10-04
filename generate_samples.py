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
    ds = prepare_data.lj_speech_dataset(cfg.prepare_data)
    model = EncoderDecoderModel.from_pretrained(pconfig.model_path(cfg.model.name, None))

    args = Seq2SeqTrainingArguments(
        f"Something",
        predict_with_generate=True,
        **cfg.training.args.__dict__,
        report_to = "none",
    )

    # Generate validation predictions
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

    print(trainer.evaluate(max_length=cfg.model.decoder.seq_length+1))
    res = trainer.predict(ds["validation"], max_length=cfg.model.decoder.seq_length+1, num_beams=4, do_sample=True)

    # Decode predictions to audio and save
    os.makedirs(f"samples/{cfg.model.name}", exist_ok=True)
    decoder, sampling_rate = prepare_data.make_token_decoder()
    for ix, ex in enumerate(ds["validation"]):
        prediction = res.predictions[ix]
        # prediction = [1024] + ex['labels'][:-1] # testing
        audio_values = decoder(prediction)
        f_name = f"samples/{cfg.model.name}/{ex['id']}.wav"
        sf.write(f_name, audio_values, sampling_rate)
