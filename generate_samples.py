import argparse
import torch
import soundfile as sf
import os

from datasets import Dataset

import pconfig
import prepare_data
import audio_encoding
from dataset_utils import cut_rows

def generate_samples(pipeline, cfg, ds, subdir="validation"):
    forward_params = dict(max_length=cfg.model.decoder.seq_length+1, num_beams=4, do_sample=True)

    # Decode predictions to audio and save
    os.makedirs(f"samples/{cfg.model.name}/{subdir}", exist_ok=True)
    for ex in ds:
        f_name = f"samples/{cfg.model.name}/{subdir}/{ex['id']}"
        with open(f_name + ".txt", "w") as f:
            f.write(ex["normalized_text"])

        audio_values = pipeline(ex["normalized_text"], forward_params=forward_params)
        sf.write(f_name + ".wav", audio_values["audio"], pipeline.sampling_rate)

        if 'labels' in ex:
            prediction = [1024] + ex['labels'][:-1] # testing
            audio_values = pipeline.vocoder([prediction]).detach().numpy()
            sf.write(f_name + "_ref.wav", audio_values, pipeline.sampling_rate)

if __name__=='__main__':
    print("Starting sample generation")

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config', action = 'append', type = str, help = 'Configuration', required=True)
    parser.add_argument('--text', action = 'store', type = str, help = 'Text to speak')
    parser.add_argument('--yconfig', action = 'append', type = str, help = 'Inline yaml config, useful for config overriding')
    parser.add_argument('--with-model', action = 'store', type = str, help = 'Model file')
    args = parser.parse_args()

    cfg = pconfig.load_config(args.config, args.yconfig)

    pipeline = audio_encoding.text_to_speech_pipeline(pconfig.model_path(cfg.model.name, args.with_model))

    if not args.text:
        torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache
        ds = prepare_data.lj_speech_dataset(cfg.prepare_data)

        sds = cut_rows(ds["train"], cfg.generate_samples.limit_rows)
        generate_samples(pipeline, cfg, sds, "train")

        sds = cut_rows(ds["validation"], cfg.generate_samples.limit_rows)
        generate_samples(pipeline, cfg, sds, "validation")

    else:
        id = "".join([c for c in args.text if c.isalpha()][:20])
        id = "test" if not id else id
        ds = Dataset.from_dict({"id": [id], "normalized_text": [args.text]})

        generate_samples(pipeline, cfg, ds, "custom")
