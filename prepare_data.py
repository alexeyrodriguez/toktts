import argparse
import pconfig
from datasets import load_dataset, Audio, concatenate_datasets
import torch
import os
from transformers import AutoTokenizer
import numpy as np

from dataset_utils import map_list
from audio_encoding import make_token_encoder

def tokenize_speech(cfg, ds, concat_shards=True):
    """
    Take an audio dataset with columns audio and normalized_text and produce tokens for
    encoder decoder transformer training.
    """

    to_tokens, sampling_rate = make_token_encoder()

    def add_codes(example):
        # clip audio
        upper_ix = cfg.clip_audio_secs * example["audio"]["sampling_rate"] if cfg.clip_audio_secs else example["audio"]["array"].shape[-1]
        audio = example["audio"]["array"][..., :upper_ix]

        res = [to_tokens(audio)]

        if "augmentation_factor" in cfg.__dict__:
            for _ in range(cfg.augmentation_factor - 1):
                noise = torch.randn(*audio.shape) * cfg.augmentation_noise
                res.append(to_tokens(audio+noise))

        return res

    def add_secs(example):
        return {"seconds": example["audio"]["array"].shape[-1] / example["audio"]["sampling_rate"] }

    def add_toks(example):
        return tokenizer(example["normalized_text"])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Process dataset in shards as there appears to be a memory leak
    dss = []
    for i in range(cfg.shards):
        ds_s = ds.shard(cfg.shards, i, contiguous=True)
        ds_s = ds_s.map(add_secs, num_proc=cfg.map_workers)
        ds_s = ds_s.map(add_toks, num_proc=cfg.map_workers)
        ds_s = map_list(ds_s, add_codes, "labels", num_proc=cfg.map_workers)
        dss.append(ds_s)

    if concat_shards:
        return concatenate_datasets(dss)
    else:
        return dss

def lj_speech_dataset(cfg):
    ds = load_dataset("lj_speech", split="train")

    if cfg.limit_rows:
        n_rows = min(cfg.limit_rows, len(ds))
        ds = ds.select(range(n_rows))

    if not cfg.in_op_threads:
        torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    ds = ds.train_test_split(train_size=cfg.split_train_size, seed=cfg.split_seed)

    ds["train"] = tokenize_speech(cfg, ds.pop("train"))
    ds["validation"] = tokenize_speech(cfg, ds.pop("test"))

    if not cfg.in_op_threads:
        torch.set_num_threads(torch_threads)

    return ds

if __name__=='__main__':
    print("Starting prepare data")

    parser = argparse.ArgumentParser(description='Data preparation script.')
    parser.add_argument('--config', action = 'store', type = str, help = 'Configuration', required=True)
    args = parser.parse_args()

    cfg = pconfig.load_config(args.config)
    cfg = cfg.prepare_data
    print(cfg)

    if not cfg.in_op_threads:
        torch.set_num_threads(1)

    ds = load_dataset("lj_speech", split="train")

    torch.manual_seed(0) # Needed to make encodec model weights deterministic and hence reuse cache

    ds_shards = tokenize_speech(cfg, ds, concat_shards=False)

    os.makedirs(f"data/{cfg.prepared_data}", exist_ok=True)
    for i, ds in enumerate(ds_shards):
        ds.save_to_disk(f"data/{cfg.prepared_data}/shard-{i}.hf")
