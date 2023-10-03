import argparse
import pconfig
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import EncodecModel, AutoProcessor
import torch
import os
from transformers import CanineTokenizer, AutoTokenizer
from datasets.fingerprint import Hasher

from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

TOK_TOKS = 1024
TOK_BOS = 1024
TOK_EOS = 1025

def tokenize_speech(cfg, ds, concat_shards=True):
    """
    Take an audio dataset with columns audio and normalized_text and produce tokens for
    encoder decoder transformer training.
    """

    # load the model + processor (for pre-processing the audio)
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    def to_tokens(audio_raw):
        '''
        Take an audio channel as a numpy array and encode using the Encodec model which return the tokens.

        For the processor documentation:
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/feature_extraction_encodec.py
        '''
        audio = processor(raw_audio=audio_raw, sampling_rate=processor.sampling_rate, return_tensors='pt')
        upper_ix = cfg.clip_audio_secs * processor.sampling_rate if cfg.clip_audio_secs else audio["input_values"].shape[-1]
        encoder_outputs = model.encode(audio["input_values"][:, :, :upper_ix], audio["padding_mask"][:, :upper_ix])
        codes = encoder_outputs["audio_codes"] # shape (#examples, channels, codebook, tokens)
        codes = codes[0, 0] # select the only example, and there is only one channel
        codes = codes.transpose(1, 0).reshape(-1) # tokens in sample order, each sample has all codebook tokens next to each other
        codes = torch.cat([codes, torch.tensor([TOK_EOS])], 0)
        return codes

    def add_codes(example):
        codes = to_tokens(example["audio"]["array"])
        return {"labels": codes}
        # Also note that the new column has values of type list rather than tensor despite that
        # the mapping function is returning tensor values.

    def add_secs(example):
        return {"seconds": example["audio"]["array"].shape[-1] / example["audio"]["sampling_rate"] }

    def add_toks(example):
        return tokenizer(example["normalized_text"])

    if cfg.limit_rows:
        n_rows = min(cfg.limit_rows, len(ds))
        ds = ds.select(range(n_rows))

    ds = ds.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Process dataset in shards as there appears to be a memory leak
    dss = []
    for i in range(cfg.shards):
        ds_s = ds.shard(cfg.shards, i, contiguous=True)
        ds_s = ds_s.map(add_secs, num_proc=cfg.map_workers)
        ds_s = ds_s.map(add_codes, num_proc=cfg.map_workers)
        ds_s = ds_s.map(add_toks, num_proc=cfg.map_workers)
        dss.append(ds_s)

    if concat_shards:
        return concatenate_datasets(dss)
    else:
        return dss

def lj_speech_dataset(cfg):
    ds = load_dataset("lj_speech", split="train")
    ds = tokenize_speech(cfg.prepare_data, ds)
    ds = ds.train_test_split(train_size=0.9, seed=20)
    ds["validation"] = ds.pop("test")
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
