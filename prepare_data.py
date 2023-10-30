import argparse
import pconfig
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import EncodecModel, AutoProcessor
import torch
import os
from transformers import CanineTokenizer, AutoTokenizer
from datasets.fingerprint import Hasher
import numpy as np

from datasets.utils.py_utils import Pickler, pklregister

# Needed for deterministic hashing of sets across python runs
# Not needed anymore when datasets is updated, see https://github.com/huggingface/datasets/pull/6318
@pklregister(set)
def _save_set(pickler, obj):
    from datasets.fingerprint import Hasher

    args = (sorted(obj, key=Hasher.hash),)
    pickler.save_reduce(set, args, obj=obj)

from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

TOK_TOKS = 1024
TOK_BOS = 1024
TOK_EOS = 1025

def flatten_audio_tokens(codes):
    'Takes a token array of shape (codebook, tokens) and flattens it'
    codes = codes.transpose(1, 0).reshape(-1) # tokens in sample order, each sample has all codebook tokens next to each other
    codes = torch.cat([codes, torch.tensor([TOK_EOS])], 0) # Add end of sentence token
    return codes

def unflatten_audio_tokens(codes):
    '''
    Remove BOS and EOS tokens and restore codebook dimension.
    It might drop samples to match number of codebooks
    Tokens are passed in as a list
    '''
    codes = codes[1:]
    if TOK_EOS in codes:
        codes = codes[:codes.index(TOK_EOS)]

    # massage tokens back into place
    num_toks = len(codes) // 2
    codes = codes[:num_toks*2] # drop tokens if prediction inserted eos in the wrong place
    codes = np.array(codes).reshape(num_toks, 2).transpose(1, 0)
    return codes.reshape(2, num_toks)

def make_token_decoder():
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    def decode(toks):
        toks = unflatten_audio_tokens(toks)
        toks = torch.tensor(toks).view(1, 1, 2, -1)
        audio_values = model.decode(toks, [None])[0]
        return audio_values[0, 0].detach()

    return decode, processor.sampling_rate


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
        return flatten_audio_tokens(codes)

    def add_codes(example):
        codes = to_tokens(example["audio"]["array"])
        return {"labels": codes}
        # Also note that the new column has values of type list rather than tensor despite that
        # the mapping function is returning tensor values.

    def add_secs(example):
        return {"seconds": example["audio"]["array"].shape[-1] / example["audio"]["sampling_rate"] }

    def add_toks(example):
        return tokenizer(example["normalized_text"])

    def process_data(example):
        d = {}
        d.update(add_secs(example))
        d.update(add_codes(example))
        return d

    ds = ds.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Process dataset in shards as there appears to be a memory leak
    dss = []
    for i in range(cfg.shards):
        ds_s = ds.shard(cfg.shards, i, contiguous=True)
        ds_s = ds_s.map(process_data, num_proc=cfg.map_workers)
        # split this one out because there is some underterministic state in the tokenizer apparently
        ds_s = ds_s.map(add_toks, num_proc=cfg.map_workers)
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

    ds = tokenize_speech(cfg, ds)

    if not cfg.in_op_threads:
        torch.set_num_threads(torch_threads)

    ds = ds.train_test_split(train_size=cfg.split_train_size, seed=cfg.split_seed)
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
