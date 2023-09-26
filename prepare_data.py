import argparse
import pconfig
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import EncodecModel, AutoProcessor
import torch
import os

parser = argparse.ArgumentParser(description='Data preparation script.')
parser.add_argument('--config', action = 'store', type = str, help = 'Configuration', required=True)
args = parser.parse_args()

cfg = pconfig.load_config(args.config)
cfg = cfg.prepare_data
print(cfg)

if not cfg.in_op_threads:
    torch.set_num_threads(1)


print("Starting prepare data")

ds = load_dataset("lj_speech", split="train")

if cfg.limit_rows:
    n_rows = min(cfg.limit_rows, len(ds))
    ds = ds.select(range(n_rows))

# load the model + processor (for pre-processing the audio)
torch.manual_seed(0) # Needed to make model weights deterministic and hence reuse cache
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

ds = ds.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

def to_tokens(audio_raw):
    '''
    Take an audio channel as a numpy array and encode using the Encodec model which return the tokens.

    For the processor documentation:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/feature_extraction_encodec.py
    '''
    audio = processor(raw_audio=audio_raw, sampling_rate=processor.sampling_rate, return_tensors='pt')
    encoder_outputs = model.encode(audio["input_values"], audio["padding_mask"])
    codes = encoder_outputs["audio_codes"] # shape (#examples, channels, codebook, tokens)
    return codes[0, 0]

def add_codes(example):
    # when used in map, the below is the spec for adding new columns
    return {"codes": to_tokens(example["audio"]["array"])}

# Also note that the new column has values of type list rather than tensor despite that
# the mapping function is returning tensor values.

from datasets.fingerprint import Hasher

os.makedirs(f"data/{cfg.prepared_data}", exist_ok=True)
for i in range(cfg.shards):
    ds_s = ds.shard(cfg.shards, i, contiguous=True)
    print("Fingerprint_s", i, ds_s._fingerprint, Hasher.hash(processor), Hasher.hash(model))
    ds_s = ds_s.map(add_codes, num_proc=cfg.map_workers, load_from_cache_file=True)
    print("Fingerprint", i, ds_s._fingerprint)
    ds_s.save_to_disk(f"data/{cfg.prepared_data}/shard-{i}.hf")
