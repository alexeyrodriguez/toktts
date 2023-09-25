
import torch
torch.set_num_threads(1)

from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

print("Starting prepare data")

#librispeech_dummy_raw = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
librispeech_dummy_raw = load_dataset("lj_speech", split="train")

n_rows = min(100, len(librispeech_dummy_raw))
librispeech_dummy_raw = librispeech_dummy_raw.select(range(n_rows))

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy_raw.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

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

# Look at the documentation for map here: https://huggingface.co/docs/datasets/process
# The num_proc=2 argument unfortunately crashes here
#
# For mac
# oh, now it works with num_proc=2
# times:  1 0:54 /  2 1:21  -- multi threads for intra ops enabled
# times:  1 1:30 /  2 1:10 /  4 1:53 -- no multi threads for intra ops
#
# Also note that the new column has values of type list rather than tensor despite that
# the mapping function is returning tensor values.
librispeech_dummy = librispeech_dummy.map(add_codes, num_proc=4)

