from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, EncodecModel, AutoProcessor, pipeline
import torch
import numpy as np
import audio_encoding

from dataset_utils import map_list

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


def make_token_encoder():
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    def to_tokens(audio_raw):
        '''
        Take an audio channel as a numpy array and encode using the Encodec model which return the tokens.

        For the processor documentation:
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/feature_extraction_encodec.py
        '''
        audio = processor(raw_audio=audio_raw, sampling_rate=processor.sampling_rate, return_tensors='pt')
        encoder_outputs = model.encode(audio["input_values"], audio["padding_mask"])
        codes = encoder_outputs["audio_codes"] # shape (#examples, channels, codebook, tokens)
        codes = codes[0, 0] # select the only example, and there is only one channel
        return flatten_audio_tokens(codes)
    
    return to_tokens, processor.sampling_rate


class EncodecDecoder():
    def __init__(self):
       self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    def __call__(self, toks):
        toks = toks[0] # We only work with batch size 1 for now
        toks = toks if isinstance(toks, list) else toks.tolist()
        toks = unflatten_audio_tokens(toks)
        toks = torch.tensor(toks).view(1, 1, 2, -1)
        audio_values = self.model.decode(toks, [None])[0]
        return audio_values[0, 0]


def text_to_speech_pipeline(model):
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    pip = pipeline("text-to-speech", model=model, tokenizer=tokenizer, sampling_rate=processor.sampling_rate)
    pip.vocoder = audio_encoding.EncodecDecoder() # cheat a bit
    return pip
