import prepare_data

from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, AutoConfig

def make_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    enc_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=100,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **cfg.model.encoder.__dict__,
    )
    dec_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=prepare_data.TOK_TOKS+2,
        n_ctx=302,
        bos_token_id=prepare_data.TOK_BOS,
        eos_token_id=prepare_data.TOK_EOS,
        **cfg.model.decoder.__dict__,
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = prepare_data.TOK_BOS
    model.config.pad_token_id = prepare_data.TOK_EOS
    return model
