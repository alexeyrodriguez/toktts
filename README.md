# An exploration of Text to Speech using a sequence to sequence transformer

`toktts` is an exploration in building a Text-to-Speech (TTS) system using a transformer
architecture with discrete tokens. The goal is to create a system that is easy to play with,
and easy to get interesting results without excessive GPU budget and without excessive hacking
time. The following design decisions support the above simplicity goal:
 * Use a pure discrete token transformer architecture rather than directly generating mel/linear
 spectrograms or time domain waveforms.
 This is not a new idea, there are several projects that also use Meta's Encodec for this purpose. (TODO REFS)
 Directly working with tokens means that we don't have to worry about finicky details about losses
 in the spectral/time domain.
 * Use an encoder decoder architecture, rather than a purely decoder transformer in order to avoid
 the bookkeeping of having text and audio tokens in the same sequence.
 * Generate only the first two seconds of audio to save compute budget.
 `toktts`` does not attempt to shorten the input text to match the smaller audio size, probably it would be a good idea as it
 appears that short text inputs have low performance in fitted models.
 But we could improve on this aspect working on the dataset.
 * Use Hugging Face's dataset to cache Encodec's discrete audio encoding and use one single large
 machine to generate the features (discrete audio tokens) rather than a more complex distributed setup.

# Setup 

Some technical details about the implementation:
 * The model is an encoder-decoder architecture. The encoder encodes the input characters
 (no phonemizer) into a latent representation which the decoder tower uses with cross-attention.
 The decoder models the audio tokens in a causal way.
 * The audio tokens are obtained using Meta's Encodec (TODO REF) from the first two
 seconds of the audio waveforms.
 We use the lowest fidelity of encoding (1.5kbps) which encodes audio at a rate of
 `75Hz` with two discrete tokens each out of a vocabulary of size `1024`. We arrive at
 the 1.5kpbs bit rate by `75 * 2 * 10` which yields `1500` bits per second.
 Taking into account two seconds of audio that we use, the TTS model needs to generate
 a sequence of 300 audio tokens.

# Installation

A reasonable machine to train the models in this repository would require at least 16 CPU cores, 120Gb of disk space,
150Gb of RAM and a GPU like the RTX4090. The amount of RAM is required to allow fast generation of the training
dataset. There seems to be a memory leak, possibly in Encodec. One can sidestep the need for RAM using the `shards`
option in the configuration.

The model in `config/two_secs_augmented/ex_small.yaml` would train in a configuration like the above in about 7 hours
with about 1 additional hour to generate the training and validation datasets.

One can create a working training conda training environment with `conda-environment.yaml`. It's important to have
the right versions of packages otherwise the dataset generation takes too long, for a reason that I haven't yet
figured out.

# Running

Initially I used Google Cloud Platform but it turned out to be much more economic to rent a server on vast.ai.

When packages are installed one can start simple training runnning:

> python trainer.py --config config/basic/small_train.yaml

This invocation will download LJSpeech (~13K training examples) and generate datasets for a small amount of examples.

It is highly recommended to log training runs in wandb by modifying and using the template file in `config/templates/wandb_TEMPLATE.yaml`.

After running the trainer one can generate samples using parts of the training and validation datasets using the generation script:

> python generate_samples.py --config config/basic/small_train.yaml

Alternatively one can use custom text passing the `--text` option or even better one can use the provided pipeline
from within a Notebook for example:

> pipeline = text_to_speech_pipeline(model_path)
> audio = pipeline("I am happy you are here", forward_params=forward_params)
> IPython.display.Audio(audio["audio"], rate=audio["sampling_rate"])

See the example notebook for an example on how to set `forward_params`.

The model that is used in the example notebook was trained from `config/two_secs_augmented/ex_small.yaml`.
The model was trained for 89 epochs (interrupted), uses an augmentation factor of 4 and a batch size of 64.
This model produces interesting examples but unfortunately the augmentation appears to generate a faint clicking
background that is absent without augmentation (and yes, augmentation prevents overfitting for that model).