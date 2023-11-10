# toktts

`toktts` is an exploration in building a Text-to-Speech (TTS) system using a sequence-to-sequence
transformer architecture with discrete tokens.

For some quick examples, see the example Notebook here:
[sample_from_model.ipynb](https://nbviewer.org/github/alexeyrodriguez/toktts/blob/main/sample_from_model.ipynb)

The goal is to create a system that is easy to play with, and with which it is easy to get interesting
results, without an excessive GPU budget and without excessive hacking time.
The following design decisions support the above simplicity goal:
 * Use a pure discrete token transformer architecture rather than directly generating mel/linear
 spectrograms or time domain waveforms.
 This is not a new idea, there are several projects that also use Meta's [Encodec](https://github.com/facebookresearch/encodec) for this purpose. (TODO REFS)
 Directly working with tokens means that we don't have to worry about finicky details about losses
 in the spectral/time domain.
 * Use an encoder decoder architecture, rather than a purely decoder transformer in order to avoid
 the bookkeeping of having both text and audio tokens in the same sequence.
 * Using a pure transformer architecture allows us to more easily get a decent utilization of the GPU
 (provided that features are computed efficiently, which is the case due to caching and
 small sequence length) and hence a better use of our GPU budget.
 * Generate only the first two seconds of audio to save compute budget.
 `toktts` does not attempt to shorten the input text to match the smaller audio size,
 probably it would be a good idea to do so as it
 appears that short text inputs have low performance in fitted models.
 * Try to keep feature engineering simple, no separate preprocessing data script, all processing
 in a single machine, no phonemizer, also no input text tokenization (one character is one token).
 * Also there is no text normalization (e.g. `1` to `one`, it would be great to add it)
 
# Setup 

Some technical details about the implementation:
 * The model is an encoder-decoder architecture. The encoder encodes the input characters
 (no phonemizer) into a latent representation which the decoder consumes using cross-attention.
 The decoder models the audio tokens in a causal way.
 * The audio tokens are obtained using Meta's [Encodec](https://github.com/facebookresearch/encodec) from the first two
 seconds of the audio waveforms.
 We use the lowest fidelity of encoding (1.5kbps) which encodes audio at a rate of
 `75Hz` with two discrete tokens each out of a vocabulary of size `1024`. We arrive at
 the 1.5kpbs bit rate by `75 * 2 * 10` which yields `1500` bits per second.
 Taking into account the two seconds of audio that we use, the TTS model needs to generate
 a sequence of 300 audio tokens.
 * To keep feature engineering simple, we prefer to use a large machine for generating datasets
 and use caching from Hugging Face's `datasets` package as much as possible.

# Installation

A reasonable machine to train the models in this repository would require at least 16 CPU cores, 120Gb of disk space,
150Gb of RAM and a GPU like the RTX4090. The amount of RAM and CPU cores are required to allow fast generation of the training
dataset. There seems to be a memory leak, possibly in Encodec. One can sidestep the need for RAM due to the leak using the `shards`
option in the configuration.

Assuming a machine like the above, the model in `config/two_secs_augmented/ex_small.yaml` would train in about 7 hours
with about 1 additional hour to generate the training and validation datasets.

One can create a `conda` training environment using `conda-environment.yaml`. It's important to have
the right versions of packages, otherwise the dataset generation takes too long for a reason that I haven't yet
figured out.

# Running

Initially I used Google Cloud Platform but it turned out to be much cheaper to rent a server from https://vast.ai/.

After installation, to start a simple training, try:

```
python trainer.py --config config/basic/small_train.yaml
```

This invocation will download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (~13K training examples) and generate datasets for a small amount of examples.

It is highly recommended to log training runs in wandb by modifying and using the template file in `config/templates/wandb_TEMPLATE.yaml`.

After running the trainer one can generate samples using parts of the training and validation datasets using the generation script:

```
python generate_samples.py --config config/basic/small_train.yaml
```

Alternatively one can use custom text passing the `--text` option or even better one can use the provided pipeline
from within a Notebook, for example:

```python
pipeline = text_to_speech_pipeline(model_path)
audio = pipeline("I am happy you are here", forward_params=forward_params)
IPython.display.Audio(audio["audio"], rate=audio["sampling_rate"])
```

See the example notebook for an example on how to set `forward_params`.

The model that is used in the example notebook was trained from `config/two_secs_augmented/ex_small.yaml`.
The model was trained for 89 epochs (interrupted), uses an augmentation factor of 4 and a batch size of 64.
This model produces interesting examples but unfortunately the augmentation appears to generate a faint clicking
background that is absent without augmentation (and yes, augmentation prevents overfitting for that model).

## Example Notebook

The example Notebook can be viewed here: [sample_from_model.ipynb](https://nbviewer.org/github/alexeyrodriguez/toktts/blob/main/sample_from_model.ipynb)
