# nlp-sarcasm-detection

## Final Project NLP

### Authors

Nam Bui (nam_bui@student.uml.edu)

Riley Conners (riley_conners@student.uml.edu)

Sam Zuk (samuel_zuk@student.uml.edu)

## Quickstart

### Downloading Datasets

Most of the data needed is included in this repository.
If needed, the Newspaper Sarcasm Detection dataset can be found
[here](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection).
word2vec embeddings can be downloaded using [data/get-embeddings.sh](data/get-embeddings.sh)

### Pre-Processing

To install required dependencies:
```
$ pip install -r src/requirements.txt
```

The sarcasm dataset is first split into train, validation, and test sets
using [data_split.py](src/data_split.py).
The sarcasm dataset is preprocessed by running [data_normalize.py](src/data_normalize.py).
The vocabulary is built by running [build_vocab.py](src/build_vocab.py).
word2vec embeddings are fine-tuned over the dataset by running [word2vec.py](src/word2vec.py).

### Running Models

The Naive Bayes model can be found in [base_model.ipynb](src/base_model.ipynb).
The Jupyter notebook can be run as-is.

The LSTM model can be found in [lstm.py](src/lstm.py)
and can be run with the following arguments and options.
```
$ python3 lstm.py --help
usage: lstm.py [-h] [-d DEVICE] [-t] [--cm CM] cfg

positional arguments:
  cfg                   JSON file with configuration parameters

options:
  -h, --help            show this help message and exit        
  -d DEVICE, --device DEVICE
                        CUDA device
  -t, --test            Model to test.
  --cm CM               Title of confusion matrix plot.        
```

The configuration file for the LSTM is a json file with the following attributes:
```
    "DATA_DIR": data folder,
    "TRAIN_FN": filename of training set,
    "VAL_FN": filename of validation set,
    "TEST_FN": filename of testing set,
    "W2I_FN": filename of word-to-index mapping,
    "OUT_DIR": (optional) output folder,
    "EMBED_FN": (optional) path of embedding,
    "EMBED_SIZE": size of embedding,
    "HIDDEN_SIZE": list of hidden sizes for fully connected network,
    "GRAD_CLIP_VALUE": gradient clipping value,
    "BATCH_SIZE": batch size,
    "NUM_EPOCHS": maximum number of epochs,
    "MIN_EPOCHS": minimum number of epochs before early stoppage,
    "EARLY_STOP_THRESHOLD": loss difference threshold for early stoppage,
    "NOISE_SD": gradient noise,
    "LEARNING_RATE": learning rate,
    "LR_GAMMA": learning rate decay,
    "CHKPT_INTERVAL": epoch interval to save model checkpoints
```
Two configuration files are provided in [src](src).

Training example:
```
src$ python3 lstm.py cfg.json
```

Testing example:
```
src$ python3 lstm.py cfg.json -t
```