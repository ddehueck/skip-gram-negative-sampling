# Word2Vec Skip-Gram Negative Sampling in PyTorch

A PyTorch Implementation of the Skipgram Negative Sampling Word2Vec model as described in Mikolov et al. See the jax implementation for a bit of speed up: https://github.com/ddehueck/jax-skip-gram-negative-sampling. However, the PyTorch code seems to provide slighty better results on smaller datasets.

## Tokenizing a dataset:

The code provided should be easily extendable to other datasets. Please see `datasets/dataset.py` for the base class you should inherit in your own (just like `datasets/world_order.py` does). Feel free to create an issue if you are having trouble. 

```python
from .dataset import SkipGramDataset
from .preprocess import Tokenizer


class ExampleDataset(SkipGramDataset):

    def __init__(self, args, examples_path=None, dict_path=None):
        SkipGramDataset.__init__(self, args)
        self.name = 'Example Dataset'
        self.queries = ['words', 'to', 'watch', 'during', 'training']

        if examples_path is not None and dict_path is not None:
            self.load(examples_path, dict_path)
        else:
            self.tokenizer = Tokenizer(args)
            # Set self.files to a list of tokenized data!
            self.files = self.tokenize_files()
            # Generates examples in window size - e.g. (center_idx, context_idx)
            self.generate_examples_serial()
            # Save dataset files - this tokenization and example generation can take awhile with a lot of data
            self.save('training_examples.pth', 'dictionary.pth')

        print(f'There are {len(self.dictionary)} tokens and {len(self.examples)} examples.')

    def load_files(self):
        """ Requires by SkipGramDataset to generate examples - must be tokenized files """
        return self.files

    def tokenize_files(self):
        # read in from a file or wherever your data is kept
        raw_data = ["this is document_1", "this is document_2", ..., "this is document_n"] 
        return [self.tokenizer.tokenize_doc(f) for f in raw_data]

```

## Running the model:

There are many hyperparameters that are easy to set via command-line arguments when calling `train.py`:

Example: 

``python train.py --embedding-len 64 --batch-size 2048 --epochs 500``

All hyperparameters in `train.py`:

```
optional arguments:
  -h, --help            show this help message and exit
  --dataset-dir DATASET_DIR
                        dataset directory (default: data/)
  --workers N           dataloader threads (default: 4)
  --window-size WINDOW_SIZE
                        Window size used when generating training examples
                        (default: 5)
  --file-batch-size FILE_BATCH_SIZE
                        Batch size used when multi-threading the generation of
                        training examples (default: 250)
  --embedding-len EMBEDDING_LEN
                        Length of embeddings in model (default: 128)
  --epochs N            number of epochs to train for - iterations over the
                        dataset (default: 15)
  --batch-size N        number of examples in a training batch (default: 1024)
  --lr LR               learning rate (default: 1e-3)
  --seed S              random seed (default: 42)
  --log-step LOG_STEP   Step at which for every step training info is logged.
                        (default: 250)
  --device DEVICE       device to train on (default: cuda:0 if cuda is
                        available otherwise cpu)
 ```

