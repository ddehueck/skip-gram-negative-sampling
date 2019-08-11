import os
import torch
import multiprocessing
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from .preprocess import Tokenizer
from multiprocessing.dummy import Pool as ThreadPool


class SkipGramDataset(Dataset):

    def __init__(self, args):
        self.args = args
        self.term_freq_dict = dict()
        self.files = self._get_files_in_dir(args.dataset_dir)
        self.tokenizer = Tokenizer(args)
        self.removed_infrequent_tokens = False
        self.examples = []
        self.n_examples = 0
        self.name = ''

    def __getitem__(self, index):
        return self._example_to_tensor(*self.examples[index])

    def __len__(self):
        return len(self.examples)

    def read_file(self, f):
        """
        Read File

        Reads file and returns string. This is to allow different file formats
        to be used.
        :param f: File to be read
        :returns: String
        """
        # Needs to be implemented by child class
        raise NotImplementedError

    def generate_examples_from_file(self, file, tf_dict):
        """
        Generate all examples from a file
        :param file: File from self.files
        :param tf_dict: Term frequency dict
        :returns: List of examples
        """

        doc_str = self.read_file(file)
        try:
            tokenized_doc = self.tokenizer.tokenize_doc(doc_str)
        except Exception as e:
            #print(doc_str)
            raise Exception(e)

        examples = []
        for i, token in enumerate(tokenized_doc):
            # Ensure token is recorded
            self._add_token_to_vocab(token, tf_dict)
            # Generate context words for token in this doc
            context_words = self._generate_contexts(i, tokenized_doc)

            # Form Examples:
            # An example consists of:
            #   center word: token
            #   context word: tokenized_doc[context_word_pos]
            # In the form of:
            # center, context - follows form: (input, target)
            new_examples = [(token, ctxt) for ctxt in context_words]

            # Add to class
            examples.extend(new_examples)
        return examples

    def generate_examples_multi(self):
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = self.args.file_batch_size
        file_batches = self._batch_files(batch_size)

        print('\nGenerating Examples for Dataset (multi-threaded)...')
        for results in tqdm(
                pool.imap_unordered(
                    self._generate_examples_worker,
                    file_batches),
                total=len(self.files) // batch_size + 1):
            # Reduce results into final locations
            examples, tf_dict = results
            self.examples.extend(examples)
            self._reduce_tf_dict(tf_dict)

        pool.close()
        pool.join()

        # Remove any tokens with a frequency of less than 10
        # Remove examples too by regenerating
        if not self.removed_infrequent_tokens:
            tokens_to_remove = set([k for k in self.term_freq_dict if self.term_freq_dict[k] < 10])
            self.tokenizer = Tokenizer(self.args, custom_stop=tokens_to_remove)
            self.removed_infrequent_tokens = True

            # Reset and regenerate examples!
            self.examples = []
            self.term_freq_dict = dict()
            self.generate_examples_multi()

    def _generate_examples_worker(self, file_batch):
        """
        Generate examples worker
        Worker to generate examples in a map reduce paradigm

        :param file_batch: List of files - a subset of self.files
        :returns: list of examples and a term frequency dict for its batch
        """
        tf_dict = dict()
        examples = []

        for f in file_batch:
            examples.extend(self.generate_examples_from_file(f, tf_dict))
        return examples, tf_dict

    def _batch_files(self, batch_size):
        """
        Batch Files
        Seperates self.files into smaller batches of files of
        size batch_size

        :param batch_size: Int - size of each batch
        :returns: Generator of batches
        """
        n_files = len(self.files)
        for b_idx in range(0, n_files, batch_size):
            # min() so we don't index outside of self.files
            yield self.files[b_idx:min(b_idx + batch_size, n_files)]

    def _add_token_to_vocab(self, token, tf_dict):
        """
        Add token to the token frequency dict
        Adds new tokens to the tf_dict  and keeps track of
        frequency of tokens

        :param token: String
        :param tf_dict: A {"token": frequency,} dict
        :returns: None
        """

        if token not in tf_dict.keys():
            tf_dict[token] = 1
        else:
            # Token in vocab - increase frequency for token
            tf_dict[token] += 1

    def _reduce_tf_dict(self, tf_dict):
        """
        Reduce a term frequency dictionary
        Updates self.term_freq_dict with values in tf_dict argument.
        Adds new keys if needed or just sums frequencies

        :param tf_dict: A term frequency dictionary
        :returns: None - updates self.term_freq_dict
        """
        for key in tf_dict:
            if key in self.term_freq_dict.keys():
                # Add frequencies
                self.term_freq_dict[key] += tf_dict[key]
            else:
                # Merge
                self.term_freq_dict[key] = tf_dict[key]

    def _generate_contexts(self, token_idx, tokenized_doc):
        """
        Generate Token's Context Words
        Generates all the context words within the window size defined
        during initialization around token.

        :param token_idx: Index at which center token is found in tokenized_doc
        :param tokenized_doc: List - Document broken into tokens
        :returns: List of context words
        """
        contexts = []
        # Iterate over each position in window
        for w in range(-self.args.window_size, self.args.window_size + 1):
            context_pos = token_idx + w

            # Make sure current center and context are valid
            is_outside_doc = context_pos < 0 or context_pos >= len(tokenized_doc)
            center_is_context = token_idx == context_pos

            if is_outside_doc or center_is_context:
                # Not valid - skip to next window position
                continue

            contexts.append(tokenized_doc[context_pos])
        return contexts

    def _example_to_tensor(self, example, target):
        """
        Takes raw example and turns it into tensor values

        :params example: Tuple of form: (center word, document id)
        :params target: String of the target word
        :returns: A tuple of tensors
        """
        center_idx = list(self.term_freq_dict.keys()).index(example[0])
        target_idx = list(self.term_freq_dict.keys()).index(target)

        doc_id = torch.tensor([int(example[1])])
        center, target = torch.tensor([int(center_idx)]), torch.tensor([int(target_idx)])
        return ((center, doc_id), target)

    @staticmethod
    def _get_files_in_dir(src_dir):
        if src_dir is None:
            return []

        files = []
        src_dir = os.path.expanduser(src_dir)
        d = os.path.join(src_dir)

        if not os.path.isdir(src_dir):
            raise Exception('Path given is not a directory.')

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                files.append(path)

        return files