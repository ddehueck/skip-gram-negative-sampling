import torch
from tqdm import tqdm
from gensim.corpora import Dictionary
from torch.utils.data.dataset import Dataset


class SkipGramDataset(Dataset):

    def __init__(self, args):
        self.args = args
        self.dictionary = None
        self.examples = []
        self.name = ''

    def __getitem__(self, index):
        return self._example_to_tensor(*self.examples[index])

    def __len__(self):
        return len(self.examples)

    def save(self, examples_path, dict_path):
        print('Saving Dataset Examples...')
        torch.save({
             'examples': self.examples,
        }, examples_path)
        print('Saving Dataset Dictionary...')
        self.dictionary.save(dict_path)
        print('Saved Dataset!')

    def load(self, examples_path, dict_path):
        print('Loading Dataset Examples...')
        self.examples = torch.load(examples_path)['examples']
        print('Loading Dataset Dictionary...')
        self.dictionary = Dictionary().load(dict_path)
        print('Loaded Saved Dataset!')

    def generate_examples_serial(self):
        """
        Generates examples with no multiprocessing - straight through!
        :return: None - updates class properties
        """
        # Now we have a Gensim Dictionary to work with
        self._build_dictionary()
        # Remove any tokens with a frequency less than 10
        self.dictionary.filter_extremes(no_below=10, no_above=0.75)

        self.examples = []
        for file in tqdm(self.load_files(), desc="Generating Examples (serial)"):
            file = self.dictionary.doc2idx(file)
            self.examples.extend(self._generate_examples_from_file(file))

    def load_files(self):
        """
        Sets self.files as a list of tokenized documents!
        :returns: List of files
        """
        # Needs to be implemented by child class
        raise NotImplementedError

    def _build_dictionary(self):
        """
        Creates a Gensim Dictionary
        :return: None - modifies self.dictionary
        """
        print("Building Dictionary...")
        self.dictionary = Dictionary(self.load_files())

    def _generate_examples_from_file(self, file):
        """
        Generate all examples from a file within window size
        :param file: File from self.files
        :returns: List of examples
        """

        examples = []
        for i, token in enumerate(file):
            if token == -1:
                # Out of dictionary token
                continue

            # Generate context tokens for the current token
            context_words = self._generate_contexts(i, file)

            # Form Examples:
            # center, context - follows form: (input, target)
            new_examples = [(token, ctxt) for ctxt in context_words if ctxt != -1]

            # Add to class
            examples.extend(new_examples)
        return examples

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

    def _example_to_tensor(self, center, target):
        """
        Takes raw example and turns it into tensor values

        :params example: Tuple of form: (center word, document id)
        :params target: String of the target word
        :returns: A tuple of tensors
        """
        center, target = torch.tensor([int(center)]), torch.tensor([int(target)])
        return center, target
