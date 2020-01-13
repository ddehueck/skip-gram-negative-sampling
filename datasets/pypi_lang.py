from .dataset import SkipGramDataset
import pandas as pd
from .preprocess import Tokenizer
from tqdm import tqdm


class PyPILangDataset(SkipGramDataset):

    def __init__(self, args, examples_path=None, dict_path=None):
        SkipGramDataset.__init__(self, args)
        self.name = 'PyPI Language Dataset'
        self.queries = ['tensorflow', 'pytorch', 'nlp', 'performance', 'encryption']

        if examples_path is not None and dict_path is not None:
            self.load(examples_path, dict_path)
        else:
            self.tokenizer = Tokenizer(args)
            self.files = self.tokenize_files()
            self.generate_examples_serial()

            self.save('pypi_examples.pth', 'pypi_dict.pth')

        print(f'There are {len(self.dictionary)} tokens and {len(self.examples)} examples.')

    def load_files(self):
        return self.files

    def tokenize_files(self):
        node_lang_df = pd.read_csv(self.args.dataset_dir, na_filter=False)
        lang_data = node_lang_df['language'].values
        return [self.tokenizer.tokenize_doc(f) for f in tqdm(lang_data, desc='Tokenizing Docs')]
