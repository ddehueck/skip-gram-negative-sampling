from .dataset import SkipGramDataset
import re


class WorldOrderDataset(SkipGramDataset):

    def __init__(self, args, examples_path=None, dict_path=None):
        SkipGramDataset.__init__(self, args)
        self.name = 'World Order Book Dataset'
        self.queries = ['nuclear', 'mankind', 'khomeini', 'ronald']

        if examples_path is not None and dict_path is not None:
            self.load(examples_path, dict_path)
        else:
            self.files = self.tokenize_files()
            self.generate_examples_serial()

        print(f'There are {len(self.dictionary)} tokens and {len(self.examples)} examples.')

    def load_files(self):
        return self.files

    def tokenize_files(self):
        files = []
        with open('data/world_order_kissinger.txt') as f:
            for line in f:
                words_no_dig_punc = (re.sub(r'[^\w]', ' ', line.lower())).split()
                words_no_dig_punc = [x for x in words_no_dig_punc if not any(c.isdigit() for c in x)]
                files.append(words_no_dig_punc)

        return files
