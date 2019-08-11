from .dataset import SkipGramDataset
from .preprocess import Tokenizer


class OpinRankDataset(SkipGramDataset):

    def __init__(self, args):
        SkipGramDataset.__init__(self, args)
        self.name = 'OpinRank Dataset'
        self.files = self.load_line_as_files()
        self.tokenizer = Tokenizer(args, custom_stop={
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'})
        self.generate_examples_multi()

        print(f'There were {len(list(self.term_freq_dict.keys()))} tokens generated')

    def load_line_as_files(self):
        files = self._get_files_in_dir(self.args.dataset_dir)
        lines = []
        for file in files:
            with open(file, 'rb') as f:
                lines.extend([str(l) for l in f])
        return lines

    def read_file(self, file):
        """
        Read In File
        Simply returns the file the was sent in as this
        datasets files are strings

        :param file: String
        :returns: String of file
        """

        return file
