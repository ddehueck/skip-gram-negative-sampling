import spacy
import re


class Tokenizer:

    def __init__(self, args, custom_stop=set()):
        self.args = args
        self.custom_stop = custom_stop
        # Define pipeline - use different nlp is using pretrained
        self.nlp = spacy.load("en_core_web_sm", disable=[])
        # Merge named entities
        merge_ents = self.nlp.create_pipe("merge_entities")
        self.nlp.add_pipe(merge_ents)

    def tokenize_doc(self, doc_str):
        """
        Tokenize a document string
        Modified version of Moody's Tokenization in:
        https://github.com/cemoody/lda2vec/blob/master/lda2vec/preprocess.py

        :params doc_str: String
        :returns: list of Strings, i.e. tokens
        """

        # Send doc_str through pipeline
        spacy_doc = self.nlp(doc_str)
        # Filter
        filtered_doc = filter(self.is_valid_token, spacy_doc)
        # Convert to text make lowercase
        clean_doc = [t.text.lower().strip() for t in filtered_doc]
        # Only allow characters in the alphabet, '_', and digits
        clean_doc = [re.sub('[^a-zA-Z0-9]', '', t) for t in clean_doc]
        # Remove any resulting empty indices
        clean_doc = [t for t in clean_doc if len(t) > 0]
        # Filter out any custom stop
        clean_doc = [t for t in clean_doc if t not in self.custom_stop]

        return clean_doc

    def is_valid_token(self, token):
        """
        Determines if a token is valid or not

        :params token: String
        :returns: Boolean
        """
        if token.like_url:
            return False
        if token.like_email:
            return False
        if token.is_stop or token.text in self.custom_stop:
            return False

        return True

    def moodys_merge_noun_chunks(self, doc):
        """
        Merge noun chunks into a single token.

        Modified from sources of:
        - https://github.com/cemoody/lda2vec/blob/master/lda2vec/preprocess.py
        - https://spacy.io/api/pipeline-functions#merge_noun_chunks

        :params doc: Doc object.
        :returns: Doc object with merged noun chunks.
        """
        bad_deps = ('amod', 'compound')

        if not doc.is_parsed:
            return doc
        with doc.retokenize() as retokenizer:
            for np in doc.noun_chunks:

                # Only keep adjectives and nouns, e.g. "good ideas"
                while len(np) > 1 and np[0].dep_ not in bad_deps:
                    np = np[1:]

                if len(np) > 1:
                    # Merge NPs
                    attrs = {"tag": np.root.tag, "dep": np.root.dep}
                    retokenizer.merge(np, attrs=attrs)
        return doc


class SimpleTokenizer:

    def __init__(self, args, custom_stop=set()):
        self.args = args
        self.custom_stop = custom_stop

    def tokenize_doc(self, doc_str):
        # Filter out any custom stop
        clean_doc = [t for t in doc_str.split() if t not in self.custom_stop]
        return clean_doc
