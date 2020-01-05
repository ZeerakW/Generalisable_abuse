import re
import pdb
import spacy
from . import base
from string import punctuation


class Preprocessors(object):

    def __init__(self):
        """Initialise cleaner class.
        """
        self.tagger = spacy.load('en', disable = ['ner', 'parser'])
        self.liwc_dict = None
        self.slurs = None

    def word_length(self, doc: base.DocType):
        """Represent sentence as the length of each token.
        :doc (base.DocType): Document to be processed.
        :returns: Processed document.
        """
        return [len(tok) for tok in doc]

    def syllable_count(self, doc: base.DocType) -> base.List[int]:
        """Represent sentence as the syllable count for each word.
        :doc (base.DocType): Document to be processed.
        :returns: Processed document.
        """
        return [self._syllable_counter(tok) for tok in doc]

    def _syllable_counter(self, tok: str) -> int:
        """Calculate syllables for each token.
        :tok (str): The token to be analyzed.
        :returns count (int): The number of syllables in the word.
        """
        count = 0
        vowels = 'aeiouy'
        exceptions = ['le', 'es', 'e']
        prev_char = None

        for i in tok:
            if i == len(tok) and (prev_char + tok[i] in exceptions or tok[i] in exceptions):
                prev_char = tok[i]
                continue
            if (tok[i] in vowels) and (prev_char not in vowels and not prev_char):
                prev_char = tok[i]
                count += 1

    def load_slurs(self):
        self.slurs = None
        # TODO Update this with a slur list

    def slur_replacement(self, doc, window_size: int):
        """Produce documents where slurs are replaced.
        :doc (base.List[str]): Document to be processed.
        :window_size (int): Size of the window.
        :returns doc: processed document
        """
        if self.slurs is not None:
            self.slurs = self.load_slurs()

        pos = [tok for tok in self.tagger(" ".join(doc))]

        for i in range(doc):
            if doc[i] in self.slurs:
                doc[i] = pos[i]

                min_pos = 0 if i - window_size < 0 else i - window_size
                max_pos = len(doc) - 1 if i + window_size > len(doc) - 1 else i + window_size

                for j in range(min_pos, max_pos, 1):  # Replace within window
                    doc[j] = pos[j]

        return doc

    def word_token(self, doc: base.List[str]):
        """Produce word tokens.
        :doc (base.List[str]): Document to be processed.
        :returns: processed document
        """
        return doc

    def ptb_tokenize(self, document: base.DocType, processes: base.List[str] = None):
        """Tokenize the document using SpaCy, get PTB tags and clean it as it is processed.
        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        self.processes = processes if processes else self.processes
        toks = [tok.tag_ for tok in self.tagger(" ".join(document))]
        return toks

    def read_liwc(self) -> dict:
        with open('/Users/zeerakw/Documents/PhD/projects/active/Generalisable_abuse/data/liwc-2015.csv', 'r') as liwc_f:
            liwc_dict = {}
            for line in liwc_f:
                k, v = line.strip('\n').split(',')
                if k in liwc_dict:
                    liwc_dict[k] += [v]
                else:
                    liwc_dict.update({k: [v]})

        return liwc_dict

    def _compute_liwc_token(self, tok, kleene_star):
        if tok in self.liwc_dict:
            term = self.liwc_dict[tok]
        else:
            liwc_cands = [r for r in kleene_star if r in tok]
            num_cands = len(liwc_cands)

            if num_cands == 0:
                term = 'NUM' if re.findall(r'[0-9]+', tok) else 'UNK'

            elif num_cands == 1:
                term = liwc_cands[0] + '*'

            elif num_cands > 1:
                sorted_cands = sorted(liwc_cands, key=len, reverse = True)  # Longest first
                term = sorted_cands[0] + '*'

            if term not in ['UNK', 'NUM']:
                liwc_term = self.liwc_dict[term]

                if isinstance(liwc_term, list):
                    term = "_".join(liwc_term)
                else:
                    term = liwc_term
        if isinstance(term, list):
            term = "_".join(term)

        return term

    def compute_unigram_liwc(self, doc: base.DocType):
        """Compute LIWC for each document document.
        :doc (base.DocType): Document to operate on.
        :returns liwc_doc (base.DocType): Document represented as LIWC categories.
        """

        if not self.liwc_dict:
            self.liwc_dict = self.read_liwc()
        liwc_doc = []
        kleene_star = [k[:-1] for k in self.liwc_dict if k[-1] == '*']

        if isinstance(doc, str):
            doc = [w if w[0] not in punctuation and w[-1] not in punctuation
                   else w.strip(punctuation) for w in doc.split()]

        liwc_doc = [self._compute_liwc_token(tok, kleene_star) for tok in doc]

        try:
            assert(len(liwc_doc) == len(doc))
        except AssertionError:
            pdb.set_trace()

        return " ".join(liwc_doc)


class Cleaner(object):
    """A class for methods for cleaning."""

    def __init__(self, processes: base.List[str] = None):
        """Initialise cleaner class.
        :processes base.List[str]: Cleaning operations to be taken.
        """
        self.processes = processes
        self.tagger = spacy.load('en', disable = ['ner', 'parser', 'textcats'])
        self.liwc_dict = None

    def clean_document(self, text: base.DocType, processes: base.List[str] = None):
        """Data cleaning method.
        :text (types.DocType): The document to be cleaned.
        :processes (List[str]): The cleaning processes to be undertaken.
        :returns cleaned: Return the cleaned text.
        """
        self.processes = self.processes if self.processes else processes
        cleaned = str(text)
        if 'lower' in self.processes or 'lower' in processes:
            cleaned = cleaned.lower()
        if 'url' in self.processes or 'url' in processes:
            cleaned = re.sub(r'https?:/\/\S+', 'URL', cleaned)
        if 'hashtag' in self.processes or 'hashtag' in processes:
            cleaned = re.sub(r'#[a-zA-Z0-9]*\b', 'HASHTAG', cleaned)
        if 'username' in self.processes or 'username' in processes:
            cleaned = re.sub(r'@\S+', 'AT_USER', cleaned)
        cleaned = re.sub("'", ' ', cleaned)

        return cleaned

    def tokenize(self, document: base.DocType, processes: base.List[str] = None):
        """Tokenize the document using SpaCy and clean it as it is processed.
        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        if processes:
            toks = [tok.text for tok in self.tagger(self.clean_document(document, processes = processes))]
        else:
            toks = [tok.text for tok in self.tagger(self.clean_document(document))]
        return toks

    def ptb_tokenize(self, document: base.DocType, processes: base.List[str] = None):
        """Tokenize the document using SpaCy, get PTB tags and clean it as it is processed.
        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        self.processes = processes if processes else self.processes
        toks = [tok.tag_ for tok in self.tagger(self.clean_document(document))]
        return " ".join(toks)

    def sentiment_tokenize(self, document: base.DocType, processes: base.List[str] = None):
        """Tokenize the document using SpaCy, get sentiment and clean it as it is processed.
        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        raise NotImplementedError
        # self.processes = processes if processes else self.processes
        # toks = [tok.sentiment for tok in self.tagger(self.clean_document(document))]
        # pdb.set_trace()
        # return toks
