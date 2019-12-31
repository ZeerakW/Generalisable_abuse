import re
import pdb
import spacy
from . import base
from string import punctuation


class Preprocessors(object):

    def __init__(self):
        """Initialise cleaner class.
        """
        self.tagger = spacy.load('en')
        self.liwc_dict = None
        self.slurs = None

    def slur_replacement(self, doc):
        """Produce documents where slurs are replaced.
        :doc (base.List[str]): Document to be processed.
        :returns: processed document
        """
        raise NotImplementedError

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
        toks = [tok.tag_ for tok in self.tagger(self.clean_document(document))]
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
        self.tagger = spacy.load('en')
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
