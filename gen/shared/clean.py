import re
import pdb
import spacy
import gen.shared.custom_types as t
from string import punctuation


class Cleaner(object):
    """A class for methods for cleaning."""

    def __init__(self, cleaners: t.List[str] = None):
        """Initialise cleaner class.
        :param cleaners t.List[str]: Cleaning operations to be taken.
        """
        self.processes = cleaners
        self.cleaners = cleaners
        self.tagger = spacy.load('en')
        self.liwc_dict = None

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

    def clean_document(self, text: t.DocType, cleaners: t.List[str] = None):
        """Data cleaning method.
        :param text (types.DocType): The document to be cleaned.
        :param cleaners (List[str]): The cleaning processes to be undertaken.
        :return cleaned: Return the cleaned text.
        """
        self.cleaners = self.cleaners if self.cleaners else cleaners
        cleaned = str(text)
        if 'lower' in self.cleaners or 'lower' in cleaners:
            cleaned = cleaned.lower()
        if 'url' in self.cleaners or 'url' in cleaners:
            cleaned = re.sub(r'https?:/\/\S+', 'URL', cleaned)
        if 'hashtag' in self.cleaners or 'hashtag' in cleaners:
            cleaned = re.sub(r'#[a-zA-Z0-9]*\b', 'HASHTAG', cleaned)
        if 'username' in self.cleaners or 'username' in cleaners:
            cleaned = re.sub(r'@\S+', 'AT_USER', cleaned)
        cleaned = re.sub("'", ' ', cleaned)

        return cleaned

    def tokenize(self, document: t.DocType, processes: t.List[str] = None):
        """Tokenize the document using SpaCy and clean it as it is processed.
        :param document: Document to be parsed.
        :param processes: The cleaning processes to engage in.
        :return toks: Document that has been passed through spacy's tagger.
        """
        if processes:
            toks = [tok.text for tok in self.tagger(self.clean_document(document, processes = processes))]
        else:
            toks = [tok.text for tok in self.tagger(self.clean_document(document))]
        return toks

    def ptb_tokenize(self, document: t.DocType, processes: t.List[str] = None):
        """Tokenize the document using SpaCy, get PTB tags and clean it as it is processed.
        :param document: Document to be parsed.
        :param processes: The cleaning processes to engage in.
        :return toks: Document that has been passed through spacy's tagger.
        """
        self.processes = processes if processes else self.processes
        toks = [tok.tag_ for tok in self.tagger(self.clean_document(document))]
        return " ".join(toks)

    def sentiment_tokenize(self, document: t.DocType, processes: t.List[str] = None):
        """Tokenize the document using SpaCy, get sentiment and clean it as it is processed.
        :param document: Document to be parsed.
        :param processes: The cleaning processes to engage in.
        :return toks: Document that has been passed through spacy's tagger.
        """
        raise NotImplementedError
        # self.processes = processes if processes else self.processes
        # toks = [tok.sentiment for tok in self.tagger(self.clean_document(document))]
        # pdb.set_trace()
        # return toks

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

    def compute_unigram_liwc(self, doc: t.DocType):
        """Compute LIWC for each document document.
        :param doc (t.DocType): Document to operate on.
        :return liwc_doc (t.DocType): Document represented as LIWC categories.
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
