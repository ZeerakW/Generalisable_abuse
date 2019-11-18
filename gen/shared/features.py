import re
import pdb
from nltk import ngrams
import gen.shared.types as t
from string import punctuation
from collections import Counter


def unigrams(doc: t.List[str]) -> t.Dict[str, int]:
    """Compute unigrams.
    :param doc: Document to be parsed.
    :return: counted ngrams.
    """
    return Counter(doc)


def n_grams(doc: t.Union[str, t.List[str]], n: int = 2) -> t.Dict[str, int]:
    """
    :param doc: Document to create ngrams from.
    :param n: Size of ngrams.
    :return: Counter object of counted token bigrams.
    """
    return Counter(["_".join(gram) for gram in ngrams(doc, n)]) + unigrams(doc)


def char_ngrams(doc: str, n: int = 2) -> t.Dict[str, int]:
    """
    :param doc: Document to character create ngrams from.
    :return: Counter object of counted char ngrams.
    """
    return Counter(["_".join(gram) for gram in ngrams(doc, 2)])


def char_count(doc: str) -> t.Dict[str, int]:
    """
    :param doc: Document to create character counts from.
    :return: Counter dict of counted characters in document.
    """
    return Counter(doc)


def find_mentions(doc: str) -> t.List[str]:
    """
    :param doc: Document to find mentions in.
    :return: t.List of mentions.
    """
    return re.findall(r'@[a-zA-Z0-9]', doc)


def find_hashtags(doc: str) -> t.List[str]:
    """
    :param doc: Document to find hashtags in.
    :return: t.List of hashtags.
    """
    return re.findall(r'#[a-zA-Z0-9]', doc)


def find_urls(doc: str) -> t.List[str]:
    """
    :param doc: Document to find URLs in.
    :return: t.List of urls.
    """
    return re.findall(r'http:\/\/[a-zA-Z0-9\.\/a-zA-Z0-9]+', doc)


def find_retweets(doc: str) -> t.List[str]:
    """
    :param doc: Document to find retweets in.
    :return: t.List of retweets.
    """
    return re.findall(r'\wRT\w @', doc)


def count_syllables(doc: t.List[str]) -> int:
    """Simplistic syllable count.
    :param doc: Document to count syllables in.
    :return: Count of syllables.
    """
    count = 0
    vowels = 'aeiouy'
    exceptions = ['le', 'es', 'e']
    for word in doc:
        prev_char = None
        for i in word:
            if i == len(word) and (prev_char + word[i] in exceptions or word[i] in exceptions):
                prev_char = word[i]
                continue
            if (word[i] in vowels) and (prev_char not in vowels and not prev_char):
                prev_char = word[i]
                count += 1
    return {'NO_SYLLABLES': count}


def word_list(doc: t.List[str], word_list: t.List[str], salt: str) -> t.Dict[str, int]:
    """Identify if words are in word_list.
    :param doc: Tokenised document.
    :param word_list: t.List containing words occurring in the wordlist.
    :param salt: To add in front of word name.
    :return: Counts of each word appearing in dict.
    """
    salt += '_WORDLIST_'
    res = []
    for w in doc:
        if w in word_list:
            res.append(salt + w)
    return Counter(res) if len(res) != 0 else {salt: 0}


def _pos_helper(docs: t.List[str]) -> t.Tuple[t.List[str], t.List[str], t.List[str]]:
    # for doc in tqdm(docs, desc = "POS helper"):
    for doc in docs:
        tokens = []
        pos = []
        confidence = []
        for tup in doc:
            tokens.append(tup[0])
            pos.append(tup[1])
            confidence.append(tup[2])
        yield tokens, pos, confidence


def sentiment_polarity(doc: str, sentiment: t.Callable) -> t.Dict[str, float]:
    """Compute sentiment polarity scores and return features.
    :param doc: Document to be computed for.
    :param sentiment: t.Callable sentiment analysis method.
    :return features: Features dict to return.
    """
    features = {}
    polarity = sentiment.polarity_scores(doc)
    features.update({'SENTIMENT_POS': polarity['pos']})
    features.update({'SENTIMENT_NEG': polarity['neg']})
    features.update({'SENTIMENT_COMPOUND': polarity['compound']})

    return features


def head_of_token(parsed):
    """Retrieve the head of the current token.
    :param parsed: The parsed document by spacy.
    """
    return {"HEAD_OF_{0}".format(token): token.head.text for token in parsed}


def children_of_token(parsed):
    """Retrieve the children of the current token.
    :param parsed: The parsed document by spacy.
    """
    return {"children_of_{0}".format(token): "_".join([str(child) for child in token.children])
            for token in parsed}


def number_of_arcs(parsed):
    """Retrieve the number of right and left arcs.
    :param parsed: The parsed document by spacy.
    """
    features = {}
    for token in parsed:
        arcs = {"NO_RIGHT_ARCS_{0}".format(token): token.n_rights,
                "NO_LEFT_ARCS_{0}".format(token): token.n_lefts,
                "NO_TOTAL_ARCS_{0}".format(token): int(token.n_rights) + int(token.n_lefts)}
        features.update(arcs)
    return features


def arcs(parsed):
    """Retrieve the right and left arcs.
    :param parsed: The parsed document by spacy.
    """
    features = {}
    for token in parsed:
        arcs = {"RIGHT_ARCS_{0}".format(token): "_".join([arc.text for arc in token.rights]),
                "LEFT_ARCS_{0}".format(token): "_".join([arc.text for arc in token.lefts])}
        features.update(arcs)
    return features


def get_brown_clusters(doc: t.List[str], cluster: t.Dict[str, str], salt: str = '') -> t.List[str]:
    """Generate cluster for each word.
    :param doc: Document ebing procesed as a list.
    :param cluster: Cluster computed using clustering algorithm.
    :param salt: To add in front of the features.
    :return: t.Dictionary of clustered values."""
    if salt != '':
        salt = salt.upper() + '_'
    return Counter([salt + cluster.get(w, 'CLUSTER_UNK') for w in doc])


def read_liwc() -> dict:
    with open('/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/liwc-2015.csv', 'r') as liwc_f:
        liwc_dict = {}
        for line in liwc_f:
            k, v = line.strip('\n').split(',')
            if k in liwc_dict:
                liwc_dict[k] += [v]
            else:
                liwc_dict.update({k: [v]})

    return liwc_dict


def compute_unigram_liwc(doc: t.DocType):
    """Compute LIWC for each document document.
    :param doc (t.DocType): Document to operate on.
    :return liwc_doc (t.DocType): Document represented as LIWC categories.
    """
    # TODO modify
    liwc_doc = []
    kleene_star = [k[:-1] for k in liwc_dict if k[-1] == '*']

    if isinstance(doc, str):
        doc = [w if w[0] not in punctuation and w[-1] not in punctuation else w.strip(punctuation) for w in doc.split()]

    for w in doc:
        if w in liwc_dict:
            liwc_doc.append(liwc_dict[w][0])
        else:
            # This because re.findall is slow.
            candidates = [r for r in kleene_star if r in w]  # Find all candidates
            num_cands = len(candidates)
            if num_cands == 0:
                term = 'NUM' if re.findall(r'[0-9]+', w) else 'UNK'
            elif num_cands == 1:
                term = candidates[0] + '*'
            elif num_cands > 1:
                sorted_cands = sorted(candidates, key=len, reverse = True)  # Longest first
                term = sorted_cands[0] + '*'
            if term == 'UNK' or 'NUM':
                liwc_doc.append(term)
            else:
                liwc_term = liwc_dict[term]
                if isinstance(liwc_term, list):
                    term = "_".join(liwc_term)
                else:
                    term = liwc_term
                liwc_doc.append(term)
    try:
        assert(len(liwc_doc) == len(doc))
    except AssertionError:
        pdb.set_trace()

    return " ".join(liwc_doc)


if __name__ == "__main__":
    global liwc_dict
    liwc_dict = read_liwc()
