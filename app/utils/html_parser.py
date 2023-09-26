import os
import sys
import logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)-15s %(message)s',
                level=logging.INFO, datefmt=None)
logger = logging.getLogger("Summarizer")


current_dir = os.getcwd()
punkt_dir = './src/punkt' if current_dir == '/usr/src/app' else './punkt'  # if prod lambda or dev lambda environment

import nltk
nltk.data.path.append(punkt_dir) # aws lambda read-only allows to write only to the tmp directory
# nltk.download('punkt', download_dir='/tmp/')
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter, namedtuple
from operator import attrgetter

from typing import List, Optional, Union, Dict


NLP = spacy.load("en_core_web_sm")

def get_lsa_extractive_summary(input_str: str, url: bool = True, sentence_count: Optional[int] = 15, language: Optional[str] = "english") -> str:
    """"Get an exctractive summary using the LSA (Latent Semantic Analysys) algorithm from an URL or from a Text.
    
    --Parameters
     - input_Str (str): the http url of the article to parse or a text.
     - url (bool): if the input_str is an url or not.
     - sentence_count (int): the number of sentences to extract.
     - language (str): the used language for setting the stemmer and getting the stop words

     return (str) the extractive summary as a string
    """
    parser = HtmlParser.from_url(input_str, Tokenizer(language)) if url else PlaintextParser.from_string(input_str, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    extractive_summary = ' '.join([sent._text for sent in summarizer(parser.document, sentence_count)])
    return extractive_summary



def get_significant_words_list(doc: spacy.tokens.doc.Doc) -> List[str]:
    """
    Get a list contained words that are important for the speech (PROPN; ADJ; NOUN; VERB): excluding stop words, punctations
    """
    words = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if (token.text in stopwords or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            words.append(token.text)
    return words

def get_frequency_words(words: List[str]) -> Counter:
    """Get a counter with the frequency of each word normalized to one."""
    freq_word = Counter(words)
    max_freq = freq_word.most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = (freq_word[word] / max_freq)
    return freq_word

def get_sent_strenght(doc: spacy.tokens.doc.Doc, freq_word: Counter) -> Dict:
    """Get a dictionary where the keys are sentence (str) and the values are float indicating the importance score of the sentence, based on most high frequencies words."""
    sent_strenght = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strenght.keys():
                    sent_strenght[sent] += freq_word[word.text]
                else:
                    sent_strenght[sent] = freq_word[word.text]
    return sent_strenght

def get_extractive_summary(sent_strenght: Dict, n_sents: int = 5) -> tuple:
    SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rates"))
    infos = (SentenceInfo(s, o, sent_strenght.get(s))
            for o, s in enumerate(sent_strenght.keys()))

    infos = sorted(infos, key=attrgetter("rates"), reverse=True)[:n_sents]
    infos = sorted(infos, key=attrgetter("order"))
    logger.info(f"Extracted {len(infos)} sentences ...")
    return tuple(i.sentence.text for i in infos)


def extractive_summary_pipeline(doc: str, n_sents=5) -> str:
    """Get a final summary of a doc, using a number n_sents of top sentences."""
    doc = NLP(doc)
    logger.info(f"Starting to compute summary from {len(list(doc.sents))} sentences ...")
    words = get_significant_words_list(doc)
    freq_word = get_frequency_words(words)
    sent_strenght = get_sent_strenght(doc, freq_word)

    summaries = get_extractive_summary(sent_strenght, n_sents=n_sents)
    summary = ' '.join(summaries)
    start_sentence = list(doc.sents)[0].text
    if start_sentence in summaries:
        return summary
    return start_sentence + summary
