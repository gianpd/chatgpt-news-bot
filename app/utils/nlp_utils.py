import os
import sys
import logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)-15s %(message)s',
                level=logging.INFO, datefmt=None)
logger = logging.getLogger("Summarizer")

# from typing import List, Dict, Optional
# from functools import lru_cache

# from .config import get_settings

# from transformers import BartTokenizerFast
from huggingface_hub.inference_api import InferenceApi

# ### instantiate the hugging face hub inference
# Config = get_settings()
# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn" 

# logger.info(f"HF_TOKEN: {Config.hf_token}")
inference = InferenceApi(repo_id="facebook/bart-large-cnn", token=os.getenv("HF_TOKEN"))

import spacy
NLP = spacy.load("en_core_web_sm")

# @lru_cache
# def load_tokenizer(tokenizer_model: str = 'facebook/bart-large-cnn'):
#     return BartTokenizerFast.from_pretrained(tokenizer_model)

# tokenizer = load_tokenizer()


def get_summaries_from_hf_api(text: str) -> str:
    """
    Get an abstractive summary from bart-large-cnn, calling the HF API.
    
    --Parameters
     - text: (str) the string text to be summarized. It comes from an exctractive summary pipeline, therefore its lenght is below 1024 tokens.

     return (str) a string contained the total abstractive summary.

     """
    summary = inference(text)
    logger.info(f"Abstractive Pipeline: get summary {summary}")
    try:
        summary = summary[0]['summary_text']
        return summary
    except KeyError as e:
        logger.info(e)
        return str(e)



# def get_nest_sentences(document: str, tokenizer: BartTokenizerFast, token_max_length = 1024) -> List[str]:
#     """
#     Starting from a large document, a list of sequential string is computed, such that each string has
#     a number of tokens equal to token_max_length.

#     ---Params
#     - document: the long text (str)
#     - tokenizer: the pre-trained tokenizer to be used.
#     - token_max_length: the maximum number of token has required by the NLP model (int)
#     """
#     sents = []
#     length = 0
#     doc = NLP(document)
#     s = ''
#     for sentence in doc.sents:
#         tokens_in_sentence = tokenizer(str(sentence), truncation=False, padding=False)[0]
#         length += len(tokens_in_sentence) # how many tokens the current sentence have summed to the previous
#         if length <= token_max_length:
#             s += sentence.text
#         else:
#             sents.append(s)
#             s = sentence.text
#             length = 0
#     # append last string with less # of tokens than token_max_length
#     sents.append(s)
#     logger.info(f'Returning {len(sents)} number of chunk strings')
#     return sents