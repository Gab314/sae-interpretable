
import pprint

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import random
import time
from transformer_lens import utils

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
language_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

sentence = ["I love chocalate"]
inputs = tokenizer(sentence)
_, cache = language_model.run_with_cache(inputs, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))