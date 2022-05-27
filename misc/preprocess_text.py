
import sys
import json
from tqdm import tqdm

import spacy
from spacy.lang.zh.stop_words import STOP_WORDS

assert sys.argv[1] in ('labeled', 'test_a')

nlp = spacy.load('zh_core_web_sm')
def remove_stopwords(string):
    doc = nlp(string)
    new_string = []
    for token in doc:
        if token.text not in STOP_WORDS:
            new_string.append(token.text)
    return ''.join(new_string)

def concat_text(input_text, preprocess=False):
    filtered = [input_text[0]]
    for i in range(1, len(input_text)):
        text = input_text[i]
        if preprocess:
            text = remove_stopwords(text)
        if len(text) > 0:
            filtered.append(text)
    return '。'.join(filtered)

with open(f'data/annotations/{sys.argv[1]}.json', 'r', encoding='utf8') as f:
    anns = json.load(f)

for idx in tqdm(range(len(anns))):
    anns[idx]['asr'] = anns[idx]['asr'][:128]
    input_text = [anns[idx]['title'], anns[idx]['asr']]
    input_text.extend([d['text'] for d in anns[idx]['ocr']])
    anns[idx]['input_text'] = concat_text(input_text, preprocess=False)
    anns[idx]['input_text_filtered'] = concat_text(input_text, preprocess=True)
    # print(f'{idx+1}/{len(anns)}')

with open(f'data/annotations/{sys.argv[1]}_preprocessed.json', 'w', encoding='utf8') as f:
    json.dump(anns, f)
