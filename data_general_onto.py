import logging
from typing import List, Dict
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

available_chars = [
    '!', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '·', '▲', '・', " "
]


def filter_text(text):

    for character in text:
        if character.lower() not in available_chars:
            return False

    if '\t' in text or len(text) == 0:
        return False

    return True


def construct_data_dictionary_string(sentence_df: pd.DataFrame, token_col: str, label_col: str):

    sentence_df = sentence_df[1:].reset_index(inplace=False)
    label_list = [
            {
                "start": i,
                "label": label,
                "text": sentence_df[token_col][i]
            } for i, label in enumerate(sentence_df[label_col])
        if label != 'O'
    ]
    
    data_dict = {
        "sentText": " ".join([str(a) for a in sentence_df[token_col]]),
        "articleId": None,
        "sentId": "1",
        "relationMentions": [],
        "entityMentions": label_list
    }

    # BASIC PURGE
    if not filter_text(data_dict['sentText']):
        return 'ERROR'
    else:
        return json.dumps(data_dict)


def make_dataset_jsons(
        file_mappings: Dict[str, str],
        col_names: List[str],
        token_col: str,
        label_col: str,
        target_vocab_file: str,
    ):
    token_counters = Counter({})

    for fin, fout in file_mappings.items():

        with open(fout, "w") as j_file:

            df = pd.read_csv(fin, sep='\t', skip_blank_lines=False, names=col_names, error_bad_lines=False, engine='python')
            sentence_list = np.split(df, df[df.isnull().all(1)].index)

            for sentence_df in tqdm(sentence_list):

                data_dictionary_string = construct_data_dictionary_string(sentence_df, token_col, label_col)
                if data_dictionary_string == "ERROR":
                    continue

                j_file.write(f"{data_dictionary_string}\n")

                token_series = sentence_df[token_col].dropna().map(lambda x: x.lower()).to_numpy()
                tokens, counts = np.unique(token_series, return_counts=True)
                d = {k: v for k, v in dict(zip(tokens, counts)).items() if filter_text(k)}
                token_counters += Counter(d)

    token_counters = {k: v for k, v in sorted(token_counters.items(), key=lambda item: item[1], reverse=True)}

    with open(target_vocab_file, "w") as vocab_txt:
        for token, count in token_counters.items():
            vocab_txt.write(f"{token}\t{count}\n")


if __name__ == '__main__':

    col_names = ["tokens", "POS", "LING", "NER"]
    token_col = "tokens"
    dataset_json_mappings = {
        '/home/radmard/repos/AL4ST/data/OntoNotes-5.0/onto.test.ner':
            '/home/radmard/repos/AL4ST/data/OntoNotes-5.0/NER/test.json',
        '/home/radmard/repos/AL4ST/data/OntoNotes-5.0/onto.train.ner':
            '/home/radmard/repos/AL4ST/data/OntoNotes-5.0/NER/train.json'
    }
    corpus_files=list(dataset_json_mappings.keys())
    label_col = "NER"
    vocab_txt = '/home/radmard/repos/AL4ST/data/OntoNotes-5.0/NER/vocab.txt'
    tags_txt = '/home/radmard/repos/AL4ST/data/OntoNotes-5.0/NER/tag2id.txt'

    logging.info("Started making dataset jsons")
    make_dataset_jsons(
        file_mappings=dataset_json_mappings,
        col_names=col_names,
        token_col=token_col,
        label_col=label_col,
        target_vocab_file=vocab_txt,
    )
    logging.info("Finished making dataset jsons")
