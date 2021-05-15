import os
import json
import re
import sys
import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec


def func(fin, fout):
    for line in fin:
        line = line.strip()
        if not line:
            continue
        sentence = json.loads(line)
        sentence = sentence["sentText"].strip().strip('"').lower()
        fout.write(sentence + "\n")


def make_corpus():
    # print("-------------haha")
    with open(os.path.join(root_dir, "corpus.txt"), "wt", encoding="utf-8") as fout:
        with open(os.path.join(root_dir, "train.json"), "rt", encoding="utf-8") as fin:
            func(fin, fout)
        with open(os.path.join(root_dir, "test.json"), "rt", encoding="utf-8") as fin:
            func(fin, fout)


def save_numpy_array(root_dir, wv):
    """
    Save word vectors in a .npy array in order they appear in vocab.txt
    """
    with open(os.path.join(root_dir, "vocab.txt")) as vocab_file:

        vocab_file_string = vocab_file.read()
        lines = vocab_file_string.split("\n")[:-1]
        vocab = [re.compile("\s[0-9]{1,}$").split(a)[0] for a in lines]
        print(f"{len(vocab)} words in vocab.txt || {len(wv.vocab)} words in W2V vocab")
        print(f"i.e. {len(vocab) - len(wv.vocab)} missing from W2V")

    word_matrix = np.stack([wv[w] for w in vocab if w in wv.vocab.keys()], axis=0)

    with open(os.path.join(root_dir, "word2vec.vectors.npy"), 'wb') as npy_file:

        np.save(npy_file, word_matrix)


if __name__ == "__main__":

    root_dir = sys.argv[1]  # Fix this later
    # e.g. "data/NYT_CoType"

    if not os.path.exists(os.path.join(root_dir, "corpus.txt")):
        make_corpus()
    print("Made corpus")

    sentences = LineSentence(os.path.join(root_dir, "corpus.txt"))
    print("Made sentences")

    model = Word2Vec(sentences, sg=1, size=300, workers=4, iter=8, negative=8, min_count=1)
    print("Made model")

    word_vectors = model.wv
    print("Made WVs")

    word_vectors.save(os.path.join(root_dir, "word2vec"))
    save_numpy_array(root_dir, wv=word_vectors)
    print("Saved WVs")

    word_vectors.save_word2vec_format(
        os.path.join(root_dir, "word2vec.txt"), fvocab=os.path.join(root_dir, "vocab.txt")
    )
    print("Complete!")
