import logging
from random import sample
from typing import List, Dict
import os
import json

from torch.utils.data import BatchSampler, SubsetRandomSampler, Subset
from tqdm import tqdm
from .util_classes import SentenceIndex, tokens_from_window


class ActiveLearningAgent:

    def __init__(
            self,
            train_set,
            batch_size,
            round_size,
            acquisition_class,
            selector_class,
            helper,
            device,
            propagation_mode,
            budget_prop = 0.5
    ):
        """
        train_set: loaded from pickle
        test_data: loaded from pickle

        score: function used to score single words
        Inputs:
            output: Tensor, shape (batch size, sequence length, number of possible tags), model outputs of all instances
        Outputs:
            a score, with higher meaning better to pick

        budget: total number of elements we can label (words)
        round_size: total number instances we label each round (sentences)
        """

        self.round_size = round_size
        self.batch_size = batch_size
        train_set = list(train_set)
        self.train_set = train_set
        self.temporary_train_set = train_set.copy()
        self.acquisition = acquisition_class
        self.selector = selector_class
        self.selector.assign_agent(self)
        self.helper = helper
        self.device = device
        self.propagation_mode = propagation_mode
        self.budget_prop = budget_prop

        # Dictionaries mapping {sentence idx: [list, of, word, idx]} for labelled and unlabelled words
        self.index = SentenceIndex(self)

        num_tokens = sum([len(sentence) for sentence, _, _ in self.train_set])
        self.budget = num_tokens * budget_prop
        self.initial_budget = self.budget

        self.unlabelled_set = None
        self.labelled_set = None
        self.num = 1
        self.round_all_word_scores = {}

    def init(self, n):
        logging.info('starting random init')
        self.random_init(n)
        self.update_datasets()
        self.num = 0
        logging.info('finished random init')

    def step(self):
        logging.info('step')
        sentence_scores: Dict[int, List[float]] = self.get_sentence_scores()
        self.update_index(sentence_scores)
        self.update_datasets()
        logging.info('finished step')

    def budget_spent(self):
        return self.initial_budget - self.budget

    def save(self, save_path):
        self.index.save(save_path)
        self.selector.save(save_path)
        with open(os.path.join(save_path, "all_word_scores_no_nan.json"), "w") as f:
            json.dump(self.round_all_word_scores, f)

    def random_init(self, num_sentences):
        """
        Randomly initialise self.labelled_idx dictionary
        """
        randomly_selected_indices = sample(list(self.index.unlabelled_idx.keys()), num_sentences)

        budget_spent = 0
        for i in randomly_selected_indices:
            self.index.label_sentence(i)
            budget_spent += len(self.train_set[i][0])

        self.budget -= budget_spent

        logging.info(
            f"""
            total sentences: {len(self.train_set)}  |   total words: {self.budget + budget_spent}
            initialised with {budget_spent} words  |   remaining word budget: {self.budget}
            """)

    def get_batch(self, i):
        # Use selector get_batch here as we want to fill things in if needed
        batch = [self.temporary_labels(j) for j in self.labelled_set[i]]
        # The window selector needs the batch indices and index sets
        return self.selector.get_batch(batch=batch, batch_indices=self.labelled_set[i], agent=self)

    def temporary_labels(self, i):
        temp_tokens = self.train_set[i][0]
        temp_letters = self.train_set[i][1]
        temp_labels = [self.train_set[i][-1][j] if j in self.index.labelled_idx[i]
                       else self.temporary_train_set[i][-1][j] for j in range(len(temp_tokens))]
        return temp_tokens, temp_letters, temp_labels

    def update_index(self, sentence_scores):
        """
        After a full pass on the unlabelled pool, apply a policy to get the top scoring phrases and add them to
        self.labelled_idx.

        Input:
            sentence_scores: {j: [list, of, scores, per, word, nan, nan]} where nan means the word has alread been
            labelled i.e. full list of scores/Nones
        Output:
            No output, but extends self.labelled_idx:
            {
                j: [5, 6, 7],
                i: [1, 2, 3, 8, 9, 10, 11],
                ...
            }
            meaning words 5, 6, 7 of word j are chosen to be labelled.
        """
        logging.info("update index")

        window_scores = []
        for i, word_scores in tqdm(sentence_scores.items()):
            # Skip if already all labelled
            if self.index.is_labelled(i):
                continue
            windows = self.selector.score_extraction(word_scores)
            window_scores.extend([(i, window[0], window[1]) for window in windows])

        window_scores.sort(key=lambda e: e[-1], reverse=True)
        best_window_scores, labelled_ngrams_lookup, budget_spent = \
            self.selector.select_best(window_scores, self.propagation_mode != 0)
        self.budget -= budget_spent
        if self.budget < 0:
            logging.warning('no more budget left!')

        labelled_ngrams_lookup = {k: v for k,v in labelled_ngrams_lookup.items() if sum(v)}

        # Simplify this whole section pls

        total_tokens = 0

        for i, r, _ in best_window_scores:
            cost = r[1] - r[0]
            total_tokens += cost
            self.index.label_window(i, r)

        manual_cost = total_tokens
        assert manual_cost == budget_spent

        if self.propagation_mode:
            # This must come after labelling initial set
            propagated_windows = self.propagate_labels(window_scores, labelled_ngrams_lookup)

            for i, r, _ in propagated_windows:
                cost = r[1] - r[0]
                total_tokens += cost
                self.index.temporarily_label_window(i, r)

        logging.info(f'added {total_tokens} words to index mapping, of which {budget_spent} manual')

        # No more windows of this size left
        if manual_cost < self.round_size:
            self.selector.reduce_window_size()

    def propagate_labels(self, window_scores, labelled_ngrams_lookup):

        out_windows = []

        for window in window_scores:
            if self.index.new_window_unlabelled(window):
                tokens = tokens_from_window(window, self.train_set)
                if tokens in labelled_ngrams_lookup.keys():
                    out_windows.append(window)
                    self.alter_temp_train_set(window, labelled_ngrams_lookup[tokens])

        return out_windows

    def alter_temp_train_set(self, window, new_labels):
        sidx, r, _ = window
        self.temporary_train_set[sidx][-1][r[0]:r[1]] = new_labels

    def get_sentence_scores(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """

        if self.budget <= 0:
            logging.warning('no more budget left!')

        sentence_scores_no_nan = {}
        logging.info('get sentence scores')
        for batch_index in tqdm(self.unlabelled_set):
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter
            # for functionality
            batch = [self.train_set[i] for i in batch_index]
            sentences, tokens, _, lengths = [a.to(self.device) for a in self.helper.get_batch(batch)]
            batch_scores = self.acquisition.score(sentences=sentences, lengths=lengths, tokens=tokens)
            for j, i in enumerate(batch_index):
                sentence_scores_no_nan[i] = batch_scores[j].tolist()

        sentence_scores = {}
        for i, scores in sentence_scores_no_nan.items():
            sentence_scores[i] = self.index.make_nan_if_labelled(i, scores)

        self.round_all_word_scores = sentence_scores_no_nan
        return sentence_scores

    def update_datasets(self):
        unlabelled_sentences = set()
        labelled_sentences = set()

        logging.info("update datasets")
        for i in tqdm(range(len(self.train_set))):
            if self.index.is_partially_unlabelled(i):
                unlabelled_sentences.add(i)
            if self.propagation_mode == 2:
                if self.index.is_partially_labelled(i):
                    labelled_sentences.add(i)
            else:
                if self.index.has_any_labels(i):
                    labelled_sentences.add(i)

        unlabelled_subset = Subset(self.train_set, list(unlabelled_sentences))
        labelled_subset = Subset(self.train_set, list(labelled_sentences))

        self.unlabelled_set = \
            list(BatchSampler(SubsetRandomSampler(unlabelled_subset.indices), self.batch_size, drop_last=False))

        self.labelled_set = \
            list(BatchSampler(SubsetRandomSampler(labelled_subset.indices), self.batch_size, drop_last=False))

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        if num < 0:
            raise StopIteration
        if num > 0:
            self.step()
        if self.budget <= 0:
            self.num = -1
        return self.budget
