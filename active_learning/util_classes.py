import json
import os


def tokens_from_window(_window, _train_set):
    s_idx, [i, j], _ = _window
    return tuple(_train_set[s_idx][0][i:j])

def labels_from_window(_window, _train_set):
    s_idx, [i, j], _ = _window
    return tuple(_train_set[s_idx][-1][i:j])


class SentenceIndex:

    def __init__(self, agent):
        self.__number_partially_labelled_sentences = 0
        self.labelled_idx = {j: set() for j in range(len(agent.train_set))}
        self.unlabelled_idx = {j: set(range(len(agent.train_set[j][0]))) for j in range(len(agent.train_set))}
        self.temp_labelled_idx = {j: set() for j in range(len(agent.train_set))}

    def label_sentence(self, i):
        self.labelled_idx[i] = self.unlabelled_idx[i]
        self.__number_partially_labelled_sentences += 1
        self.unlabelled_idx[i] = set()

    def label_window(self, i, r):
        if not self.labelled_idx[i] and r[1] - r[0] > 0:
            self.__number_partially_labelled_sentences += 1
        self.labelled_idx[i].update(range(r[0], r[1]))
        self.unlabelled_idx[i] -= set(range(r[0], r[1]))
        self.temp_labelled_idx[i] -= set(range(r[0], r[1]))

    def temporarily_label_window(self, i, r):
        self.unlabelled_idx[i] -= set(range(r[0], r[1]))
        self.temp_labelled_idx[i].update(range(r[0], r[1]))

    def new_window_unlabelled(self, new_window):
        sidx = new_window[0]
        if set(range(new_window[1][0], new_window[1][1])).intersection(self.labelled_idx[sidx]):
            return False
        else:
            return True

    def is_partially_labelled(self, i):
        return len(self.labelled_idx[i]) > 0

    def is_partially_temporarily_labelled(self, i):
        return len(self.temp_labelled_idx[i]) > 0

    def has_any_labels(self, i):
        return self.is_partially_labelled(i) or self.is_partially_temporarily_labelled(i)

    def is_labelled(self, i):
        return len(self.unlabelled_idx[i]) == 0

    def is_partially_unlabelled(self, i):
        return len(self.unlabelled_idx[i]) > 0

    def get_number_partially_labelled_sentences(self):
        return self.__number_partially_labelled_sentences

    def make_nan_if_labelled(self, i, scores):
        res = []
        for j in range(len(scores)):
            if j in self.labelled_idx[i]:
                res.append(float('nan'))
            else:
                res.append(scores[j])
        return res

    def save(self, save_path):
        with open(os.path.join(save_path, "agent_index.pk"), "w") as f:
            json.dump(
                {
                    "labelled_idx": {k: list(v) for k, v in self.labelled_idx.items()},
                    "unlabelled_idx": {k: list(v) for k, v in self.unlabelled_idx.items()},
                    "temporarily_labelled_idx": {k: list(v) for k, v in self.temp_labelled_idx.items()},
                },
                f
            )


class BeamSearchSolution:
    def __init__(self, windows, max_size, B, labelled_ngrams, init_size=None, init_score=None, init_overlap_index={}):
        self.windows = windows
        if not init_score:
            self.score = sum([w[-1] for w in windows])
        else:
            self.score = init_score
        if not init_size:
            self.size = sum([w[1][1] - w[1][0] for w in windows])
        else:
            self.size = init_size
        self.overlap_index = init_overlap_index
        self.max_size = max_size
        self.lock = False
        self.B = B
        self.labelled_ngrams = labelled_ngrams

    def add_window(self, new_window, train_set):
        if self.size >= self.max_size:
            self.lock = True
            return self
        init_size = self.size + new_window[1][1] - new_window[1][0]
        init_score = self.score + new_window[-1]
        init_overlap_index = self.overlap_index.copy()
        if new_window[0] in init_overlap_index:
            init_overlap_index[new_window[0]] = init_overlap_index[new_window[0]].union(set(range(*new_window[1])))
        else:
            init_overlap_index[new_window[0]] = set(range(*new_window[1]))
        new_ngram = tokens_from_window(new_window, train_set)
        ngram_annotations = labels_from_window(new_window, train_set)
        self.labelled_ngrams[new_ngram] = ngram_annotations
        return BeamSearchSolution(self.windows + [new_window], self.max_size, self.B, self.labelled_ngrams,
                                  init_size=init_size, init_score=init_score, init_overlap_index=init_overlap_index)

    def is_permutationally_distinct(self, other):
        # We do a proxy-check for permutation invariance by checking for score and size of solutions
        if abs(self.score - other.score) < 1e-6 and self.size == other.size:
            return False
        else:
            return True

    def all_permutationally_distinct(self, others):
        for other_solution in others:
            if not self.is_permutationally_distinct(other_solution):
                return False
        else:
            return True

    def new_window_unlabelled(self, new_window):
        if new_window[0] not in self.overlap_index:
            self.overlap_index[new_window[0]] = set() # Just in case!
            return True
        else:
            new_word_idx = set(range(*new_window[1]))
            if self.overlap_index[new_window[0]].intersection(new_word_idx):
                return False
            else:
                return True

    def branch_out(self, other_solutions, window_scores, train_set, allow_propagation):
        # ASSUME window_scores ALREADY SORTED
        local_branch = []
        for window in window_scores:
            if self.new_window_unlabelled(window):
                new_ngram = tokens_from_window(window, train_set)
                # i.e. if we are allowing automatic labelling and we've already seen this ngram, then skip
                if new_ngram in self.labelled_ngrams.keys() and allow_propagation:
                    continue
                else:
                    possible_node = self.add_window(window, train_set)
                if possible_node.all_permutationally_distinct(other_solutions):
                    local_branch.append(possible_node)
                if len(local_branch) == self.B:
                    return local_branch
            if self.lock:
                return [self]

        # No more windows addable
        if len(local_branch) == 0:
            self.lock = True
            return [self]
        else:
            return local_branch
