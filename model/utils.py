import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence


class Helper:

    def __init__(self, vocab, tag_set, charset, measure_type):
        self.vocab = vocab
        self.tag_set = tag_set
        self.charset = charset
        self.measure_type = measure_type

    def get_batch(self, batch):
        sentences, tokens, tags = zip(*batch)

        padded_sentences, lengths = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in sentences], enforce_sorted=False),
            batch_first=True,
            padding_value=self.vocab["<pad>"],
        )
        padded_tokens, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tokens], enforce_sorted=False),
            batch_first=True,
            padding_value=self.charset["<pad>"],
        )
        padded_tags, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tags], enforce_sorted=False),
            batch_first=True,
            padding_value=self.tag_set["O"],
        )

        padded_tags = nn.functional.one_hot(padded_tags, num_classes=len(self.tag_set)).float()

        return padded_sentences, padded_tokens, padded_tags, lengths

    def measure(self, output, targets, lengths):
        assert output.size(0) == targets.size(0) and targets.size(0) == lengths.size(0)
        tp = 0
        tp_fp = 0
        tp_fn = 0
        batch_size = output.size(0)
        output = torch.argmax(output, dim=-1)
        targets = torch.argmax(targets, dim=-1)

        if self.measure_type == 'relations':
            for i in range(batch_size):
                length = lengths[i]
                out = output[i][:length].tolist()
                target = targets[i][:length].tolist()
                out_triplets = self.get_triplets(out, self.tag_set)
                tp_fp += len(out_triplets)
                target_triplets = self.get_triplets(target, self.tag_set)
                tp_fn += len(target_triplets)
                for target_triplet in target_triplets:
                    for out_triplet in out_triplets:
                        if out_triplet == target_triplet:
                            tp += 1

        elif self.measure_type == 'entities':
            for i in range(batch_size):
                length = lengths[i]
                p = output[i][:length].cpu()
                t = targets[i][:length].cpu()
                tp_ = int(sum(np.equal(p, t)[t != 0]))        # Exact match by word and is not 'O'
                fn_ = int(sum(t[p == 0] > 0))                 # Predictions are 'O' but targets are not
                fp_ = int(sum(1 - np.equal(p, t)[p != 0]))    # Predictions are not 'O' and are wrong
                tp += tp_
                tp_fp += tp_ + fp_
                tp_fn += tp_ + fn_

        return tp, tp_fp, tp_fn

    def get_triplets(self, tags, tag_set):
        temp = {}
        triplets = []
        for idx, tag in enumerate(tags):
            if tag == tag_set["O"]:
                continue
            pos, relation_label, role = tag_set[tag].split("-")
            if pos == "B" or pos == "S":
                if relation_label not in temp:
                    temp[relation_label] = [[], []]
                temp[relation_label][int(role) - 1].append(idx)
        for relation_label in temp:
            role1, role2 = temp[relation_label]
            if role1 and role2:
                len1, len2 = len(role1), len(role2)
                if len1 > len2:
                    for e2 in role2:
                        idx = np.argmin([abs(e2 - e1) for e1 in role1])
                        e1 = role1[idx]
                        triplets.append((e1, relation_label, e2))
                        del role1[idx]
                else:
                    for e1 in role1:
                        idx = np.argmin([abs(e2 - e1) for e2 in role2])
                        e2 = role2[idx]
                        triplets.append((e1, relation_label, e2))
                        del role2[idx]
        return triplets