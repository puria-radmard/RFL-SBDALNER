import numpy as np
import torch

class Acquisition:

    def __init__(self, model):
        self.model = model

    def score(self, sentences, lengths, tokens):
        pass


class RandomBaselineAcquisition(Acquisition):

    def __init__(self, model):
        super().__init__(model=model)

    def score(self, sentences, lengths, tokens):
        return [np.random.randn(length) for length in lengths]


class LowestConfidenceAcquisition(Acquisition):

    def __init__(self, model):
        super().__init__(model=model)

    def score(self, sentences, lengths, tokens):
        # logits (batch_size x sent_length x num_tags [193])
        y_hat = self.model(sentences, tokens, anneal=True).detach().cpu()
        scores = -y_hat.max(dim=-1).values  # negative highest logits (batch_size x sent_length)
        return [scores[i, :length].reshape(-1) for i, length in enumerate(lengths)]


class MaximumEntropyAcquisition(Acquisition):

    def __init__(self, model):
        super().__init__(model=model)

    def score(self, sentences, lengths, tokens):
        # logits (batch_size x sent_length x num_tags [193])
        y_hat = self.model(sentences, tokens, anneal=True).detach().cpu()
        scores = torch.sum(-y_hat * torch.exp(y_hat), dim=-1)  # entropies of shape (batch_size x sent_length)
        return [scores[i, :length].reshape(-1) for i, length in enumerate(lengths)]


class BALDAcquisition(Acquisition):
    """
        SUSPENDED FOR NOW
    """

    def __init__(self, m):
        # We might want to set a range of ps and change them using model.*.p = p[i] during the M runs
        self.m = 100
        super().__init__()

    def score(self, sentences, tokens, model):
        # Do the M forward passes:
        model.char_encoder.drop.train()
        model.word_encoder.drop.train()
        model.drop.train()

        dropout_model_preds = [model(sentences, tokens).detach().cpu()[0].max(dim=1)[1].numpy() for _ in
                               range(self.m)]  # list of self.M tensors of size (seq_len)
        dropout_model_preds = np.dstack(dropout_model_preds)
        [0]  # Of size (seq_len, self.M)                                # Test scores here
        majority_vote = np.array([np.argmax(np.bincount(dropout_model_preds[:, i])) for i in range(self.M)])
        scores = 1 - np.array(
            [sum(dropout_model_preds[j] == majority_vote[j]) for j in range(dropout_model_preds.shape[0])]
        ) / self.m

        model.eval()

        return scores
