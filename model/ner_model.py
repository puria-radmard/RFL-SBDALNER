import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            residual=True,
    ):
        super(ConvBlock, self).__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.activate = nn.ReLU()
        self.residual = residual
        self.down_sample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if residual and in_channels != out_channels
            else None
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(
            self.conv.weight.data, mode="fan_in", nonlinearity="relu"
        )
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)
        if self.down_sample is not None:
            nn.init.kaiming_uniform_(
                self.down_sample.weight.data, mode="fan_in", nonlinearity="relu"
            )
            if self.down_sample.bias is not None:
                self.down_sample.bias.data.fill_(0)

    def forward(self, inputs):
        output = self.activate(self.conv(inputs))
        if self.residual:
            output += self.down_sample(inputs) if self.down_sample else inputs
        return output


class ConvNet(nn.Module):
    def __init__(
            self, channels, kernel_size=3, dropout=0.5, dilated=False, residual=True
    ):
        super(ConvNet, self).__init__()
        num_levels = len(channels) - 1
        layers = []
        for i in range(num_levels):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            dilation = kernel_size ** i if dilated else 1
            padding = (kernel_size - 1) // 2 * dilation
            layers += [
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    residual=residual,
                ),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, inputs):
        return self.net(inputs)


class CharEncoder(nn.Module):
    """
    Input: (batch_size, seq_len)
    Output: (batch_size, conv_size)
    """

    def __init__(
            self,
            char_num,
            embedding_size,
            channels,
            kernel_size,
            padding_idx,
            dropout,
            emb_dropout,
    ):
        super(CharEncoder, self).__init__()
        self.embed = nn.Embedding(char_num, embedding_size, padding_idx=padding_idx)
        self.drop = nn.Dropout(emb_dropout)
        self.conv_net = ConvNet(channels, kernel_size, dropout=dropout)
        self.init_weights()

    def forward(self, inputs):
        seq_len = inputs.size(1)

        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size) -> (batch_size, embedding_size, seq_len)
        embeddings = self.drop(self.embed(inputs)).transpose(1, 2).contiguous()

        # (batch_size, embedding_size, seq_len) -> (batch_size, conv_size, seq_len)
        #  -> (batch_size, conv_size, 1) -> (batch_size, conv_size)
        return F.max_pool1d(self.conv_net(embeddings), seq_len).squeeze()

    def init_weights(self):
        nn.init.kaiming_uniform_(
            self.embed.weight.data, mode="fan_in", nonlinearity="relu"
        )


class WordEncoder(nn.Module):
    """
    Input: (batch_size, seq_len), (batch_size, seq_len, char_features)
    """

    def __init__(self, weight, channels, kernel_size, dropout, emb_dropout):
        super(WordEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(weight, freeze=False)
        self.drop = nn.Dropout(emb_dropout)
        self.conv_net = ConvNet(
            channels, kernel_size, dropout, dilated=True, residual=False
        )

    def forward(self, word_input, char_input):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        #  -> (batch_size, seq_len, embedding_size + char_features)
        #  -> (batch_size, embedding_size + char_features, seq_len)
        embeddings = (
            torch.cat((self.embed(word_input), char_input), 2)
                .transpose(1, 2)
                .contiguous()
        )

        # print("embeddings:----------",embeddings.size())

        # (batch_size, embedding_size + char_features, seq_len) -> (batch_size, conv_size, seq_len)
        conv_out = self.conv_net(self.drop(embeddings))

        # (batch_size, conv_size, seq_len) -> (batch_size, conv_size + embedding_size + char_features, seq_len)
        #  -> (batch_size, seq_len, conv_size + embedding_size + char_features)
        return torch.cat((embeddings, conv_out), 1).transpose(1, 2).contiguous()


# self.char_conv_size+self.word_embedding_size+self.word_conv_size, num_tag


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, NUM_LAYERS):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=NUM_LAYERS)
        self.hidden2label = nn.Linear(hidden_dim, output_size)
        self.init_weight()

    def forward(self, inputs):
        self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(inputs, None)
        y = self.hidden2label(lstm_out)
        return y

    def init_weight(self):
        nn.init.kaiming_uniform_(
            self.hidden2label.weight.data, mode="fan_in", nonlinearity="relu"
        )

    def init_hidden(self, batch_size):
        return (
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
        )


class Model(nn.Module):
    def __init__(
            self,
            charset_size,
            char_embedding_size,
            char_channels,
            char_padding_idx,
            char_kernel_size,
            weight,
            word_embedding_size,
            word_channels,
            word_kernel_size,
            num_tag,
            dropout,
            emb_dropout,
            T
    ):
        super(Model, self).__init__()
        self.char_encoder = CharEncoder(
            charset_size,
            char_embedding_size,
            char_channels,
            char_kernel_size,
            char_padding_idx,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.word_encoder = WordEncoder(
            weight,
            word_channels,
            word_kernel_size,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.char_conv_size = char_channels[-1]
        self.word_embedding_size = word_embedding_size
        self.word_conv_size = word_channels[-1]
        # self.decoder = nn.Linear(self.char_conv_size+self.word_embedding_size+self.word_conv_size, num_tag)
        self.decoder = Decoder(
            self.char_conv_size + self.word_embedding_size + self.word_conv_size,
            self.char_conv_size + self.word_embedding_size + self.word_conv_size,
            num_tag,
            NUM_LAYERS=1,
        )
        self.init_weights()
        self.T = T

    def forward(self, word_input, char_input, anneal=False):
        batch_size = word_input.size(0)
        seq_len = word_input.size(1)
        char_output = self.char_encoder(char_input.reshape(-1, char_input.size(2))).reshape(
            batch_size, seq_len, -1
        )
        word_output = self.word_encoder(word_input, char_output)
        y = self.decoder(word_output)

        if anneal:
            return F.log_softmax(y / self.T, dim=2)
        else:
            return F.log_softmax(y, dim=2)

    def init_weights(self):
        pass
        # self.decoder.bias.data.fill_(0)
        # nn.init.kaiming_uniform_(self.decoder.weight.data, mode='fan_in', nonlinearity='relu')


if __name__ == "__main__":
    word_embeddings = torch.tensor(np.load("data/NYT_CoType/word2vec.vectors.npy"))
    print(word_embeddings.shape)
    dropout = (0.5,)
    emb_dropout = 0.25

    model = Model(
        charset_size=96,
        char_embedding_size=50,
        char_channels=[50, 50, 50, 50],
        char_padding_idx=94,
        char_kernel_size=3,
        weight=word_embeddings,
        word_embedding_size=300,
        word_channels=[350, 300, 300, 300],
        word_kernel_size=3,
        num_tag=193,
        dropout=0.5,
        emb_dropout=0.25,
    )
    print(model)
