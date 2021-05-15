import argparse
import datetime
import os
import sys
import time
import json
from tqdm import tqdm

import torch
from torch import optim
from torch._utils import _accumulate

from active_learning.helper import configure_al_agent
from model.ner_model import Model
from model.utils import Helper
from training_utils import *
from utils import Charset, Vocabulary, Index, load, time_display


def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths)).tolist()
    return indices, [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def configure_logger():
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).addHandler(TqdmLoggingHandler())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint Extraction of Entities and Relations"
    )
    parser.add_argument(
        "-W", "--window", nargs='+',
        help="size of window acquired each time in words. set -1 for full sentence, two for range",
        required=True
    )
    parser.add_argument(
        "-A", "--acquisition", type=str, help="acquisition function used by agent. choose from 'rand' and 'lc'",
        required=True
    )
    parser.add_argument(
        "-I", "--initprop", type=float,
        help="proportion of sentences of training set labelled before first round. [0,1]",
        default=0.1
    )
    parser.add_argument(
        "-R", "--roundsize", type=int,
        help="number of words acquired made per round (rounded up to closest possible each round)", default=80000
    )
    parser.add_argument(
        "-propagation", "--propagation_mode", type=int,
        help="0 = no propagation, 1 = propagation, training on all sentences, 2 = propagation, training only on sentences with some real labels"
    )
    parser.add_argument(
        "--beta", type=float, help="Weight (should be in [0,1]) that self-supervised losses are multiplied by",
        required=True
    )
    parser.add_argument(
        "-alpha", "--alpha", type=float,
        help="sub-sequence are normalised by L^-alpha where L is the subsequence length",
        required=True
    )
    parser.add_argument(
        "-T", "--temperature", type=float,
        help="Temperature of scoring annealing. Does not affect W=-1 and beta=0 cases", required=True
    )
    parser.add_argument("-B", "--beam_search", type=int, default=1,
                        help="Beam search parameter. B=1 means a greedy search")
    parser.add_argument("-D", "--data_path", type=str, default="/home/pradmard/repos/data/OntoNotes-5.0/NER/")
    # parser.add_argument(
    #     "--labelthres", type=float, help="proportion of sentence that must be manually labelled before it is used
    #     for training", required = True
    # )

    parser.add_argument(
        "--earlystopping", type=int, help="number of epochs of F1 decrease before early stopping", default=2
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 32)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="use CUDA (default: True)")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="dropout applied to layers (default: 0.5)"
    )
    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.25,
        help="dropout applied to the embedded layer (default: 0.25)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.35,
        help="gradient clip, -1 means no clip (default: 0.35)",
    )
    # debug
    parser.add_argument(
        "--epochs", type=int, default=30, help="upper epoch limit (default: 30)"
    )
    parser.add_argument(
        "--char_kernel_size",
        type=int,
        default=3,
        help="character-level kernel size (default: 3)",
    )
    parser.add_argument(
        "--word_kernel_size",
        type=int,
        default=3,
        help="word-level kernel size (default: 3)",
    )
    parser.add_argument(
        "--emsize", type=int, default=50, help="size of character embeddings (default: 50)"
    )
    parser.add_argument(
        "--char_layers",
        type=int,
        default=3,
        help="# of character-level convolution layers (default: 3)",
    )
    parser.add_argument(
        "--word_layers",
        type=int,
        default=3,
        help="# of word-level convolution layers (default: 3)",
    )
    parser.add_argument(
        "--char_nhid",
        type=int,
        default=50,
        help="number of hidden units per character-level convolution layer (default: 50)",
    )
    parser.add_argument(
        "--word_nhid",
        type=int,
        default=300,
        help="number of hidden units per word-level convolution layer (default: 300)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="report interval (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1, help="initial learning rate (default: 1)"
    )
    parser.add_argument(
        "--lr_decrease", type=float, default=1,
        help="learning rate annealing factor on non-improving epochs (default: 1)"
    )
    parser.add_argument(
        "--optim", type=str, default="SGD", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--seed", type=int, default=1111, help="random seed (default: 1111)"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=10.0,
        help='manual rescaling weight given to each tag except "O"',
    )
    return parser.parse_args()


def make_root_dir(args, indices):
    rn = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    
    root_dir = os.path.join(
        '.',
        f"{'-'.join(sys.argv[1:])}--{rn}".replace("/", "")
    )
    os.mkdir(root_dir)

    with open(os.path.join(root_dir, "config.txt"), "w") as config_file:
        config_file.write(f"Run started at: {rn} \n")
        config_file.write(f"\n")
        # config_file.write(f"Number of train sentences {len(train_data)} \n")
        # config_file.write(f"Number of test sentences {len(test_data)} \n")
        config_file.write(f"AFTER 17/12/2020 LOGGING CHANGE - ALL UNUSED WINDOWS INCLUDED\n")
        config_file.write(f"Proportion of initial labelled sentences: {args.initprop} \n")
        config_file.write(f"Number of acquisitions per round: {args.roundsize} \n")
        config_file.write(f"\n")
        config_file.write(f"Acquisition strategy: {args.acquisition} \n")
        config_file.write(f"Size of windows (-1 means whole sentence): {args.window} \n")
        config_file.write(f"\n")
        config_file.write(f"All args: \n")
        for k, v in vars(args).items():
            config_file.write(str(k))
            config_file.write(" ")
            config_file.write(str(v))
            config_file.write("\n")

    with open(os.path.join(root_dir, "dataset_indices.json"), "w") as jfile:
        json.dump(indices, jfile)

    return root_dir


def train_epoch(model, device, agent, start_time, epoch, optimizer, criterion, args):
    model.train()
    total_loss = 0
    count = 0

    sampler = agent.labelled_set
    for idx, batch_indices in enumerate(sampler):

        model.eval()
        sentences, tokens, targets, lengths, self_supervision_mask = [a.to(device) for a in agent.get_batch(idx)]
        model.train()

        optimizer.zero_grad()
        output = model(sentences, tokens)
        # output in shape [batch_size, length_of_sentence, num_tags (193)]

        # output = pack_padded_sequence(output, lengths.cpu(), batch_first=True).data
        # targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True).data

        loss = criterion(output, targets, self_supervision_mask)
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        count += len(targets)

        if (idx + 1) % args.log_interval == 0:
            cur_loss = total_loss / count
            elapsed = time.time() - start_time
            percent = ((epoch - 1) * len(sampler) + (idx + 1)) / (
                    args.epochs * len(sampler)
            )
            remaining = elapsed / percent - elapsed
            logging.info(
                "| epoch {:2d}/{:2d} | batch {:5d}/{:5d} | elapsed time {:s} | remaining time {:s} | "
                "lr {:4.2e} | train_loss {:5.3f} |".format(
                    epoch,
                    args.epochs,
                    idx + 1,
                    len(sampler),
                    time_display(elapsed),
                    time_display(remaining),
                    args.lr,
                    cur_loss,
                )
            )
            total_loss = 0
            count = 0


# NOT CHANGED BY AL
def evaluate(model, data_sampler, dataset, helper, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    tp_total = 0
    tp_fp_total = 0
    tp_fn_total = 0
    with torch.no_grad():
        for batch_indices in data_sampler:
            batch = [dataset[j] for j in batch_indices]
            sentences, tokens, targets, lengths = [a.to(device) for a in helper.get_batch(batch)]

            output = model(sentences, tokens)
            tp, tp_fp, tp_fn = helper.measure(output, targets, lengths)

            tp_total += tp
            tp_fp_total += tp_fp
            tp_fn_total += tp_fn

            loss = criterion(output, targets, 1)
            total_loss += loss.item()

            count += len(targets)
    if tp_fp_total == 0:
        tp_fp_total = 1
    if tp_fn_total == 0:
        tp_fn_total = 1
    if count == 0:
        count = 1
    return (
        total_loss / count,
        tp_total / tp_fp_total,
        tp_total / tp_fn_total,
        2 * tp_total / (tp_fp_total + tp_fn_total)
    )


def train_full(model, device, agent, helper, val_set, val_data_groups, original_lr, criterion, args):
    lr = args.lr
    early_stopper = EarlyStopper(patience=args.earlystopping, model=model)

    all_val_loss = []
    all_val_precision = []
    all_val_recall = []
    all_val_f1 = []

    all_train_loss = []
    all_train_precision = []
    all_train_recall = []
    all_train_f1 = []

    start_time = time.time()

    num_sentences = agent.index.get_number_partially_labelled_sentences()
    num_words = agent.budget_spent()

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=original_lr)

    logging.info(f"Starting training with {num_words} words labelled in {num_sentences} sentences")
    for epoch in range(1, args.epochs + 1):

        train_epoch(model, device, agent, start_time, epoch, optimizer, criterion, args)

        logging.info('beginning evaluation')
        val_loss, val_precision, val_recall, val_f1 = \
            evaluate(model, GroupBatchRandomSampler(val_data_groups, args.batch_size, drop_last=False), val_set,
                     helper, criterion, device)
        train_loss, train_precision, train_recall, train_f1 = \
            evaluate(model, agent.labelled_set, agent.train_set, helper, criterion, device)

        elapsed = time.time() - start_time
        logging.info(
            "| epoch {:2d} | elapsed time {:s} | train_loss {:5.4f} | val_loss {:5.4f} | prec {:5.4f} "
            "| rec {:5.4f} | f1 {:5.4f} |".format(
                epoch, time_display(elapsed), train_loss, val_loss, val_precision, val_recall, val_f1))

        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        if len(all_val_loss) and val_loss > max(all_val_loss):
            lr = lr / args.lr_decrease
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        all_val_loss.append(val_loss)
        all_val_precision.append(val_precision)
        all_val_recall.append(val_recall)
        all_val_f1.append(val_f1)

        all_train_loss.append(train_loss)
        all_train_precision.append(train_precision)
        all_train_recall.append(train_recall)
        all_train_f1.append(train_f1)

        if early_stopper.is_overfitting(val_loss):
            break

    return {
        "num_words": num_words,
        "num_sentences": num_sentences,
        "all_val_loss": all_val_loss,
        "all_val_precision": all_val_precision,
        "all_val_recall": all_val_recall,
        "all_val_f1": all_val_f1,
        "all_train_loss": all_train_loss,
        "all_train_precision": all_train_precision,
        "all_train_recall": all_train_recall,
        "all_train_f1": all_train_f1,
    }


def log_round(root_dir, round_results, agent, test_loss, test_precision, test_recall, test_f1, round_num):
    num_words = round_results["num_words"]
    num_sentences = round_results["num_sentences"]

    round_dir = os.path.join(root_dir, f"round-{round_num}")
    os.mkdir(round_dir)
    logging.info(f"logging round {round_num}")

    agent.save(round_dir)

    with open(os.path.join(round_dir, f"record.tsv"), "wt", encoding="utf-8") as f:
        f.write("Epoch\tT_LOSS\tT_PREC\tT_RECL\tT_F1\tV_LOSS\tV_PREC\tV_RECL\tV_F1\n")
        for idx in range(len(round_results["all_val_loss"])):
            f.write(
                "{:d}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                    idx + 1,
                    round_results["all_train_loss"][idx],
                    round_results["all_train_precision"][idx],
                    round_results["all_train_recall"][idx],
                    round_results["all_train_f1"][idx],
                    round_results["all_val_loss"][idx],
                    round_results["all_val_precision"][idx],
                    round_results["all_val_recall"][idx],
                    round_results["all_val_f1"][idx],
                )
            )
        f.write(
            "\nTEST_LOSS\tTEST_PREC\tTEST_RECALL\tTEST_F1\n"
        )
        f.write(
            "{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                test_loss, test_precision, test_recall, test_f1
            )
        )
        f.write(
            f"\n{num_words} words in {num_sentences} sentences. Total {len(agent.train_set)} sentences.\n\n"
        )

    with open(
            os.path.join(round_dir, f"sentence_prop-{round_num}.tsv"), "wt", encoding="utf-8"
    ) as f:

        f.write("sent_length\tnum_labelled\tnum_temp_labelled\n")

        i = 0
        for sentence_idx, labelled_idx in agent.index.labelled_idx.items():

            num_unlabelled = len(agent.index.unlabelled_idx[sentence_idx])
            num_temp_labelled = len(agent.index.temp_labelled_idx[sentence_idx])
            f.write(str(len(labelled_idx) + num_unlabelled + num_temp_labelled))
            f.write("\t")
            f.write(str(len(labelled_idx)))
            f.write("\t")
            f.write(str(num_temp_labelled))
            f.write("\n")

    logging.info(f"finished logging round {round_num}")


# TODO: Make this properly, maybe a config file in the dataset path
def get_measure_type(path):
    if "NYT_CoType" in path:
        return 'relations'
    elif "OntoNotes-5.0" in path or 'conll' in path:
        return 'entities'
    else:
        raise NotImplementedError(path)


def load_dataset(path):
    charset = Charset()

    vocab = Vocabulary()
    vocab.load(f"{path}/vocab.txt")

    tag_set = Index()
    tag_set.load(f"{path}/tag2id.txt")

    measure_type = get_measure_type(path)

    tag_set = Index()
    if measure_type == 'relations':
        tag_set.load(f"{path}/tag2id.txt")
    elif measure_type == 'entities':
        tag_set.load(f"{path}/entity_labels.txt")

    helper = Helper(vocab, tag_set, charset, measure_type=measure_type)

    # relation_labels = Index()
    # relation_labels.load(f"{path}/relation_labels.txt")

    train_data = load(f"{path}/train.pk")
    test_data = load(f"{path}/test.pk")

    word_embeddings = np.load(f"{path}/word2vec.vectors.npy")

    return helper, word_embeddings, train_data, test_data, tag_set


def active_learning_train(args):
    # set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning("you have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")

    # TODO: make the path a parameter
    helper, word_embeddings, train_set, test_set, tag_set = load_dataset(args.data_path)

    # CHANGED FOR DEBUG
    val_size = int(0.01 * len(train_set))
    indices, (train_set, val_set) = random_split(train_set, [len(train_set) - val_size, val_size])

    # [vocab[a] for a in test_data[0][0]]   gives a sentence
    # [tag_set[a] for a in test_data[0][2]] gives the corresponding tagseq

    val_data_groups = group(val_set, [10, 20, 30, 40, 50, 60])
    test_data_groups = group(test_set, [10, 20, 30, 40, 50, 60])

    word_embeddings = torch.tensor(word_embeddings)
    word_embedding_size = word_embeddings.size(1)
    pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
    unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
    word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])

    char_channels = [args.emsize] + [args.char_nhid] * args.char_layers

    word_channels = [word_embedding_size + args.char_nhid] + [
        args.word_nhid
    ] * args.word_layers

    weight = [args.weight] * len(helper.tag_set)
    weight[helper.tag_set["O"]] = 1
    weight = torch.tensor(weight).to(device)
    criterion = ModifiedKL(weight)

    model = Model(
        charset_size=len(helper.charset),
        char_embedding_size=args.emsize,
        char_channels=char_channels,
        char_padding_idx=helper.charset["<pad>"],
        char_kernel_size=args.char_kernel_size,
        weight=word_embeddings,
        word_embedding_size=word_embedding_size,
        word_channels=word_channels,
        word_kernel_size=args.word_kernel_size,
        num_tag=len(helper.tag_set),
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        T=args.temperature
    ).to(device)

    agent = configure_al_agent(args, device, model, train_set, helper)
    agent.init(int(len(train_set) * args.initprop))

    # logger
    root_dir = make_root_dir(args, indices)

    round_num = 0
    for _ in agent:

        original_lr = args.lr

        round_results = train_full(model, device, agent, helper, val_set, val_data_groups, original_lr, criterion, args)

        # Run on test data
        test_loss, test_precision, test_recall, test_f1 = evaluate(
            model, GroupBatchRandomSampler(test_data_groups, args.batch_size, drop_last=False), test_set, helper,
            criterion, device)

        logging.info(
            "| end of training | test loss {:5.4f} | prec {:5.4f} "
            "| rec {:5.4f} | f1 {:5.4f} |".format(test_loss, test_precision, test_recall, test_f1)
        )

        log_round(root_dir, round_results, agent, test_loss, test_precision, test_recall, test_f1, round_num)

        round_num += 1


if __name__ == "__main__":
    configure_logger()
    op_args = parse_args()
    active_learning_train(args=op_args)
