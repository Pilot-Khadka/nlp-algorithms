import argparse
import time
import math
import hashlib
import numpy as np
import torch
from tqdm import tqdm

import data

# import model
import my_model
from utils import batchify, get_batch, repackage_hidden
from splitcross import SplitCrossEntropyLoss
from weight_drop import WeightDrop


class LanguageModelTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_reproducibility()
        self.corpus = self.load_corpus()
        self.train_data, self.val_data, self.test_data = self.prepare_data()
        self.model, self.criterion = self.build_model()
        self.optimizer = self.create_optimizer()
        self.best_val_loss = []
        self.stored_loss = float("inf")

    def setup_reproducibility(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, consider using --cuda")
            else:
                torch.cuda.manual_seed(self.args.seed)

    def load_corpus(self):
        cache_path = f"corpus.{hashlib.md5(self.args.data.encode()).hexdigest()}.data"
        # if os.path.exists(cache_path):
        #     print("Loading cached dataset...")
        #     return torch.load(cache_path)
        # else:
        print("Producing dataset...")
        corpus = data.Corpus(self.args.data)
        # torch.save(corpus, cache_path)
        return corpus

    def prepare_data(self):
        eval_batch_size = 10
        test_batch_size = 1
        train_data = batchify(self.corpus.train, self.args.batch_size, self.args)
        val_data = batchify(self.corpus.valid, eval_batch_size, self.args)
        test_data = batchify(self.corpus.test, test_batch_size, self.args)
        return train_data, val_data, test_data

    def build_model(self):
        ntokens = len(self.corpus.dictionary)

        splits = self.determine_splits(ntokens)
        print(f"Using splits: {splits}")

        print("self.args.model:", self.args.model)
        model_instance = my_model.RNNModel(
            self.args.model,
            ntokens,
            self.args.emsize,
            self.args.nhid,
            self.args.nlayers,
            self.args.dropout,
            self.args.dropouth,
            self.args.dropouti,
            self.args.dropoute,
            self.args.wdrop,
            self.args.tied,
        )

        criterion = SplitCrossEntropyLoss(
            self.args.emsize, splits=splits, verbose=False
        )

        if self.args.resume:
            model_instance, criterion, _ = self.load_checkpoint(self.args.resume)
            self.update_model_params(model_instance)

        if self.args.cuda:
            model_instance = model_instance.cuda()
            criterion = criterion.cuda()

        params = list(model_instance.parameters()) + list(criterion.parameters())
        total_params = sum(x.numel() for x in params if x.numel() > 0)
        print(f"Args: {self.args}")
        print(f"Model total parameters: {total_params}")

        return model_instance, criterion

    def determine_splits(self, ntokens):
        if ntokens > 500000:
            return [4200, 35000, 180000]
        elif ntokens > 75000:
            return [2800, 20000, 76000]
        return []

    def update_model_params(self, model_instance):
        model_instance.dropouti = self.args.dropouti
        model_instance.dropouth = self.args.dropouth
        model_instance.dropout = self.args.dropout

        if self.args.wdrop:
            for rnn in model_instance.rnns:
                if isinstance(rnn, WeightDrop):
                    rnn.dropout = self.args.wdrop
                elif hasattr(rnn, "zoneout") and rnn.zoneout > 0:
                    rnn.zoneout = self.args.wdrop

    def create_optimizer(self):
        params = list(self.model.parameters()) + list(self.criterion.parameters())

        if self.args.optimizer == "sgd":
            return torch.optim.SGD(
                params, lr=self.args.lr, weight_decay=self.args.wdecay
            )
        elif self.args.optimizer == "adam":
            return torch.optim.Adam(
                params, lr=self.args.lr, weight_decay=self.args.wdecay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

    def save_checkpoint(self, path):
        with open(path, "wb") as f:
            torch.save([self.model, self.criterion, self.optimizer], f)

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            return torch.load(f)

    def evaluate(self, data_source, batch_size=10):
        self.model.eval()
        if self.args.model == "QRNN":
            self.model.reset()

        total_loss = 0
        hidden = self.model.init_hidden(batch_size)

        num_batches = (data_source.size(0) - 1) // self.args.bptt

        with torch.no_grad():
            # for i in tqdm(
            #     range(0, data_source.size(0) - 1, self.args.bptt),
            #     desc="Evaluating",
            #     leave=False,
            #     total=num_batches,
            # ):
            for i in range(0, data_source.size(0) - 1, self.args.bptt):
                data, targets = get_batch(data_source, i, self.args, evaluation=True)
                output, hidden = self.model(data, hidden)
                total_loss += (
                    len(data)
                    * self.criterion(
                        self.model.decoder.weight,
                        self.model.decoder.bias,
                        output,
                        targets,
                    ).data
                )
                hidden = repackage_hidden(hidden)

        return total_loss.item() / len(data_source)

    def train_epoch(self, epoch):
        if self.args.model == "QRNN":
            self.model.reset()

        total_loss = 0
        start_time = time.time()
        hidden = self.model.init_hidden(self.args.batch_size)

        num_batches = self.train_data.size(0) // self.args.bptt
        # pbar = tqdm(total=num_batches, desc=f"Epoch {epoch}", unit="batch")

        batch, i = 0, 0
        while i < self.train_data.size(0) - 1 - 1:
            seq_len = self.get_sequence_length()

            lr_scale = seq_len / self.args.bptt
            original_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.param_groups[0]["lr"] = original_lr * lr_scale

            self.model.train()
            data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)
            # (??, batch size) for the input
            # data shape: (67,20)
            # 67 * 20 = 1340
            # targets shape: (1340)

            hidden = repackage_hidden(hidden)
            self.optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = self.model(
                data, hidden, return_h=True
            )
            raw_loss = self.criterion(
                self.model.decoder.weight, self.model.decoder.bias, output, targets
            )

            loss = self.compute_regularized_loss(raw_loss, rnn_hs, dropped_rnn_hs)
            loss.backward()

            if self.args.clip:
                params = list(self.model.parameters()) + list(
                    self.criterion.parameters()
                )
                torch.nn.utils.clip_grad_norm_(params, self.args.clip)

            self.optimizer.step()
            total_loss += raw_loss.data
            self.optimizer.param_groups[0]["lr"] = original_lr

            if batch % self.args.log_interval == 0 and batch > 0:
                # self.log_training_progress(total_loss, pbar)
                total_loss = 0
                start_time = time.time()

            batch += 1
            i += seq_len
            # pbar.update(1)

        # pbar.close()

    def get_sequence_length(self):
        bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.0
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        return seq_len

    def compute_regularized_loss(self, raw_loss, rnn_hs, dropped_rnn_hs):
        loss = raw_loss

        if self.args.alpha:
            loss = loss + sum(
                self.args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )

        if self.args.beta:
            loss = loss + sum(
                self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )

        return loss

    def log_training_progress(self, total_loss, pbar):
        cur_loss = total_loss.item() / self.args.log_interval

        pbar.set_postfix(
            {
                "loss": f"{cur_loss:.2f}",
                "ppl": f"{math.exp(cur_loss):.2f}",
                "bpc": f"{cur_loss / math.log(2):.3f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.5f}",
            }
        )

    def should_switch_to_asgd(self, val_loss):
        return (
            self.args.optimizer == "sgd"
            and "t0" not in self.optimizer.param_groups[0]
            and len(self.best_val_loss) > self.args.nonmono
            and val_loss > min(self.best_val_loss[: -self.args.nonmono])
        )

    def switch_to_asgd(self):
        print("Switching to ASGD")
        self.optimizer = torch.optim.ASGD(
            self.model.parameters(),
            lr=self.args.lr,
            t0=0,
            lambd=0.0,
            weight_decay=self.args.wdecay,
        )

    def evaluate_with_averaging(self, epoch_start_time, epoch):
        uses_asgd = any(("ax" in state) for state in self.optimizer.state.values())

        if not uses_asgd:
            val_loss = self.evaluate(self.val_data)
            self.print_validation_results(epoch, epoch_start_time, val_loss)

            if val_loss < self.stored_loss:
                self.save_checkpoint(self.args.save)
                print("Saving (no averaging).")
                self.stored_loss = val_loss

            return val_loss

        tmp = {}
        for prm in self.model.parameters():
            tmp[prm] = prm.data.clone()
            prm.data = self.optimizer.state[prm]["ax"].clone()

        val_loss = self.evaluate(self.val_data)
        self.print_validation_results(epoch, epoch_start_time, val_loss)

        if val_loss < self.stored_loss:
            self.save_checkpoint(self.args.save)
            print("Saving Averaged!")
            self.stored_loss = val_loss

        for prm in self.model.parameters():
            prm.data = tmp[prm].clone()

        return val_loss

    def evaluate_standard(self, epoch_start_time, epoch):
        val_loss = self.evaluate(self.val_data, 10)
        self.print_validation_results(epoch, epoch_start_time, val_loss)

        if val_loss < self.stored_loss:
            self.save_checkpoint(self.args.save)
            print("Saving model (new best validation)")
            self.stored_loss = val_loss

        if self.should_switch_to_asgd(val_loss):
            self.switch_to_asgd()

        if epoch in self.args.when:
            print("Saving model before learning rate decreased")
            self.save_checkpoint(f"{self.args.save}.e{epoch}")
            print("Dividing learning rate by 10")
            self.optimizer.param_groups[0]["lr"] /= 10.0

        return val_loss

    def print_validation_results(self, epoch, epoch_start_time, val_loss):
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | "
            f"valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f} | "
            f"valid bpc {val_loss / math.log(2):8.3f}"
        )
        print("-" * 89)

    def train(self):
        try:
            # for epoch in tqdm(
            #     range(1, self.args.epochs + 1), desc="Training Progress", unit="epoch"
            # ):
            for epoch in range(1, self.args.epochs + 1):
                epoch_start_time = time.time()
                self.train_epoch(epoch)

                if isinstance(self.optimizer, torch.optim.ASGD):
                    val_loss = self.evaluate_with_averaging(epoch_start_time, epoch)
                else:
                    val_loss = self.evaluate_standard(epoch_start_time, epoch)

                self.best_val_loss.append(val_loss)

        except KeyboardInterrupt:
            print("-" * 89)
            print("Exiting from training early")

    def test(self):
        self.model, self.criterion, self.optimizer = self.load_checkpoint(
            self.args.save
        )
        test_loss = self.evaluate(self.test_data, 1)

        print("=" * 89)
        print(
            f"| End of training | test loss {test_loss:5.2f} | "
            f"test ppl {math.exp(test_loss):8.2f} | test bpc {test_loss / math.log(2):8.3f}"
        )
        print("=" * 89)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch PennTreeBank RNN/LSTM Language Model"
    )
    parser.add_argument(
        "--data", type=str, default="data/penn/", help="location of the data corpus"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LSTM",
        help="type of recurrent net (LSTM, QRNN, GRU)",
    )
    parser.add_argument(
        "--emsize", type=int, default=400, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=1150, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=3, help="number of layers")
    parser.add_argument("--lr", type=float, default=30, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=8000, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=80, metavar="N", help="batch size"
    )
    parser.add_argument("--bptt", type=int, default=70, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dropouth",
        type=float,
        default=0.3,
        help="dropout for rnn layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dropouti",
        type=float,
        default=0.65,
        help="dropout for input embedding layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dropoute",
        type=float,
        default=0.1,
        help="dropout to remove words from embedding layer (0 = no dropout)",
    )
    parser.add_argument(
        "--wdrop",
        type=float,
        default=0.5,
        help="amount of weight dropout to apply to the RNN hidden to hidden matrix",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--nonmono", type=int, default=5, help="random seed")
    parser.add_argument("--cuda", action="store_false", help="use CUDA")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )

    randomhash = "".join(str(time.time()).split("."))
    parser.add_argument(
        "--save",
        type=str,
        default=randomhash + ".pt",
        help="path to save the final model",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2,
        help="alpha L2 regularization on RNN activation (alpha = 0 means no regularization)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)",
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=1.2e-6,
        help="weight decay applied to all weights",
    )
    parser.add_argument(
        "--resume", type=str, default="", help="path of model to resume"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="optimizer to use (sgd, adam)"
    )
    parser.add_argument(
        "--when",
        nargs="+",
        type=int,
        default=[-1],
        help="When (which epochs) to divide the learning rate by 10 - accepts multiple",
    )

    args = parser.parse_args()
    args.tied = True
    return args


def main():
    args = parse_arguments()
    trainer = LanguageModelTrainer(args)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
