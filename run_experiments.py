import json
import os

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from torch import nn

from daniel_lib import (
    BERThoven,
    augment_dataset,
    check_accuracy,
    get_sentence_embeddings,
    get_test_labels,
    get_tokenized,
    getDataLoader,
    import_file,
    pad,
    progress,
    removeOutliers,
    tokenizer,
    train_part,
)
from transformers import (
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def experiments_match(exp1, exp2):
    if set(exp1.keys()) == set(exp2.keys()):
        return all([exp1[key1] == exp2[key1] for key1 in exp1.keys()])
    return False


def generate_results_filename(filename):
    extension = filename.split(".")[-1]
    return ".".join(filename.split(".")[:-1]) + f"_results.{extension}"


def get_loss_from_string(loss_string):
    return {
        "l1_smooth": F.smooth_l1_loss,
        "l1": F.l1_loss,
        "l2": F.mse_loss,
        "nll": F.nll_loss,
    }[loss_string]


def get_optimizer_from_string(optimizer_string):
    return {"AdamW": AdamW}[optimizer_string]


def build_model(params):
    concat_outputs = params["combine_inputs"] == "concat"
    sum_outputs = params["combine_inputs"] == "sum"
    dropout = params["dropout"] == 0.0
    return BERThoven(
        cls=params["cls"],
        concat_outputs=concat_outputs,
        sum_outputs=sum_outputs,
        dropout=dropout,
        dropout_prob=params["dropout"],
    )


class ExperimentRunner:
    def __init__(self, experiments_file, dataset_path, use_gpu=True):
        self.experiments_file = experiments_file
        self.results_file = generate_results_filename(experiments_file)
        self.dataset_path = dataset_path
        self.finished_experiments = []
        self.remaining_experiments = []

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.load_dataset()

    def load_dataset(self):
        train_df = import_file("train")
        dev_df = import_file("dev")
        test_df = import_file("test")
        train_df_aug = augment_dataset(
            train_df,
            lambda score: score < -1,
            lambda score: score < -0.3,
            lambda score: score > 0.55,
            lambda score: score > 1,
            lambda score: score > 1.3,
        )

        self.dataLoader_train = getDataLoader(train_df, batch_size=32)
        self.dataLoader_train_aug = getDataLoader(train_df_aug, batch_size=32)
        self.dataLoader_dev = getDataLoader(dev_df, batch_size=32)
        self.dataLoader_test = getDataLoader(test_df, batch_size=32, test=True)

    def reload_experiments(self):
        with open(self.experiments_file) as f:
            experiments = json.load(f)

        with open(self.results_file) as f:
            self.finished_experiments = json.load(f)

        self.remaining_experiments = [
            experiment
            for experiment in experiments.items()
            if self.check_if_done(experiment)
        ]

    def check_if_done(self, experiment):
        for finished_experiment in self.finished_experiments:
            if experiments_match(experiment, finished_experiment["params"]):
                return True
        return False

    def save_experiment(self, params, mae, mse, pearson):
        self.finished_experiments.append(
            {"params": params, "results": {"mae": mae, "mse": mse, "pearson": pearson}}
        )
        with open(self.results_file, "w") as f:
            json.dump(self.finished_experiments, f)

    def train(self, model, params, print_every=60):
        loss_function = get_loss_from_string(params["loss_function"])

        steps_per_epoch = len(self.dataLoader_train_aug)
        training_steps = steps_per_epoch * params["epochs"]
        warmup_steps = int(training_steps * params["warmup_proportion"])

        optimizer = params["optimizer"](
            model.parameters(), lr=params["lr"], eps=params["eps"], correct_bias=False
        )
        scheduler = params["scheduler"](
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        aug_epochs = int(params["upsampling"] * params["epochs"])

        train_part(
            model,
            self.dataLoader_train_aug,
            optimizer,
            scheduler,
            val_loader=self.dataLoader_dev,
            epochs=aug_epochs,
            print_every=print_every,
            loss_function=loss_function,
            device=self.device,
        )

        train_part(
            model,
            self.dataLoader_train,
            optimizer,
            scheduler,
            val_loader=self.dataLoader_dev,
            epochs=params["epochs"] - aug_epochs,
            print_every=print_every,
            loss_function=loss_function,
            device=self.device,
        )

    def run(self):
        self.reload_experiments()
        for experiment in self.remaining_experiments:
            model = build_model(experiment)
            self.train(model, experiment)
