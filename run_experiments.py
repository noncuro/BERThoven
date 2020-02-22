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
    return {"l1_smooth": F.smooth_l1_loss, "l1": F.l1_loss, "l2": F.mse_loss}[
        loss_string
    ]


def get_optimizer_from_string(optimizer_string):
    return {"AdamW": AdamW}[optimizer_string]


class ExperimentRunner:
    def __init__(self, experiments_file, dataset_path):
        self.experiments_file = experiments_file
        self.results_file = generate_results_filename(experiments_file)
        self.dataset_path = dataset_path
        self.finished_experiments = []
        self.remaining_experiments = []

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

        steps_per_epoch = len(dataLoader_train_aug)
        training_steps = steps_per_epoch * params["epochs"]
        warmup_steps = int(training_steps * params["warmup_proportion"])

        optimizer = params["optimizer"](
            model.parameters(), lr=params["lr"], eps=params["eps"], correct_bias=False
        )
        scheduler = params["scheduler"](
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )
        # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

        aug_epochs = int(params["upsampling"] * params["epochs"])

        train_part(
            nlp_model,
            dataLoader_train_aug,
            optimizer,
            scheduler,
            val_loader=dataLoader_dev,
            epochs=aug_epochs,
            print_every=print_every,
            loss_function=loss_function,
            device=device,
        )

        train_part(
            nlp_model,
            dataLoader_train,
            optimizer,
            scheduler,
            val_loader=dataLoader_dev,
            epochs=params["epochs"] - aug_epochs,
            print_every=print_every,
            loss_function=loss_function,
            device=device,
        )

    def run(self):

        self.reload_experiments()
