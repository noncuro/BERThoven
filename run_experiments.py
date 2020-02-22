import json
import os
import zipfile

import requests
import torch
import torch.nn.functional as F

from daniel_lib import (
    BERThoven,
    augment_dataset,
    getDataLoader,
    import_file,
    train_part,
)
from transformers import (
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def exp_to_string(experiment):
    return " ".join([f"{key} = {value}" for key, value in experiment.items()])


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


def get_scheduler_from_string(scheduler_string):
    return {
        "linear": get_linear_schedule_with_warmup,
        "constant": get_constant_schedule_with_warmup,
    }[scheduler_string]


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

    def maybe_download(
        self,
        filename="files.zip",
        download_url="https://competitions.codalab.org/my/datasets/download/c748d2c0-d6be-4e36-9f12-ca0e88819c4d",
    ):
        if not os.path.isfile(filename):
            print(f"Dataset not found, downloading from {download_url}")
            r = requests.get(download_url)
            with open(filename, "wb") as f:
                f.write(r.content)
            if not os.path.isdir(self.dataset_path):
                os.makedirs(self.dataset_path)
            zipfile.ZipFile(filename).extractall(self.dataset_path)
        else:
            print("Dataset already downloaded")

    def load_dataset(self):
        train_df = import_file("train", path=self.dataset_path)
        dev_df = import_file("dev", path=self.dataset_path)
        test_df = import_file("test", path=self.dataset_path)
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
        if not os.path.isfile(self.experiments_file):
            print("No experiments found. Nothing to do here.")
            exit(0)

        with open(self.experiments_file) as f:
            experiments = json.load(f)

        if os.path.isfile(self.results_file):
            with open(self.results_file) as f:
                self.finished_experiments = json.load(f)
        else:
            self.finished_experiments = []

        self.remaining_experiments = [
            experiment
            for experiment in experiments
            if not self.check_if_done(experiment)
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
            json.dump(self.finished_experiments, f, indent=4)

    def train(self, model, params, print_every=60):
        loss_function = get_loss_from_string(params["loss_function"])

        steps_per_epoch = len(self.dataLoader_train_aug)
        training_steps = steps_per_epoch * params["epochs"]
        warmup_steps = int(training_steps * params["warmup_proportion"])

        optimizer = get_optimizer_from_string(params["optimizer"])(
            model.parameters(), lr=params["lr"], eps=params["eps"], correct_bias=False
        )

        scheduler = get_scheduler_from_string(params["scheduler"])(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        aug_epochs = params["epochs"] - 1

        if aug_epochs > 0:
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

        final_mae, final_mse, final_pr = train_part(
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

        return final_mae, final_mse, final_pr

    def run(self):
        print("Looking for dataset...")
        self.maybe_download()
        print("Loading datasets into memory...")
        self.load_dataset()
        print("Loading experiments file...")
        self.reload_experiments()
        print(f"Found {len(self.remaining_experiments)} experiments to run")
        print("Started experiments...")
        for i, experiment in enumerate(self.remaining_experiments):
            print("=" * 30)
            print(f"Experiment {i+1} of {len(self.remaining_experiments)}")
            print(exp_to_string(experiment))
            model = build_model(experiment)
            mae, mse, pr = self.train(model, experiment)
            self.save_experiment(experiment, mae, mse, pr)


if __name__ == "__main__":
    experiments_file = "experiments.json"
    dataset_path = "dataset_files"
    runner = ExperimentRunner(experiments_file, dataset_path)
    runner.run()
