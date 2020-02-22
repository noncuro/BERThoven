import json


def experiments_match(exp1, exp2):
    if set(exp1.keys()) == set(exp2.keys()):
        return all([exp1[key1] == exp2[key1] for key1 in exp1.keys()])
    return False


class ExperimentRunner():
    def __init__(self, experiments_file):
        self.experiments_file = experiments_file
        self.finished_experiments = []

    def reload_experiments(self):
        with open(self.experiments_file) as f:
            self.experiments = json.load(f)

    def check_if_done(self, experiment):
        for finished_experiment in self.finished_experiments:
            if experiments_match(experiment, finished_experiment):
                return True
        return False

    def run(self):
        pass