# %%
import torch
import scipy.io as sio
from os.path import dirname, join as pjoin
import scipy
import math
import matplotlib.pyplot as plt
import numpy as np

# %% Setup some variables
rock = 0
paper = 1
scissors = 2

# %%


class TrialData:
    def __init__(self, data, label: int):
        # Min/max scale the data
        self.data = data / np.max(data)
        self.label = label
        assert data.shape == (3, 1500)


# %%
class Experiment:
    trials: list[TrialData] = []
    name: str

    def __init__(self, name, path: str, labels: "list[int]"):
        self.name = name
        raw_df = sio.loadmat(path)['EMG'][0][0]
        matlab_data = raw_df[4]
        assert matlab_data.shape == (3, 1500, 30), "data shape is wrong"

        for trial_id in range(30):
            assert matlab_data[:, :, trial_id].shape == (3, 1500)
            self.trials.append(
                TrialData(matlab_data[:, :, trial_id], labels[trial_id]))

# %%


class EmgDataset:
    labels: list[int]
    # data


# %%


def load_emg_data_trait() -> EmgDataset:
    experiments: list[Experiment] = []

    experiments.append(Experiment(
        "EMGdata-RPS-04231538",
        "./../data/aFIN/EMGdata-RPS-04231538.mat", [
            scissors, rock, rock, rock, scissors, paper, paper, scissors, paper, scissors,
            rock, rock, rock, rock, paper, rock, rock, paper, scissors, scissors, paper,
            scissors, paper, paper, scissors, paper, scissors, scissors, paper, rock,
        ]))

    experiments.append(Experiment(
        "EMGdata-RPS-04231542",
        "./../data/aFIN/EMGdata-RPS-04231542.mat", [
            rock, rock, rock, scissors, scissors, scissors, paper, rock, scissors, scissors,
            scissors, scissors, rock, paper, rock, scissors, scissors, paper, paper, paper,
            paper, paper, rock, rock, rock, rock, scissors, paper, paper, paper,
        ]))

    experiments.append(Experiment(
        "EMGdata-RPS-04231545",
        "./../data/aFIN/EMGdata-RPS-04231545.mat", [
            scissors, scissors, scissors, scissors, scissors, rock, scissors, rock, scissors,
            rock, paper, rock, rock, paper, rock, paper, paper, paper, paper, rock, paper,
            paper, scissors, rock, scissors, scissors, rock, paper, rock, paper,
        ]))

    experiments.append(Experiment(
        "EMGdata-RPS-04231549",
        "./../data/aFIN/EMGdata-RPS-04231549.mat", [
            rock, scissors, paper, paper, scissors, rock, scissors, paper, paper, rock, paper,
            paper, rock, scissors, paper, paper, scissors, scissors, scissors, rock, rock,
            scissors, rock, rock, rock, scissors, rock, scissors, paper, paper,
        ]))

    experiments.append(Experiment(
        "EMGdata-RPS-04231553",
        "./../data/aFIN/EMGdata-RPS-04231553.mat", [
            paper, rock, rock, paper, paper, paper, paper, rock, scissors, rock, scissors,
            rock, scissors, rock, paper, scissors, scissors, rock, paper, rock, paper,
            scissors, rock, scissors, scissors, scissors, paper, paper, scissors, rock,
        ]))

    dataset = EmgDataset()

    outarray = np.array([], [], [])

    for exp in experiments:
        for trial in exp.trials:
            dataset.labels.append(trial.label)
            outarray.concatenate()

    return dataset


def load_emg_data_test():

    # experiments["EMGdata-RPS-04231556"] = Experiment(
    #     "EMGdata-RPS-04231556",
    #     "./../data/aFIN/EMGdata-RPS-04231556.mat", [
    #         rock, paper, paper, paper, rock, rock, scissors, scissors, paper, rock, paper,
    #         rock, paper, rock, paper, scissors, scissors, scissors, rock, scissors, scissors,
    #         paper, scissors, scissors, rock, rock, scissors, paper, paper, rock,
    #     ])

    # experiments["EMGdata-RPS-04231559"] = Experiment(
    #     "EMGdata-RPS-04231559",
    #     "./../data/aFIN/EMGdata-RPS-04231559.mat", [
    #         rock, paper, scissors, paper, rock, rock, scissors, rock, rock, paper, scissors,
    #         paper, paper, rock, scissors, rock, scissors, paper, scissors, scissors, rock,
    #         paper, rock, paper, rock, paper, paper, scissors, scissors, scissors,
    #     ])

    pass


# %% Tests
if __name__ == "__main__":

    # %%
    print("he;;")

# %%
    exp = Experiment(
        "./../data/aFIN/EMGdata-RPS-04231538.mat", "asd", [
            scissors, rock, rock, rock, scissors, paper, paper, scissors, paper, scissors,
            rock, rock, rock, rock, paper, rock, rock, paper, scissors, scissors, paper,
            scissors, paper, paper, scissors, paper, scissors, scissors, paper, rock,
        ])


# %%
    dat = sio.loadmat("./../data/aFIN/EMGdata-RPS-04231538.mat")
    raw_df = dat['EMG'][0][0]
    matlab_data = raw_df[4]
    print(matlab_data[:, :, 29].shape)
# %%
    for x in range(0, 3):
        fig = plt.figure(x, dpi=200)
        plt.plot(matlab_data[0, :, x])
        plt.plot(matlab_data[1, :, x])
        plt.plot(matlab_data[2, :, x])
# %%

    exp = Experiment(
        "EMGdata-RPS-04231542",
        "../data/aFIN/EMGdata-RPS-04231542.mat", [
            rock, rock, rock, scissors, scissors, scissors, paper, rock, scissors, scissors,
            scissors, scissors, rock, paper, rock, scissors, scissors, paper, paper, paper,
            paper, paper, rock, rock, rock, rock, scissors, paper, paper, paper,
        ])

    pass
