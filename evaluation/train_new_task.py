from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def run_adaptation(task, PATH, seed=0, share_params=False):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml('fine_tuned/vmas/conf/vmas_parameters.yaml')

    # Some basic other configs
    algorithm_config = MaddpgConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Override Necessary Configs
    algorithm_config.share_param_critic = False
    experiment_config.share_policy_params = False
    experiment_config.share_policy_params = share_params
    experiment_config.max_n_frames = 4_200_000

    experiment = Experiment(
        algorithm_config = algorithm_config,
        task = task,
        seed = seed,
        config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )

    x = torch.load(PATH)

    experiment = experiment.load_experiment_policy(x)
    experiment.run()

if __name__ == "__main__":
    # Use Similar Reward Value Points
    universal_path_unshared = "outputs/Final Models/10-31-32/maddpg_simple_reference_mlp__d5090c0f_24_01_15-10_31_32/checkpoints/checkpoint_4200000.pt"
    universal_path_shared = "outputs/Final Models/2024-02-01/01-33-36/maddpg_simple_reference_mlp__0370dbb7_24_02_01-01_33_36/checkpoints/checkpoint_9900000.pt"
    noise_path = "outputs/Final Models/19-38-40/maddpg_simple_reference_idiolect_mlp__18300887_24_01_14-19_38_40/checkpoints/checkpoint_4200000.pt"
    
    # Set Seed
    # seeds = 3

    for idx in range(10):
        task_uni = getattr(IdiolectEvoTask, f"ADAPT_COLOR_{idx}")
        task_idio = getattr(IdiolectEvoTask, f"ADAPT_COLOR_NOISE_{idx}")
        run_adaptation(task_uni.get_from_yaml(), universal_path_shared, share_params=True)
        run_adaptation(task_idio.get_from_yaml(), noise_path, share_params=False)