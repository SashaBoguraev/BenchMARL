from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os, random, sys, argparse
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def run_experiment(task, shared, seed, max):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml('fine_tuned/vmas/conf/vmas_parameters.yaml')

    # Some basic other configs
    algorithm_config = MaddpgConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Override Necessary Configs
    algorithm_config.share_param_critic = False
    experiment_config.share_policy_params = shared
    experiment_config.max_n_frames = max_n_frames

    # Load Experiment
    experiment = Experiment(
        algorithm_config = algorithm_config,
        task = task,
        seed = seed,
        config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )

    # Train
    experiment.run(eval=True)

if __name__ == "__main__":
    # Set Necessary Variables
    task_noiseless = VmasTask.SIMPLE_REFERENCE.get_from_yaml()
    task_noise = VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml()
    max_n_frames = 5_400_000

    # We want to train (3/9) 
    # 1. 2x Shared Noise
    # 2. 2x Shared Noiseless 
    # 3. 1x Unshared Noise

    # Below order is prioritized such that I can run experiments 

    # Shared Noise
    run_experiment(task_noise, shared=True, seed = 1, max = max_n_frames)
    # Unshared Noise
    run_experiment(task_noiseless, shared=True, seed = 1, max = max_n_frames)
    # Shared Noise    
    run_experiment(task_noise, shared=True, seed = 2, max = max_n_frames)
    # Unshared Noise
    run_experiment(task_noiseless, shared=True, seed = 2, max = max_n_frames)
    # Unshared Noise
    run_experiment(task_noise, shared=False, seed = 1, max = max_n_frames)





    