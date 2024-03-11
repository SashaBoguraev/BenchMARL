from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os, random
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def run_train(task, num_envs, lr, batch_size, share_params, attempt):
    save_path = "evaluation/tuning_params/logging/search"+str(attempt)
    if save_path != None and not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.share_policy_params = share_params
    experiment_config.off_policy_n_envs_per_worker = num_envs 
    experiment_config.off_policy_collected_frames_per_batch = num_envs*100 
    experiment_config.evaluation_interval = num_envs*100*3
    experiment_config.checkpoint_interval = num_envs*100*30
    experiment_config.off_policy_train_batch_size = batch_size
    experiment_config.max_n_frames = max(8_000_000, num_envs*10000) 
    experiment_config.save_folder = save_path

    algorithm_config = MaddpgConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        algorithm_config = algorithm_config,
        task = task,
        seed = 0,
        config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )
    
    experiment.run()
    experiment.close()

if __name__ == "__main__":
    # Tasks
    original_env_uni = VmasTask.SIMPLE_REFERENCE.get_from_yaml()
    original_env_idio = VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml()

    # Define Ranges
    num_envs_range = [1, 5]
    lr_range = [-5, -2]
    batch_size_range = [6, 13]

    # Determine Number of Iterations
    iters = 10

    # Attempt #
    attempt = 1

    # Save File for Combinations
    f = open("evaluation/tuning_params/search"+str(attempt)+".txt", "w")

    for iter in range(iters):
        num_envs = 6*10**(random.randrange(num_envs_range[0], num_envs_range[1]))
        lr = 10**(random.randrange(lr_range[0], lr_range[1]))
        batch_size = 2**(random.randrange(batch_size_range[0], batch_size_range[1]))

        print("Num Envs: "+str(num_envs)+", Learning Rate: "+str(lr)+", Batch Size: "+str(batch_size))
        f.write("Num Envs: "+str(num_envs)+", Learning Rate: "+str(lr)+", Batch Size: "+str(batch_size))

        run_train(original_env_uni, num_envs, lr, batch_size, share_params = False, attempt = attempt)
        # run_train(original_env_idio, num_envs, lr, batch_size, share_params = False, attempt = attempt)

    f.close()