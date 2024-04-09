from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os, time
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from train_new_agent import select_random_checkpoints

def run_benchmark(task, load_path, save_path, seed, share_params):
    if save_path != None and not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.share_policy_params = share_params
    experiment_config.save_folder = save_path

    algorithm_config = MaddpgConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        algorithm_config = algorithm_config,
        task = task,
        seed = seed,
        config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )

    x = torch.load(load_path)
    experiment = experiment.load_experiment_policy(x)
    experiment._evaluation_loop()
    experiment.close()

if __name__ == "__main__":

    # Checkpoint Paths
    universal_path_one =  "outputs/2024-02-01/19-08-21/maddpg_simple_reference_mlp__ef76cae9_24_02_01-19_08_21/checkpoints"
    universal_path_two = "outputs/Final Models/2024-02-01/01-33-36/maddpg_simple_reference_mlp__0370dbb7_24_02_01-01_33_36/checkpoints"
    noise_path_one = "outputs/Final Models/19-38-40/maddpg_simple_reference_idiolect_mlp__18300887_24_01_14-19_38_40/checkpoints"
    noise_path_two = "outputs/2024-02-01/19-07-50/maddpg_simple_reference_idiolect_mlp__2aa7883e_24_02_01-19_07_50/checkpoints"

    universal_path_shared = "evaluation/checkpoints/final/universal.pt"
    noise_path_shared = "evaluation/checkpoints/final/noise.pt"

    # Tasks
    original_env_uni = VmasTask.SIMPLE_REFERENCE.get_from_yaml()
    original_env_idio = VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml()
    # novel_env_uni = IdiolectEvoTask.SPEED_NEW.get_from_yaml()
    # novel_env_idio = IdiolectEvoTask.SPEED_NEW_NOISE.get_from_yaml()
    
    # shared_pairs = [
    #     (universal_path_shared, original_env_uni),
    #     (noise_path_shared, original_env_idio),
    #     (universal_path_shared, novel_env_uni),
    #     (noise_path_shared, novel_env_idio)
    # ]
    
    # unshared_pairs = [
    #     (universal_path_unshared, original_env_uni),
    #     (noise_path_unshared, original_env_idio),
    #     (universal_path_unshared, novel_env_uni),
    #     (noise_path_unshared, novel_env_idio)
    # ]

    # Save Path
    save_path = "/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/graphs/2-14-update/vids/correct-pairs"

    # Seed
    seeds = 1

    # # Run Shared Evaluations
    # for idx, pair in enumerate(shared_pairs):
    #     save_path = save_folder+"/original_env/" if idx<2 else save_folder+"/novel_env/"
        
    #     load_path = pair[0]
    #     task = pair[1]
    #     run_benchmark(task = task, load_path=load_path, save_path=save_path, seed=seed, share_params=True)

    # time.sleep(10)    
    # # Run Unshared Evaluations
    # for idx, pair in enumerate(unshared_pairs):
    #     save_path = save_folder+"/original_env/" if idx<2 else save_folder+"/novel_env/"
        
    #     load_path = pair[0]
    #     task = pair[1]
    #     run_benchmark(task = task, load_path=load_path, save_path=save_path, seed=seed, share_params=False)
    
    # Combinations per Seed
    combs = 5

    # Get Checkpoints
    checkpoint_ones = []
    checkpoint_twos = []

    for comb in range(combs):
        checkpoint_one, checkpoint_two = select_random_checkpoints(min_one=3_000_000, min_two=3_000_000)
        checkpoint_ones.append(checkpoint_one)
        checkpoint_twos.append(checkpoint_two)
    
    # One Shot Agent Integration
    for seed in range(seeds):
        for comb in range(combs):
            print("Combination:", comb)

            checkpoint_one = checkpoint_ones[comb]
            checkpoint_two = checkpoint_twos[comb]

            PATH_ONE_UNIVERSAL = universal_path_one+"/checkpoint_"+str(checkpoint_one)+".pt"
            PATH_TWO_UNIVERSAL = universal_path_two+"/checkpoint_"+str(checkpoint_two)+".pt"

            PATH_ONE_IDIOLECT = noise_path_one+"/checkpoint_"+str(checkpoint_one)+".pt"
            PATH_TWO_IDIOLECT = noise_path_two+"/checkpoint_"+str(checkpoint_two)+".pt"
            
            run_benchmark(task = original_env_uni, load_path=PATH_ONE_UNIVERSAL, save_path=save_path, seed=seed, share_params=True)
            run_benchmark(task = original_env_uni, load_path=PATH_TWO_UNIVERSAL, save_path=save_path, seed=seed, share_params=True)
            run_benchmark(task = original_env_idio, load_path=PATH_ONE_IDIOLECT, save_path=save_path, seed=seed, share_params=False)
            run_benchmark(task = original_env_idio, load_path=PATH_TWO_IDIOLECT, save_path=save_path, seed=seed, share_params=False)






