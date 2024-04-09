from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from eval_seed import run_benchmark, get_error

def get_means(paths, task, shared, seeds_per_eval, checkpoints):
    means = np.zeros((len(paths), len(checkpoints)))
    errors = np.zeros(len(checkpoints))
    for idx_checkpoint, checkpoint in enumerate(checkpoints):
        for idx_path, path in enumerate(paths):
            checkpoint_path = path+str(checkpoint)+".pt"
            seed_means = []
            for seed in range(seeds_per_eval):
                stats, mean_stats, to_graphs = run_benchmark(task=task, PATH=checkpoint_path, seed=seed, share_params=shared)
                seed_means.append(mean_stats["Mean Rewards"])
            means[idx_path, idx_checkpoint] = np.mean(seed_means)
        errors[idx_checkpoint] = get_error(means[:, idx_checkpoint])
    return means, errors

def write_csv(save_path, eval_type, means):
    mean_address = save_path+"/stats/"
    
    if not os.path.isdir(mean_address):
        os.makedirs(mean_address)

    # Open the file with the append state
    with open(mean_address+eval_type+'.csv', mode='w+') as file:
        writer = csv.writer(file)
        # Write the state information to the CSV file
        for mean in means:
            writer.writerow(mean)

def plot_means(shared_noiseless_paths, unshared_noiseless_paths, shared_noise_paths, unshared_noise_paths, seeds, checkpoints, new, save_path, random=False):
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # Tasks
    if not random:
        task_noiseless = VmasTask.SIMPLE_REFERENCE.get_from_yaml() if not new else IdiolectEvoTask.SPEED_NEW.get_from_yaml()
        task_noise = VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml() if not new else IdiolectEvoTask.SPEED_NEW_NOISE.get_from_yaml()

    # Get Means and Errors
    shared_noiseless_means, shared_noiseless_errors = get_means(paths=shared_noiseless_paths, task=task_noiseless, shared=True, seeds_per_eval=seeds, checkpoints=checkpoints)
    unshared_noiseless_means, unshared_noiseless_errors = get_means(paths=unshared_noiseless_paths, task=task_noiseless, shared=False, seeds_per_eval=seeds, checkpoints=checkpoints)
    shared_noise_means, shared_noise_errors = get_means(paths=shared_noise_paths, task=task_noise, shared=True, seeds_per_eval=seeds, checkpoints=checkpoints)
    unshared_noise_means, unshared_noise_errors = get_means(paths=unshared_noise_paths, task=task_noise, shared=False, seeds_per_eval=seeds, checkpoints=checkpoints)

    # Save to Spreadsheet
    write_csv(save_path, "shared_noiseless", shared_noiseless_means)
    write_csv(save_path, "unshared_noiseless", unshared_noiseless_means)
    write_csv(save_path, "shared_noise", shared_noise_means)
    write_csv(save_path, "unshared_noise", unshared_noise_means)

    # Plot Rewards
    plot_reward(
        [shared_noiseless_means, shared_noiseless_errors],
        [unshared_noiseless_means, unshared_noiseless_errors],
        [shared_noise_means, shared_noise_errors],
        [unshared_noise_means, unshared_noise_errors],
        checkpoints=checkpoints,
        save_path=save_path
    )

    plot_distributions(shared_noiseless_means, unshared_noiseless_means, shared_noise_means, unshared_noise_means, checkpoints, save_path)

def plot_reward(shared_noiseless_stats, unshared_noiseless_stats, shared_noise_stats, unshared_noise_stats, checkpoints, save_path):
    labels = [str(checkpoint) for checkpoint in checkpoints]
    
    episode_means = {
        'Shared Policy, Clear Channel': np.mean(shared_noiseless_stats[0], axis=0),
        'Unshared Policy, Clear Channel': np.mean(unshared_noiseless_stats[0], axis=0),
        'Shared Policy, Noisy Channel': np.mean(shared_noise_stats[0], axis=0),
        'Unshared Policy, Noisy Channel': np.mean(unshared_noise_stats[0], axis=0)
    }
    episode_errs = {
        'Shared Policy, Clear Channel': shared_noiseless_stats[1],
        'Unshared Policy, Clear Channel': unshared_noiseless_stats[1],
        'Shared Policy, Noisy Channel': shared_noise_stats[1],
        'Unshared Policy, Noisy Channel': unshared_noise_stats[1]
    }

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0 
    

    fig, ax = plt.subplots(figsize=(20,15), layout='constrained')

    for attribute, measurement in episode_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=episode_errs[attribute])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Reward')
    ax.set_ylabel('Model Checkpoint')
    ax.set_title("Mean Episode Reward at Each Checkpoint")
    ax.set_xticks(x+width, labels)
    ax.legend()

    plt.savefig(save_path+"mean_rewards.png", dpi=1000)
    plt.close()

def plot_distributions(shared_noiseless_means, unshared_noiseless_means, shared_noise_means, unshared_noise_means, checkpoints, save_path):
    for idx, checkpoint in enumerate(checkpoints):
        save_path_checkpoint = save_path+str(checkpoint)+"/"    
        if not os.path.isdir(save_path_checkpoint):
            os.makedirs(save_path_checkpoint)
        plt.hist(shared_noiseless_means[:, idx])
        plt.savefig(save_path_checkpoint+"mean_rewards_distribution.png")
        plt.close()


if __name__ == "__main__":

    # Checkpoint Paths
    universal_path_shared_one =  "outputs/Final Models/2024-02-01/01-33-36/maddpg_simple_reference_mlp__0370dbb7_24_02_01-01_33_36/checkpoints/checkpoint_"
    universal_path_shared_two = "outputs/2024-03-10/16-42-00/maddpg_simple_reference_mlp__ba90da34_24_03_10-16_42_00/checkpoints/checkpoint_"
    universal_path_shared_three = "outputs/2024-03-11/10-59-22/maddpg_simple_reference_mlp__4fb2c2eb_24_03_11-10_59_22/checkpoints/checkpoint_"
    
    universal_path_unshared_one = "outputs/Final Models/10-31-32/maddpg_simple_reference_mlp__d5090c0f_24_01_15-10_31_32/checkpoints/checkpoint_"
    universal_path_unshared_two = "outputs/23-00-33/maddpg_simple_reference_mlp__c900fd76_24_03_11-23_00_35/checkpoints/checkpoint_"
    universal_path_unshared_three = "outputs/08-09-22/maddpg_simple_reference_mlp__f34d3da6_24_03_13-08_09_29/checkpoints/checkpoint_"

    noise_path_shared_one = "outputs/Final Models/00-54-38/maddpg_simple_reference_idiolect_mlp__913becc1_24_02_20-00_54_38/checkpoints/checkpoint_"
    noise_path_shared_two = "outputs/2024-03-11/00-18-48/maddpg_simple_reference_idiolect_mlp__118b494c_24_03_11-00_18_48/checkpoints/checkpoint_"
    noise_path_shared_three = "outputs/2024-03-11/18-48-55/maddpg_simple_reference_idiolect_mlp__26cf9021_24_03_11-18_48_55/checkpoints/checkpoint_"

    noise_path_unshared_one = "outputs/Final Models/19-38-40/maddpg_simple_reference_idiolect_mlp__18300887_24_01_14-19_38_40/checkpoints/checkpoint_"
    noise_path_unshared_two = "outputs/2024-03-12/01-09-30/maddpg_simple_reference_idiolect_mlp__9c92345a_24_03_12-01_09_30/checkpoints/checkpoint_"
    noise_path_unshared_three = "outputs/16_21_21/checkpoint_"

    shared_noiseless_paths = [universal_path_shared_one, universal_path_shared_two, universal_path_shared_three]
    unshared_noiseless_paths = [universal_path_unshared_one, universal_path_unshared_two, universal_path_unshared_three]
    shared_noise_paths = [noise_path_shared_one, noise_path_shared_two, noise_path_shared_three]
    unshared_noise_paths = [noise_path_unshared_one, noise_path_unshared_two, noise_path_unshared_three]

    # Checkpoints
    checkpoints = range(300000, 5700000, 300000)

    # Seeds
    seeds = 5

    plot_means(shared_noiseless_paths, unshared_noiseless_paths, shared_noise_paths, unshared_noise_paths, seeds, checkpoints, new=False, save_path="evaluation/graphs/3-15-update/old_evals/")
    plot_means(shared_noiseless_paths, unshared_noiseless_paths, shared_noise_paths, unshared_noise_paths, seeds, checkpoints, new=True, save_path="evaluation/graphs/3-15-update/new_evals/")