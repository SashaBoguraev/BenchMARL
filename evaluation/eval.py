from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def run_benchmark(task, PATH, seed):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # You can override from the script
    experiment_config.train_device = "cpu"  # Change the training device
    experiment_config.off_policy_n_envs_per_worker = 10_000
    experiment_config.off_policy_collected_frames_per_batch = 1_000_000
    experiment_config.max_n_frames = 1_000_000
    experiment_config.evaluation = False
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.checkpoint_interval = 100_000_000

    # Some basic other configs
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

    x = torch.load(PATH)

    policy = experiment.algorithm.get_policy_for_collection()
    policy.load_state_dict(x['collector']['policy_state_dict'])
    experiment.policy = policy
    experiment.run(eval = True)
    reward = experiment.reward
    episode_reward = experiment.episode_reward

    stats, mean_stats, to_graphs = process_rewards(reward, episode_reward)

    return stats, mean_stats, to_graphs

def process_rewards(reward, episode_reward):
    # Get the individual rewards for each environment
    reward = torch.squeeze(reward, -1)[:,:,0]
    max_rewards, _ = torch.max(reward.clone().detach(), dim=1, keepdim=True)
    min_rewards, _ = torch.min(reward.clone().detach(), dim=1, keepdim=True)
    mean_rewards = torch.mean(reward.clone().detach(), dim=1, keepdim=True)

    # Find the last time the rewards dip under a certain threshold (speed)
    def find_speed(row, threshold):
        out = float('nan')
        for idx, val in enumerate(row):
            if (-1*val)<threshold:
                return idx
        return float('nan')
    
    # Return the speed for each row
    speed_data = map(lambda row: find_speed(row, threshold=.2), reward.clone().detach())
    speed_tensor = torch.Tensor(list(speed_data))
    speed_length = torch.sum(torch.isnan(speed_tensor)).item()

    thresholds = []
    num_nans = []
    speed_means = []

    num_steps = 20

    for step in np.linspace(0, 2, num_steps):
        # thresh = thresh - i*inc
        speed = map(lambda row: find_speed(row, threshold=step), reward.clone().detach())
        speed = torch.Tensor(list(speed))
        speed_mean = np.nanmean(speed)
        num_nan = torch.sum(torch.isnan(speed)).item()

        thresholds.append(step)
        num_nans.append(num_nan)
        speed_means.append(speed_mean)

    # Get Unique Episode Rewards
    episode_reward = episode_reward[::2]

    stats = {
        "Max Rewards": max_rewards.squeeze(),
        "Min Rewards": min_rewards.squeeze(),
        "Mean Rewards": mean_rewards.squeeze(),
        "Episode Rewards": episode_reward,
        "Speeds": speed_tensor,
        "Speed Length": speed_length
    }

    mean_stats = {
        "Max Rewards": max_rewards.squeeze().mean().item(),
        "Min Rewards": min_rewards.squeeze().mean().item(),
        "Mean Rewards": mean_rewards.squeeze().mean().item(),
        "Episode Rewards": episode_reward.mean().item(),
        "Speeds": np.nanmean(speed_tensor),
        "Num Nans": speed_length
    }

    to_graphs = {
        "thresholds": thresholds,
        "num nans": num_nans,
        "speed means": speed_means
    }

    return stats, mean_stats, to_graphs

def compare_environments(eval_one, eval_two, alt):
    keys = eval_one.keys()
    p_vals = []
    for idx, key in enumerate(keys):
        if idx < 5:
            data_one = eval_one[key].tolist()
            data_two = eval_two[key].tolist()
            stat, p = ttest_ind(data_one, data_two, alternative=alt, nan_policy='omit')
            p_vals.append(p)    
    return p_vals

def write_csv(filename, data, arrays=True):
    with open(filename, 'w+', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        if arrays:
            for i in range(len(data[fieldnames[0]])):
                row_data = {field: data[field][i] for field in fieldnames}
                writer.writerow(row_data)
        else:
            row_data = {field: data[field] for field in fieldnames}
            writer.writerow(row_data)

def eval_pairs(pairs, out_dict):
    for pair in pairs:
        first = pair[0]
        second = pair[1]
        p_less = compare_environments(first, second, 'less')
        p_greater = compare_environments(first, second, 'greater')
        out_dict["less"].append(p_less)
        out_dict["greater"].append(p_greater)

def graph_data(datasets, titles, type):

    title = type+" Environments"
    threshold = datasets[0]["thresholds"]

    fig, axs = plt.subplots(2, figsize=(15, 15))
    axs[0].set_title('Speed')
    axs[0].set(xlabel='Negative Threshold', ylabel='Speed')
    axs[1].set_title('Number of Incompletions')
    axs[1].set(xlabel='Negative Threshold', ylabel='Number of Incompletions')

    for idx, data in enumerate(datasets):
        nan = data["num nans"]
        speed = data["speed means"]

        label = titles[idx]

        axs[0].plot(threshold, speed, label=label)
        axs[1].plot(threshold, nan, label=label)
    
    axs[0].legend()
    axs[1].legend()

    save_path = type+'_graph.png'
    plt.savefig(save_path)

def generate_data(paths, seed):
    # Get Paths
    universal_path = paths[0]
    noise_path = paths[1]
    mem_path = paths[2]
    noise_mem_path = paths[3]

    # Get Stats for old environment
    universal_old, universal_old_means, universal_old_graphs = run_benchmark(IdiolectEvoTask.SPEED_OLD.get_from_yaml(), universal_path, seed)
    noise_old, noise_old_means, noise_old_graphs = run_benchmark(IdiolectEvoTask.SPEED_OLD_NOISE.get_from_yaml(), noise_path, seed)
    mem_old, mem_old_means, mem_old_graphs = run_benchmark(IdiolectEvoTask.SPEED_OLD_MEM_BUFFER.get_from_yaml(), mem_path, seed)
    noise_mem_old, noise_mem_old_means, noise_mem_old_graphs = run_benchmark(IdiolectEvoTask.SPEED_OLD_NOISE_MEM.get_from_yaml(), noise_mem_path, seed)
    old_evals = [universal_old, noise_old, mem_old, noise_mem_old]
    old_pairs = list(itertools.combinations(old_evals, 2))
    old_graphs = [universal_old_graphs, noise_old_graphs, mem_old_graphs, noise_mem_old_graphs]

    # Get Stats for new environment
    universal_new, universal_new_means, universal_new_graphs = run_benchmark(IdiolectEvoTask.SPEED_NEW.get_from_yaml(), universal_path, seed)
    noise_new, noise_new_means, noise_new_graphs = run_benchmark(IdiolectEvoTask.SPEED_NEW_NOISE.get_from_yaml(), noise_path, seed)
    mem_new, mem_new_means, mem_new_graphs = run_benchmark(IdiolectEvoTask.SPEED_NEW_MEM_BUFFER.get_from_yaml(), mem_path, seed)
    noise_mem_new, noise_mem_new_means, noise_mem_new_graphs = run_benchmark(IdiolectEvoTask.SPEED_NEW_NOISE_MEM.get_from_yaml(), noise_mem_path, seed)
    new_evals = [universal_new, noise_new, mem_new, noise_mem_new]
    new_pairs = list(itertools.combinations(new_evals, 2))
    new_graphs = [universal_new_graphs, noise_new_graphs, mem_new_graphs, noise_mem_new_graphs]

    titles = ["Universal", "Noise", "Attention-Based Memory", "Both"]

    # Get names for all possible pairs
    pairs = [
        "universal - noise",
        "universal - mem",
        "universal - noise_mem",
        "noise - mem",
        "noise - noise_mem",
        "mem - noise_mem",
    ]

    # Initialize the dictionaries for all p-values
    p_vals_old = {
        "pairs": pairs,
        "less": [],
        "greater": []
    }
    p_vals_new = {
        "pairs": pairs,
        "less": [],
        "greater": []
    }

    # Populate P-Value Dictionaries
    eval_pairs(old_pairs, p_vals_old)
    eval_pairs(new_pairs, p_vals_new)

    # Initialize the dictionaries for all means
    means_old = {
        "universal": universal_old_means,
        "noise": noise_old_means,
        "attention": mem_old_means,
        "both": noise_mem_old_means
    }
    means_new = {
        "universal": universal_new_means,
        "noise": noise_new_means,
        "attention": mem_new_means,
        "both": noise_mem_new_means
    }
    
    # Write results to files
    output_folder = '/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/stats'
    write_csv(os.path.join(output_folder, 'p_vals_old_seed'+str(seed)+'.csv'), p_vals_old)
    write_csv(os.path.join(output_folder, 'p_vals_new'+str(seed)+'.csv'), p_vals_new)
    write_csv(os.path.join(output_folder, 'means_old'+str(seed)+'.csv'), means_old, False)
    write_csv(os.path.join(output_folder, 'means_new'+str(seed)+'.csv'), means_new, False)

    # Graph Data
    graph_data(old_graphs, titles, "Old_seed"+str(seed))
    graph_data(new_graphs, titles, "Novel_seed"+str(seed))

if __name__ == "__main__":

    # Checkpoint paths
    universal_path_final = "evaluation/checkpoints/final/universal.pt"
    noise_path_final = "evaluation/checkpoints/final/noise.pt"
    mem_path_final = "evaluation/checkpoints/final/mem_buffer.pt"
    noise_mem_path_final = "evaluation/checkpoints/final/noise_mem.pt"
    final_paths = [universal_path_final, noise_path_final, mem_path_final, noise_mem_path_final]

    universal_path_sim = "evaluation/checkpoints/sim_reward/universal_42_38.pt"
    # noise_path_sim = "evaluation/checkpoints/sim_reward/noise_41_725.pt"
    noise_path_sim = "outputs/2023-12-27/21-35-30/maddpg_simple_reference_idiolect_mlp__6353b4cc_23_12_27-21_35_31/checkpoints/checkpoint_7500000.pt"
    mem_path_sim = "evaluation/checkpoints/sim_reward/mem_buffer_41_841.pt"
    noise_mem_path_sim = "evaluation/checkpoints/sim_reward/both_41_725.pt"
    sim_paths = [universal_path_sim, noise_path_sim, mem_path_sim, noise_mem_path_sim]

    # Seed
    seeds = 10

    # Generate Everything
    for seed in range(seeds):
        print("SEED ", seed)
        seed = seed + 10
        generate_data(sim_paths, seed)