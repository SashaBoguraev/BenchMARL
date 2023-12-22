from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os
import numpy as np
from scipy.stats import ttest_ind

def run_benchmark(task, PATH):
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
    seed = 0

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

    stats = process_rewards(reward, episode_reward)

    return stats

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
    speed_data = map(lambda row: find_speed(row, threshold=.5), reward.clone().detach())
    speed_tensor = torch.Tensor(list(speed_data))
    speed_length = torch.sum(torch.isinf(speed_tensor)).item()

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

    return stats

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

def write_csv(filename, data):
    with open(filename, 'w+', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(data[fieldnames[0]])):
            row_data = {field: data[field][i] for field in fieldnames}
            writer.writerow(row_data)

def eval_pairs(pairs, out_dict):
    for pair in pairs:
        first = pair[0]
        second = pair[1]
        p_less = compare_environments(first, second, 'less')
        p_greater = compare_environments(first, second, 'greater')
        out_dict["less"].append(p_less)
        out_dict["greater"].append(p_greater)

if __name__ == "__main__":
    # Get checkpoint paths
    universal_path = "evaluation/checkpoints/final/universal.pt"
    noise_path = "evaluation/checkpoints/final/noise.pt"
    mem_path = "evaluation/checkpoints/final/mem_buffer.pt"
    noise_mem_path = "evaluation/checkpoints/final/noise_mem.pt"

    # Get Stats for old environment
    universal_old = run_benchmark(IdiolectEvoTask.SPEED_OLD.get_from_yaml(), universal_path)
    noise_old = run_benchmark(IdiolectEvoTask.SPEED_OLD_NOISE.get_from_yaml(), noise_path)
    mem_old = run_benchmark(IdiolectEvoTask.SPEED_OLD_MEM_BUFFER.get_from_yaml(), mem_path)
    noise_mem_old = run_benchmark(IdiolectEvoTask.SPEED_OLD_NOISE_MEM.get_from_yaml(), noise_mem_path)
    old_evals = [universal_old, noise_old, mem_old, noise_mem_old]
    old_pairs = list(itertools.combinations(old_evals, 2))

    # Get Stats for new environment
    universal_new = run_benchmark(IdiolectEvoTask.SPEED_NEW.get_from_yaml(), universal_path)
    noise_new = run_benchmark(IdiolectEvoTask.SPEED_NEW_NOISE.get_from_yaml(), noise_path)
    mem_new = run_benchmark(IdiolectEvoTask.SPEED_NEW_MEM_BUFFER.get_from_yaml(), mem_path)
    noise_new = run_benchmark(IdiolectEvoTask.SPEED_NEW_NOISE_MEM.get_from_yaml(), noise_mem_path)
    new_evals = [universal_new, noise_new, mem_new, noise_new]
    new_pairs = list(itertools.combinations(new_evals, 2))

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

    eval_pairs(old_pairs, p_vals_old)
    eval_pairs(new_pairs, p_vals_new)
    
    output_folder = '/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/p_vals'
    write_csv(os.path.join(output_folder, 'p_vals_old.csv'), p_vals_old)
    write_csv(os.path.join(output_folder, 'p_vals_new.csv'), p_vals_new)