from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch
import numpy as np

def run_benchmark(task, PATH):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # You can override from the script
    experiment_config.train_device = "cpu"  # Change the training device
    experiment_config.off_policy_n_envs_per_worker = 100
    experiment_config.off_policy_collected_frames_per_batch = 10_000
    experiment_config.max_n_frames = 10_000
    experiment_config.evaluation: False
    experiment_config.render: False
    experiment_config.loggers = []

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
    def find_last(row, threshold):
        last_val = np.inf
        out = -np.inf
        for idx, val in enumerate(row):
            if (-1*last_val) > threshold and (-1*val)<threshold:
                out = idx
            last_val = val
        return out
    
    # Return the speed for each row
    speed_tensor = map(lambda row: find_last(row, threshold=.25), reward.clone().detach())
    speed_tensor = torch.Tensor(list(speed_tensor))


    # Get Unique Episode Rewards
    episode_reward = episode_reward[::2]

    stats = {
        "Max Rewards": max_rewards,
        "Min Rewards": min_rewards,
        "Mean Rewards": mean_rewards,
        "Episode Rewards": episode_reward,
        "Speeds": speed_tensor,
    }

    return stats


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

    # Get Stats for new environment
    universal_new = run_benchmark(IdiolectEvoTask.SPEED_NEW.get_from_yaml(), universal_path)
    noise_new = run_benchmark(IdiolectEvoTask.SPEED_NEW_NOISE.get_from_yaml(), noise_path)
    mem_new = run_benchmark(IdiolectEvoTask.SPEED_NEW_MEM_BUFFER.get_from_yaml(), mem_path)
    noise_new = run_benchmark(IdiolectEvoTask.SPEED_NEW_NOISE_MEM.get_from_yaml(), noise_mem_path)

    

    