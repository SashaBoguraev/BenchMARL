from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.benchmark import Benchmark
from pathlib import Path
import json, torch

def run_benchmark(task):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # You can override from the script
    experiment_config.train_device = "cpu"  # Change the training device
    experiment_config.off_policy_n_envs_per_worker = 2
    experiment_config.off_policy_collected_frames_per_batch = 200
    experiment_config.max_n_frames = 200
    experiment_config.evaluation: False
    experiment_config.render: False
    experiment_config.loggers = ["csv"]

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

    min_returns = None
    max_returns = None
    mean_returns = None

    x = torch.load("/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/outputs/2023-11-20/FIrst Universal/maddpg_simple_reference_mlp__3141fed9_23_11_20-14_40_19/checkpoints/checkpoint_9900000.pt")

    policy = experiment.algorithm.get_policy_for_collection()
    policy.load_state_dict(x['collector']['policy_state_dict'])
    experiment.policy = policy
    experiment.run(eval = True)
    min_returns = experiment.min_returns
    max_returns = experiment.max_returns
    mean_returns = experiment.mean_returns

    return min_returns, max_returns, mean_returns

if __name__ == "__main__":
    outputs = run_benchmark(IdiolectEvoTask.SPEED_OLD.get_from_yaml())
    print(outputs)