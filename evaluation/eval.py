from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.benchmark import Benchmark

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # You can override from the script
    experiment_config.train_device = "cpu"  # Change the training device
    experiment_config.off_policy_n_envs_per_worker = 1
    experiment_config.off_policy_collected_frames_per_batch = 100
    experiment_config.max_n_iters = 1
    experiment_config.max_n_frames: 100
    experiment_config.evaluation: True
    experiment_config.render: True
    experiment_config.evaluation_interval: 100
    experiment_config.evaluation_episodes: 1
    experiment_config.restore_file = "/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/outputs/2023-11-20/FIrst Universal/maddpg_simple_reference_mlp__3141fed9_23_11_20-14_40_19/checkpoints/checkpoint_9900000.pt"

    # Some basic other configs
    tasks = [IdiolectEvoTask.SPEED_OLD.get_from_yaml()]
    algorithm_configs = [MaddpgConfig.get_from_yaml()]
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    seeds = range(100_000)

    benchmark = Benchmark(
        algorithm_configs = algorithm_configs,
        tasks = tasks,
        seeds = seeds,
        experiment_config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )
    benchmark.run_sequential()