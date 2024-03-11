from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def run_benchmark(task, PATH, seed, share_params):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # You can override from the script
    experiment_config.train_device = "cpu"  # Change the training device
    experiment_config.off_policy_n_envs_per_worker = 1_000
    experiment_config.off_policy_collected_frames_per_batch = 100_000
    experiment_config.max_n_frames = 100_000
    experiment_config.evaluation = False
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.checkpoint_interval = 10_000_000
    experiment_config.share_policy_params = share_params

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
    experiment = experiment.load_experiment_policy(x)
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
    speeds = []

    num_steps = 20

    for step in np.linspace(0, 2, num_steps):
        # thresh = thresh - i*inc
        speed = map(lambda row: find_speed(row, threshold=step), reward.clone().detach())
        speed = torch.Tensor(list(speed))
        speeds.append(speed)
        speed_mean = np.nanmean(speed)
        num_nan = torch.sum(torch.isnan(speed)).item()

        thresholds.append(step)
        num_nans.append(num_nan)
        speed_means.append(speed_mean)

    stats = {
        "Max Rewards": max_rewards.squeeze(),
        "Min Rewards": min_rewards.squeeze(),
        "Mean Rewards": mean_rewards.squeeze(),
        "Episode Rewards": episode_reward,
        "Speeds": speed_tensor,
        "All Speed Data": [np.nanmean(tsr) for tsr in speeds],
        "Speed Length": speed_length,
        "Percent Nans": [x/10000 for x in num_nans]
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
            stat, p = st.ttest_ind(data_one, data_two, alternative=alt, nan_policy='omit')
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

# def graph_speed_nans(datasets, titles, type):

#     title = type+" Environments"
#     threshold = datasets[0]["thresholds"]

#     fig, axs = plt.subplots(2, figsize=(15, 15))
#     axs[0].set_title('Speed')
#     axs[0].set(xlabel='Negative Threshold', ylabel='Speed')
#     axs[1].set_title('Number of Incompletions')
#     axs[1].set(xlabel='Negative Threshold', ylabel='Number of Incompletions')

#     for idx, data in enumerate(datasets):
#         nan = np.array(data["num nans"])
#         speed = np.array(data["speed means"])

#         # Calculate mean and standard deviation along the axis 0 (columns)
#         mean_nan = np.mean(nan, axis=0)
#         std_dev_nan = np.std(nan, axis=0)

#         mean_speed = np.mean(speed, axis=0)
#         std_dev_speed = np.std(speed, axis=0)

#         # Calculate 95% confidence interval
#         confidence_interval_nan = 1.96 * (std_dev_nan / np.sqrt(len(nan)))
#         confidence_interval_speed = 1.96 * (std_dev_speed / np.sqrt(len(speed)))

#         label = titles[idx]

#         axs[0].plot(threshold, mean_speed, label=label)
#         axs[0].fill_between(threshold, mean_speed-(confidence_interval_speed), mean_speed+(confidence_interval_speed), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        
#         axs[1].plot(threshold, mean_nan, label=label)
#         axs[1].fill_between(threshold, mean_nan-(confidence_interval_nan), mean_nan+(confidence_interval_nan), alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    
#     axs[0].legend()
#     axs[1].legend()

#     save_path = type+'_graph.png'
#     plt.savefig(save_path)

def generate_data(paths, seed, share_params, noise = False):
    # NOISE PARAMETER REPRESENTS WETHER NOISY COMPARISONS OR NON-NOISY
    old_task = VmasTask.SIMPLE_REFERENCE.get_from_yaml() if not noise else VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml()
    new_task = IdiolectEvoTask.SPEED_NEW.get_from_yaml() if not noise else IdiolectEvoTask.SPEED_NEW_NOISE.get_from_yaml()

    # Get Paths
    universal_path = paths[0]
    noise_path = paths[1]

    # Get Stats for old environment
    universal_old, universal_old_means, universal_old_graphs = run_benchmark(old_task, universal_path, seed, share_params[0])
    noise_old, noise_old_means, noise_old_graphs = run_benchmark(old_task, noise_path, seed, share_params[1])
    old_evals = [universal_old, noise_old]
    old_means = [universal_old_means, noise_old_means]
    old_pairs = list(itertools.combinations(old_evals, 2))
    old_graphs = [universal_old_graphs, noise_old_graphs]

    # Get Stats for new environment
    universal_new, universal_new_means, universal_new_graphs = run_benchmark(new_task, universal_path, seed, share_params[0])
    noise_new, noise_new_means, noise_new_graphs = run_benchmark(new_task, noise_path, seed, share_params[1])
    new_evals = [universal_new, noise_new]
    new_means = [universal_new_means, noise_new_means]
    new_pairs = list(itertools.combinations(new_evals, 2))
    new_graphs = [universal_new_graphs, noise_new_graphs]

    # Get names for all possible pairs
    pairs = [
        "universal - noise",
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
    }
    means_new = {
        "universal": universal_new_means,
        "noise": noise_new_means,
    }
    
    # Write results to files
    output_folder = '/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/stats'
    write_csv(os.path.join(output_folder, 'p_vals_old_seed'+str(seed)+'.csv'), p_vals_old)
    write_csv(os.path.join(output_folder, 'p_vals_new'+str(seed)+'.csv'), p_vals_new)
    write_csv(os.path.join(output_folder, 'means_old'+str(seed)+'.csv'), means_old, False)
    write_csv(os.path.join(output_folder, 'means_new'+str(seed)+'.csv'), means_new, False)

    return old_evals, new_evals, old_means, new_means, old_graphs, new_graphs

def graph_seed(old_evals, new_evals, seed):
    # Isolate Data
    old_max_uni = old_evals[0]["Max Rewards"]
    old_max_noisy = old_evals[1]["Max Rewards"]
    new_max_uni = new_evals[0]["Max Rewards"]
    new_max_noisy = new_evals[1]["Max Rewards"]

    old_mean_uni = old_evals[0]["Mean Rewards"]
    old_mean_noisy = old_evals[1]["Mean Rewards"]
    new_mean_uni = new_evals[0]["Mean Rewards"]
    new_mean_noisy = new_evals[1]["Mean Rewards"]

    old_speed_uni = old_evals[0]["Percent Nans"]
    old_speed_noisy = old_evals[1]["Percent Nans"]
    new_speed_uni = new_evals[0]["Percent Nans"]
    new_speed_noisy = new_evals[1]["Percent Nans"]

    save_path = "/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/graphs/seed"+str(seed)+"/"

    # Plot Data
    def plot_rewards(old_uni, old_noisy, new_uni, new_noisy, stat: str, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        labels = ['In-Distribution Environment', 'Novel Environment']
        all_stats = [old_uni, old_noisy, new_uni, new_noisy]
        means = []
        errors = []
        for statistic in all_stats:
            n = len(np.array(statistic))
            mean = np.mean(np.array(statistic))
            std_dev = np.std(np.array(statistic), axis=0)
            se = (std_dev / np.sqrt(n))
            # Calculate 95% confidence interval
            margin = 1.96 * se

            means.append(mean)
            errors.append(margin)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, [means[0], means[2]], width, label='Universal Population', yerr=[errors[0], errors[2]])
        rects1 = ax.bar(x + width/2, [means[1], means[3]], width, label='Idiolect Population', yerr=[errors[1], errors[3]])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Reward')
        ax.set_title(stat)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.savefig(save_path+stat+".png")
    
    def plot_rewards_const_env(old_uni, old_noisy, stat: str, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        labels = ['In-Distribution Environment']
        all_stats = [old_uni, old_noisy]
        means = []
        errors = []
        for statistic in all_stats:
            mean = np.mean(np.array(statistic))
            std_dev = np.std(np.array(statistic), axis=0)
            # Calculate 95% confidence interval
            margin = get_error(statistic)

            means.append(mean)
            errors.append(margin)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, [means[0]], width, label='Universal Population', yerr=[errors[0]])
        rects1 = ax.bar(x + width/2, [means[1]], width, label='Idiolect Population', yerr=[errors[1]])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Reward')
        ax.set_title(stat)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.savefig(save_path+stat+".png")

    plot_rewards(old_max_uni, old_max_noisy, new_max_uni, new_max_noisy, "Maximum Reward", save_path)
    plot_rewards(old_mean_uni, old_mean_noisy, new_mean_uni, new_mean_noisy, "Mean Reward", save_path)
    plot_nans(old_speed_uni, old_speed_noisy, new_speed_uni, new_speed_noisy, save_path)

def plot_speed_nans(speed, old_uni, old_noisy, new_uni, new_noisy, save_path):
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    fig, axs = plt.subplots(2, figsize=(15, 15))
    
    axs[0].set_title('In-Distribution Environment Completion')
    if speed:
        axs[0].set(xlabel='Negative Threshold', ylabel='Speed of Environments Completion')
    else:
        axs[0].set(xlabel='Negative Threshold', ylabel='Number of Environments Incomplete')
    axs[1].set_title('Novel Environment Completion')
    if speed:
        axs[1].set(xlabel='Negative Threshold', ylabel='Speed of Environments Completion')
    else:
        axs[1].set(xlabel='Negative Threshold', ylabel='Number of Environments Incomplete')

    speed_old_uni_ci = get_error(old_uni)
    speed_old_noisy_ci = get_error(old_noisy)
    speed_new_uni_ci = get_error(new_uni)
    speed_new_noisy_ci = get_error(new_noisy)

    threshold = np.linspace(0, 2, 20)

    axs[0].plot(threshold, old_uni, label="Universal Population")
    axs[0].fill_between(threshold, old_uni-(speed_old_uni_ci), old_uni+(speed_old_uni_ci), alpha = .3, edgecolor='#1B2ACC', facecolor='#089FFF')
    axs[0].plot(threshold, old_noisy, label="Idiolect Population")
    axs[0].fill_between(threshold, old_noisy-(speed_old_noisy_ci), old_noisy+(speed_old_noisy_ci), alpha = .3, edgecolor='#CC4F1B', facecolor='#FF9848')

    axs[1].plot(threshold, new_uni, label="Universal Population")
    axs[1].fill_between(threshold, new_uni-(speed_new_uni_ci), new_uni+(speed_new_uni_ci), alpha = .3, edgecolor='#1B2ACC', facecolor='#089FFF')
    axs[1].plot(threshold, new_noisy, label="Idiolect Population")
    axs[1].fill_between(threshold, new_noisy-(speed_new_noisy_ci), new_noisy+(speed_new_noisy_ci), alpha = .3, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    axs[0].legend()
    axs[1].legend()

    plt.savefig(save_path+"nan.png") if not speed else plt.savefig(save_path+"speed.png")

def get_error(data):
    # Calculate 95% confidence interval
    data = np.array(data)
    
    interval = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    margin = (interval[1]-interval[0])/2

    return margin
    
def graph_stats(evals, folder = ""):
    
    new_max_uni = []
    new_max_noisy = []
    new_mean_uni = []
    new_mean_noisy = []
    new_speeds_uni = []
    new_speeds_noisy = []
    new_nans_uni = []
    new_nans_noisy = []

    for eval in evals["new"]:
        new_max_uni.append(eval[0]["Max Rewards"])
        new_max_noisy.append(eval[1]["Max Rewards"])
        new_mean_uni.append(eval[0]["Mean Rewards"])
        new_mean_noisy.append(eval[1]["Mean Rewards"])
        new_speeds_uni.append(eval[0]["All Speed Data"])
        new_speeds_noisy.append(eval[1]["All Speed Data"])
        new_nans_uni.append(eval[0]["Percent Nans"])
        new_nans_noisy.append(eval[1]["Percent Nans"])
    
    old_max_uni = []
    old_max_noisy = []
    old_mean_uni = []
    old_mean_noisy = []
    old_speeds_uni = []
    old_speeds_noisy = []
    old_nans_uni = []
    old_nans_noisy = []

    for eval in evals["old"]:
        old_max_uni.append(eval[0]["Max Rewards"])
        old_max_noisy.append(eval[1]["Max Rewards"])
        old_mean_uni.append(eval[0]["Mean Rewards"])
        old_mean_noisy.append(eval[1]["Mean Rewards"])
        old_speeds_uni.append(eval[0]["All Speed Data"])
        old_speeds_noisy.append(eval[1]["All Speed Data"])
        old_nans_uni.append(eval[0]["Percent Nans"])
        old_nans_noisy.append(eval[1]["Percent Nans"])
    
    old_rewards_uni = []
    old_rewards_noisy = []
    new_rewards_uni = []
    new_rewards_noisy = []

    for eval in evals["old"]:
        old_rewards_uni.append(eval[0]["Episode Rewards"])
        old_rewards_noisy.append(eval[1]["Episode Rewards"])

    for eval in evals["new"]:
        new_rewards_uni.append(eval[0]["Episode Rewards"])
        new_rewards_noisy.append(eval[1]["Episode Rewards"])

    new_max_uni = np.mean(new_max_uni, axis = 1)
    new_max_noisy = np.mean(new_max_noisy, axis = 1)
    new_mean_uni = np.mean(new_mean_uni, axis = 1)
    new_mean_noisy = np.mean(new_mean_noisy, axis = 1)

    old_max_uni = np.mean(old_max_uni, axis = 1)
    old_max_noisy = np.mean(old_max_noisy, axis = 1)
    old_mean_uni = np.mean(old_mean_uni, axis = 1)
    old_mean_noisy = np.mean(old_mean_noisy, axis = 1)

    old_rewards_uni = np.mean(old_rewards_uni, axis = 1)
    old_rewards_noisy = np.mean(old_rewards_noisy, axis = 1)
    new_rewards_uni = np.mean(new_rewards_uni, axis = 1)
    new_rewards_noisy = np.mean(new_rewards_noisy, axis = 1)

    print(new_max_uni.shape)
    
    flat = lambda a : np.array(a).flatten()

    new_max_uni = flat(new_max_uni)
    new_max_noisy = flat(new_max_noisy)
    new_mean_uni = flat(new_mean_uni)
    new_mean_noisy = flat(new_mean_noisy)

    old_max_uni = flat(old_max_uni)
    old_max_noisy = flat(old_max_noisy)
    old_mean_uni = flat(old_mean_uni)
    old_mean_noisy = flat(old_mean_noisy)

    old_rewards_uni = flat(old_rewards_uni)
    old_rewards_noisy = flat(old_rewards_noisy)
    new_rewards_uni = flat(new_rewards_uni)
    new_rewards_noisy = flat(new_rewards_noisy)

    # Speeds and Nans

    old_speeds_uni = np.nanmean(old_speeds_uni, axis=0)
    old_speeds_noisy = np.nanmean(old_speeds_noisy, axis=0)
    old_nans_uni = np.nanmean(old_nans_uni, axis=0)
    old_nans_noisy = np.nanmean(old_nans_noisy, axis=0)
    
    new_speeds_uni = np.nanmean(new_speeds_uni, axis=0)
    new_speeds_noisy = np.nanmean(new_speeds_noisy, axis=0)
    new_nans_uni = np.nanmean(new_nans_uni, axis=0)
    new_nans_noisy = np.nanmean(new_nans_noisy, axis=0)

    num = len(new_max_uni)

    save_path = "/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/graphs/3-15-update/"+folder

    if not os.path.isdir(save_path):
            os.makedirs(save_path)

    plot_reward(old_max_uni, old_max_noisy, new_max_uni, new_max_noisy, num, "Maximum Reward", save_path=save_path)
    plot_reward(old_mean_uni, old_mean_noisy, new_mean_uni, new_mean_noisy, num, "Mean Reward", save_path=save_path)
    plot_distribution(old_max_uni, old_max_noisy, new_max_uni, new_max_noisy, num, "Maximum Reward", save_path=save_path)
    plot_distribution(old_mean_uni, old_mean_noisy, new_mean_uni, new_mean_noisy, num, "Mean Reward", save_path=save_path)
    
    plot_reward(old_rewards_uni, old_rewards_noisy, new_rewards_uni, new_rewards_noisy, num, "Episode Rewards", save_path=save_path)
    plot_distribution(old_rewards_uni, old_rewards_noisy, new_rewards_uni, new_rewards_noisy, num, "Episode Rewards", save_path=save_path)

    plot_speed_nans(True, old_speeds_uni, old_speeds_noisy, new_speeds_uni, new_speeds_noisy, save_path=save_path)
    plot_speed_nans(False, old_nans_uni, old_nans_noisy, new_nans_uni, new_nans_noisy, save_path=save_path)

def plot_reward(old_uni, old_noisy, new_uni, new_noisy, num, stat:str, save_path):

    labels = ['In-Distribution Environment', 'Novel Environment']
    all_stats = [old_uni, old_noisy, new_uni, new_noisy]
    means = []
    errors = []
    
    for statistic in all_stats:
        mean = np.mean(np.array(statistic))

        # Calculate 95% confidence interval
        margin = get_error(statistic)

        means.append(mean)
        errors.append(margin)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [means[0], means[2]], width, label='Universal Population', yerr=[errors[0], errors[2]])
    rects1 = ax.bar(x + width/2, [means[1], means[3]], width, label='Idiolect Population', yerr=[errors[1], errors[3]])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Reward')
    ax.set_title(stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig(save_path+stat+str(num)+".png")

def plot_distribution(old_uni, old_noisy, new_uni, new_noisy, num, stat, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(20,15))

    ax[0][0].set_title("Universal Population, In-Distribution Environments")
    ax[0][1].set_title("Idiolect Population, In-Distribution Environments")
    ax[1][0].set_title("Universal Population, Out-of-Distribution Environments")
    ax[1][1].set_title("Idiolect Population, Out-of-Distribution Environments")

    n1, bins1, patches1 = ax[0][0].hist(old_uni, bins=10)
    n2, bins2, patches2 = ax[0][1].hist(old_noisy, bins=10)
    n3, bins3, patches3 = ax[1][0].hist(new_uni, bins=10)
    n4, bins4, patches4 = ax[1][1].hist(new_noisy, bins=10)

    plt.savefig(save_path+stat+"_distribution"+str(num)+".png")

def make_list(trials):

    nans_universal = []
    nans_idiolect = []
    speeds_universal = []
    speeds_idiolect = []

    for data in trials:
        # Get nan and speed numbers
        nan_uni = data[0]["num nans"]
        nan_idio = data[1]["num nans"]
        
        speed_uni = data[0]["speed means"]
        speed_idio = data[1]["speed means"]

        # Append them to create 2D list
        nans_universal.append(nan_uni)
        nans_idiolect.append(nan_idio)
        speeds_universal.append(speed_uni)
        speeds_idiolect.append(speed_idio)
    
    uni_to_graphs = {
        "thresholds": trials[0][0]["thresholds"],
        "num nans": nans_universal,
        "speed means": speeds_universal
    }

    idio_to_graphs = {
        "thresholds": trials[0][0]["thresholds"],
        "num nans": nans_idiolect,
        "speed means": speeds_idiolect
    }

    return uni_to_graphs, idio_to_graphs

def plot_comparisons(sim_paths, folder = "", noise = False):

    # Initialize arrays for graphing
    old_trials = []
    new_trials = []

    evals = {"old": [], "new": []}

    # Generate Stats
    for seed in range(seeds):
        print("SEED ", seed)
        old_evals, new_evals, old_means, new_means, old_graphs, new_graphs = generate_data(sim_paths, seed, [True, False], noise=noise)
        old_trials.append(old_graphs)
        new_trials.append(new_graphs)
        evals["old"].append(old_evals)
        evals["new"].append(new_evals)
    
    # Generate Graphs
    titles = ["Universal", "Noise"]

    old_graph_list = make_list(old_trials)
    new_graph_list = make_list(new_trials)
        
    # graph_speed_nans(old_graph_list, titles, "Old")
    # graph_speed_nans(new_graph_list, titles, "Novel")

    graph_stats(evals, folder = folder)

if __name__ == "__main__":

    # Checkpoint Paths
    universal_path_shared_one =  "outputs/Final Models/2024-03-09/19-55-16/maddpg_simple_reference_mlp__9ff688df_24_03_09-19_55_17/checkpoints/checkpoint_4500000.pt"
    universal_path_shared_two = "outputs/Final Models/2024-02-01/01-33-36/maddpg_simple_reference_mlp__0370dbb7_24_02_01-01_33_36/checkpoints/checkpoint_4500000.pt"
    universal_path_shared_three = ""
    
    universal_path_unshared_one = "outputs/Final Models/10-31-32/maddpg_simple_reference_mlp__d5090c0f_24_01_15-10_31_32/checkpoints/checkpoint_4500000.pt"
    universal_path_unshared_two = "outputs/Final Models/01-00-18/maddpg_simple_reference_mlp__d06778fe_24_02_20-01_00_18/checkpoints/checkpoint_4500000.pt"
    universal_path_unshared_three = "outputs/Final Models/2024-02-01/19-08-21/maddpg_simple_reference_mlp__ef76cae9_24_02_01-19_08_21/checkpoints/checkpoint_4500000.pt"

    noise_path_shared_one = "outputs/Final Models/00-54-38/maddpg_simple_reference_idiolect_mlp__913becc1_24_02_20-00_54_38/checkpoints/checkpoint_4500000.pt"
    noise_path_shared_two = "outputs/2024-03-10/01-25-51/maddpg_simple_reference_idiolect_mlp__197fec22_24_03_10-01_25_52/checkpoints/checkpoint_4500000.pt"
    noise_path_shared_three = ""

    noise_path_unshared_one = "outputs/Final Models/19-38-40/maddpg_simple_reference_idiolect_mlp__18300887_24_01_14-19_38_40/checkpoints/checkpoint_4500000.pt"
    noise_path_unshared_two = "outputs/Final Models/2024-02-01/19-07-50/maddpg_simple_reference_idiolect_mlp__2aa7883e_24_02_01-19_07_50/checkpoints/checkpoint_4500000.pt"
    noise_path_unshared_three = ""
    
    sim_paths_noiseless_one = [universal_path_shared_one, universal_path_unshared_one]
    sim_paths_noisy_one = [noise_path_shared_one, noise_path_unshared_one]    
    
    sim_paths_noiseless_two = [universal_path_shared_two, universal_path_unshared_two]
    sim_paths_noisy_two = [noise_path_shared_two, noise_path_unshared_two]

    # Seed
    seeds = 30

    plot_comparisons(sim_paths_noiseless_one, folder="FirstCheckpoints/NoNoise/", noise = False)
    plot_comparisons(sim_paths_noisy_one, folder="FirstCheckpoints/WithNoise/", noise = True)
    plot_comparisons(sim_paths_noiseless_two, folder="SecondCheckpoints/NoNoise/", noise = False)
    plot_comparisons(sim_paths_noisy_two, folder="SecondCheckpoints/WithNoise/", noise = True)