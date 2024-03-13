from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import IdiolectEvoTask, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import torch, itertools, csv, os, random, sys, argparse
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

def compare_nested_dicts(dict1, dict2, parent_key=""):
    for key in dict1.keys():
        if key not in dict2:
            print(f"Key '{parent_key}.{key}' is present in the first dictionary but not in the second.")
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                compare_nested_dicts(dict1[key], dict2[key], f"{parent_key}.{key}" if parent_key else key)
            elif torch.is_tensor(dict1[key]) and torch.is_tensor(dict2[key]):
                if not torch.equal(dict1[key], dict2[key]):
                    print(f"Values for key '{parent_key}.{key}' differ: {dict1[key]} != {dict2[key]}")
            elif dict1[key] != dict2[key]:
                print(f"Values for key '{parent_key}.{key}' differ: {dict1[key]} != {dict2[key]}")
    print("Done Comparing!")

def sub_policy_shared(experiment_policy, policy_one, policy_two):
    agent = 0
    
    # Substitute another agent's policy
    for idx in range(3):
        layer_idx = idx*2

        # Replace layers of policy
        policy_weight0 = 'module.0.td_module.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        policy_bias0 = 'module.0.td_module.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        
        policy_weight1 = 'module.0.td_module.module.0.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.weight'
        policy_bias1 = 'module.0.td_module.module.0.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.bias'

        pop0_policy_weight = policy_one['collector']['policy_state_dict'][policy_weight0]
        pop0_policy_bias = policy_one['collector']['policy_state_dict'][policy_bias0]

        pop1_policy_weight = policy_two['collector']['policy_state_dict'][policy_weight0]
        pop1_policy_bias = policy_two['collector']['policy_state_dict'][policy_bias0]

        experiment_policy['collector']['policy_state_dict'][policy_weight0] = pop0_policy_weight
        experiment_policy['collector']['policy_state_dict'][policy_bias0] = pop0_policy_bias
        
        experiment_policy['collector']['policy_state_dict'][policy_weight1] = pop1_policy_weight
        experiment_policy['collector']['policy_state_dict'][policy_bias1] = pop1_policy_bias

        # Replace layers of actor network
        actor_weight0 = 'actor_network_params.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        actor_bias0 = 'actor_network_params.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        
        actor_weight1 = 'actor_network_params.module.0.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.weight'
        actor_bias1 = 'actor_network_params.module.0.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.bias'

        pop0_actor_weight = policy_one['loss_agents'][actor_weight0]
        pop0_actor_bias = policy_one['loss_agents'][actor_bias0]

        pop1_actor_weight = policy_two['loss_agents'][actor_weight0]
        pop1_actor_bias = policy_two['loss_agents'][actor_bias0]

        experiment_policy['loss_agents'][actor_weight0] = pop0_actor_weight
        experiment_policy['loss_agents'][actor_bias0] = pop0_actor_bias
        
        experiment_policy['loss_agents'][actor_weight1] = pop1_actor_weight
        experiment_policy['loss_agents'][actor_bias1] = pop1_actor_bias

        # Replace layers of value network

        value_network_weight0 = 'value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        value_network_bias0 = 'value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        
        value_network_weight1 = 'value_network_params.module.1.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.weight'
        value_network_bias1 = 'value_network_params.module.1.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.bias'

        pop0_value_weight = policy_one['loss_agents'][value_network_weight0]
        pop0_value_bias = policy_one['loss_agents'][value_network_bias0]

        pop1_value_weight = policy_two['loss_agents'][value_network_weight0]
        pop1_value_bias = policy_two['loss_agents'][value_network_bias0]

        experiment_policy['loss_agents'][value_network_weight0] = pop0_value_weight
        experiment_policy['loss_agents'][value_network_bias0] = pop0_value_bias
        
        experiment_policy['loss_agents'][value_network_weight1] = pop1_value_weight
        experiment_policy['loss_agents'][value_network_bias1] = pop1_value_bias

        # Replace the layers of target value network
        target_value_network_weight0 = 'target_value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        target_value_network_bias0 = 'target_value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        
        target_value_network_weight1 = 'target_value_network_params.module.1.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.weight'
        target_value_network_bias1 = 'target_value_network_params.module.1.mlp.agent_networks.'+str(agent+1)+'.'+str(layer_idx)+'.bias'

        pop0_target_value_weight = policy_one['loss_agents'][target_value_network_weight0]
        pop0_target_value_bias = policy_one['loss_agents'][target_value_network_bias0]

        pop1_target_value_weight = policy_two['loss_agents'][target_value_network_weight0]
        pop1_target_value_bias = policy_two['loss_agents'][target_value_network_bias0]

        experiment_policy['loss_agents'][target_value_network_weight0] = pop0_target_value_weight
        experiment_policy['loss_agents'][target_value_network_bias0] = pop0_target_value_bias
        
        experiment_policy['loss_agents'][target_value_network_weight1] = pop1_target_value_weight
        experiment_policy['loss_agents'][target_value_network_bias1] = pop1_target_value_bias
    
    return experiment_policy

def sub_policy_unshared(policy_one, policy_two):
    # Currently hard coded to take the first agent from each policy, 
    # put policy one's agent into the first agent of the experiment policy, 
    # and the first agent of policy two into the experiment policy

    agent = 0

    combined_policy = policy_one
    
    # Substitute another agent's policy
    for idx in range(3):
        layer_idx = idx*2

        # Replace layers of policy
        policy_weight = 'module.0.td_module.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        policy_bias = 'module.0.td_module.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        combined_policy['collector']['policy_state_dict'][policy_weight] = policy_two['collector']['policy_state_dict'][policy_weight]
        combined_policy['collector']['policy_state_dict'][policy_bias] = policy_two['collector']['policy_state_dict'][policy_bias]

        # Replace layers of actor network
        actor_weight = 'actor_network_params.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        actor_bias = 'actor_network_params.module.0.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        combined_policy['loss_agents'][actor_weight] = policy_two['loss_agents'][actor_weight]
        combined_policy['loss_agents'][actor_bias] = policy_two['loss_agents'][actor_bias]

        # Replace layers of value network
        value_network_weight = 'value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        value_network_bias = 'value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        combined_policy['loss_agents'][value_network_weight] = policy_two['loss_agents'][value_network_weight]
        combined_policy['loss_agents'][value_network_bias] = policy_two['loss_agents'][value_network_bias]

        # Replace the layers of target value network
        target_value_network_weight = 'target_value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.weight'
        target_value_network_bias = 'target_value_network_params.module.1.mlp.agent_networks.'+str(agent)+'.'+str(layer_idx)+'.bias'
        combined_policy['loss_agents'][target_value_network_weight] = policy_two['loss_agents'][target_value_network_weight]
        combined_policy['loss_agents'][target_value_network_bias] = policy_two['loss_agents'][target_value_network_bias]

    return combined_policy

def agent_integration(task, PATH_ONE, PATH_TWO, seed, agent, shared, save_path = None, train=False, vis = False):
    assert agent < 2

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml('fine_tuned/vmas/conf/vmas_parameters.yaml')

    # Some basic other configs
    algorithm_config = MaddpgConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Override Necessary Configs
    algorithm_config.share_param_critic = False
    experiment_config.share_policy_params = False
    
    if train:
        experiment_config.max_n_frames = 900_000
        experiment_config.loggers = []
    
    elif vis:
        if save_path != None and not os.path.isdir(save_path):
            os.makedirs(save_path)
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.save_folder = save_path

    else:
        experiment_config.off_policy_n_envs_per_worker = 100
        experiment_config.off_policy_collected_frames_per_batch = 10_000
        experiment_config.max_n_frames = 10_000
        experiment_config.evaluation = False
        experiment_config.render = False
        experiment_config.loggers = []
        experiment_config.checkpoint_interval = 10_000_000

    experiment = Experiment(
        algorithm_config = algorithm_config,
        task = task,
        seed = seed,
        config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )

    experiment_policy = experiment.state_dict()

    policy_one = torch.load(PATH_ONE)
    policy_two = torch.load(PATH_TWO)
                            
    combined_policy = sub_policy_unshared(policy_one, policy_two) if not shared else sub_policy_shared(experiment_policy, policy_one, policy_two)

    combined_policy['buffer_agents']['_storage']['_storage'] = policy_one['buffer_agents']['_storage']['_storage']
    combined_policy['collector']['policy_state_dict']['module.0.sigma'] = policy_one['collector']['policy_state_dict']['module.0.sigma']

    experiment = experiment.load_experiment_policy(combined_policy)
    
    episode_reward = None
    
    if not vis:
        # RUN EVALUATION
        experiment.run(eval=True)

        reward = experiment.reward
        episode_reward = experiment.episode_reward
    
    else:
        # VISUALIZE AGENTS
        experiment._evaluation_loop()
        experiment.close()
    
    return episode_reward

def select_random_checkpoints(min_one, min_two, max=9_900_000, step=300_000):
    max = 5_400_000

    range_one = list(range(min_one, max, step))
    range_two = list(range(min_two, max, step))

    checkpoint_one = random.choice(range_one)
    checkpoint_two = random.choice(range_two)

    return checkpoint_one, checkpoint_two

def graph_rewards(universal_rewards, idiolect_rewards, idiolect_wrong_roles_rewards, save_path):

    uni_mean = np.mean(universal_rewards)
    idio_mean = np.mean(idiolect_rewards)
    idio_wrong_roles_mean = np.mean(idiolect_wrong_roles_rewards)

    uni_interval = st.t.interval(confidence=0.95, df=len(universal_rewards)-1, loc=uni_mean, scale=st.sem(universal_rewards)) 
    uni_margin = (uni_interval[1]-uni_interval[0])/2

    idio_interval = st.t.interval(confidence=0.95, df=len(idiolect_rewards)-1, loc=idio_mean, scale=st.sem(idiolect_rewards)) 
    idio_margin = (idio_interval[1]-idio_interval[0])/2
    
    idio_wrong_roles_interval = st.t.interval(confidence=0.95, df=len(idiolect_wrong_roles_rewards)-1, loc=idio_wrong_roles_mean, scale=st.sem(idiolect_wrong_roles_rewards)) 
    idio_wrong_roles_margin = (idio_wrong_roles_interval[1]-idio_wrong_roles_interval[0])/2

    labels = ['New Agent Integration']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(x - width/3, uni_mean, width/3, label='Universal Population', yerr=uni_margin)
    plt.bar(x, idio_mean, width/3, label='Idiolect Population with Correct Roles', yerr=idio_margin)
    plt.bar(x + width/3, idio_wrong_roles_mean, width/3, label="Idiolect Population with Wrong Roles", yerr=idio_wrong_roles_margin)

    plt.ylabel('Reward')
    plt.title('Mean Reward')
    plt.legend()

    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    
    plt.savefig(save_path+"AgentIntegration.png")
    
    # Clear Plot
    plt.clf()

def plot_distribution(uni, noisy, wrong_roles, save_path):

    fig, ax = plt.subplots(3, figsize=(20,15))

    ax[0].set_title("Universal Population")
    ax[1].set_title("Idiolect Population with Correct Roles")
    ax[2].set_title("Idiolect Population with Incorrect Roles")

    n1, bins1, patches1 = ax[0].hist(uni, bins=10)
    n3, bins3, patches3 = ax[1].hist(noisy, bins=10)
    n5, bins5, patches5 = ax[2].hist(wrong_roles, bins=10)

    plt.savefig(save_path+"AgentIntegration_distribution.png")
    plt.clf()

def perform_comp(paths, task, vis, seeds, combs, agent, save_path):

    # Get Checkpoints
    checkpoint_ones = []
    checkpoint_twos = []
    
    for comb in range(combs):
        checkpoint_one, checkpoint_two = select_random_checkpoints(min_one=4_200_000, min_two=4_200_000)
        checkpoint_ones.append(checkpoint_one)
        checkpoint_twos.append(checkpoint_two)

    universal_rewards = []
    idiolect_rewards = []
    idiolect_wrong_roles_reward = []

    # One Shot Agent Integration
    for seed in range(seeds):
        print("SEED: ", seed)
        uni_seed_reward = []
        idio_seed_reward = []
        idio_wrong_roles_reward = []
        for comb in range(combs):
            print("Combination:", comb)

            checkpoint_one = checkpoint_ones[comb]
            checkpoint_two = checkpoint_twos[comb]

            PATH_ONE_SHARED = paths[0]+"/checkpoint_"+str(checkpoint_one)+".pt"
            PATH_TWO_SHARED = paths[1]+"/checkpoint_"+str(checkpoint_two)+".pt"

            PATH_ONE_UNSHARED = paths[2]+"/checkpoint_"+str(checkpoint_one)+".pt"
            PATH_TWO_UNSHARED = paths[3]+"/checkpoint_"+str(checkpoint_two)+".pt"

            uni_reward = agent_integration(task, PATH_ONE_SHARED, PATH_TWO_SHARED, seed=seed, shared=True, agent=agent, vis = vis, save_path=save_path)
            idio_reward = agent_integration(task, PATH_ONE_UNSHARED, PATH_TWO_UNSHARED, seed=seed, shared=False, agent=agent, vis = vis, save_path=save_path)
            idio_roleless_reward = agent_integration(task, PATH_ONE_UNSHARED, PATH_TWO_UNSHARED, seed=seed, shared=True, agent=agent, vis = vis, save_path=save_path)

            uni_seed_reward.append(torch.mean(uni_reward).item())
            idio_seed_reward.append(torch.mean(idio_reward).item())
            idio_wrong_roles_reward.append(torch.mean(idio_roleless_reward))

        if not vis:
            universal_rewards.append(np.mean(uni_seed_reward))
            idiolect_rewards.append(np.mean(idio_seed_reward))
            idiolect_wrong_roles_reward.append(np.mean(idio_wrong_roles_reward))
    
    if not vis:
        graph_rewards(universal_rewards=universal_rewards, idiolect_rewards=idiolect_rewards, idiolect_wrong_roles_rewards=idiolect_wrong_roles_reward, save_path = save_path)
        plot_distribution(uni=universal_rewards, noisy=idiolect_rewards, wrong_roles=idiolect_wrong_roles_reward, save_path=save_path)

def get_comms(task, PATH_ONE, PATH_TWO, shared, seed, agent):
    assert agent < 2

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml('fine_tuned/vmas/conf/vmas_parameters.yaml')

    # Some basic other configs
    algorithm_config = MaddpgConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Override Necessary Configs
    algorithm_config.share_param_critic = False
    experiment_config.share_policy_params = False

    experiment_config.off_policy_n_envs_per_worker = 1
    experiment_config.off_policy_collected_frames_per_batch = 100
    experiment_config.max_n_frames = 100
    experiment_config.evaluation = False
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.checkpoint_interval = 100_000_000

    experiment = Experiment(
        algorithm_config = algorithm_config,
        task = task,
        seed = seed,
        config = experiment_config,
        model_config = model_config,
        critic_model_config = critic_model_config
    )

    PATH_ONE = PATH_ONE+"/checkpoint_5400000.pt"
    PATH_TWO = PATH_TWO+"/checkpoint_5400000.pt"

    experiment_policy = experiment.state_dict()

    policy_one = torch.load(PATH_ONE)
    policy_two = torch.load(PATH_TWO)
                            
    combined_policy = sub_policy_unshared(policy_one, policy_two) if not shared else sub_policy_shared(experiment_policy, policy_one, policy_two)

    combined_policy['buffer_agents']['_storage']['_storage'] = policy_one['buffer_agents']['_storage']['_storage']
    combined_policy['collector']['policy_state_dict']['module.0.sigma'] = policy_one['collector']['policy_state_dict']['module.0.sigma']

    experiment = experiment.load_experiment_policy(combined_policy)

    experiment.run(eval=True)
    experiment.close()

    return combined_policy

def plot_heatmap(paths, task, seeds, shared, checkpoints, save_path, scenario):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    heat = np.zeros((len(checkpoints)*3, len(checkpoints)*3))
    
    paths = [path+"checkpoint_"+str(checkpoint)+".pt" for path in paths for checkpoint in checkpoints]
    labels = [str(checkpoint) for checkpoint in checkpoints]+[str(checkpoint) for checkpoint in checkpoints]+[str(checkpoint) for checkpoint in checkpoints]

    for idx_x, path_x in enumerate(paths):
        for idx_y, path_y in enumerate(paths):
            rewards = []
            for seed in range(seeds):
                reward = agent_integration(task=task, PATH_ONE=path_x, PATH_TWO=path_y, seed=seed, agent=0, shared=shared)
                rewards.append(reward)
            heat[idx_x][idx_y] = np.mean(rewards)

    # Plot Heatmap
    ax = sns.heatmap(heat, linewidth=0.5, xticklabels=labels, yticklabels=labels)
    plt.tick_params(axis='both', which='major', length=2, bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xlabel("Agent 1")
    plt.ylabel("Agent 0")
    plt.tight_layout()
    plt.savefig(save_path+scenario+"_heatmap.png", dpi=500)
    plt.close()

    # Save vals
    write_csv(save_path+scenario, heatmap_array=heat)

def write_csv(save_path, heatmap_array):
    # Open the file with the append state
    with open(save_path+'_vals.csv', mode='w+') as file:
        writer = csv.writer(file)
        # Write the state information to the CSV file
        for row in heatmap_array:
            writer.writerow(row)

if __name__ == "__main__":
    # Parse Command line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--vis")
    parser.add_argument("--log_comms")
    parser.add_argument("--heatmap")
    
    args=parser.parse_args()
    vis = args.vis
    log_comms = args.log_comms
    heatmap = args.heatmap

    # Checkpoint Paths
    universal_path_shared_one =  "outputs/Final Models/2024-02-01/01-33-36/maddpg_simple_reference_mlp__0370dbb7_24_02_01-01_33_36/checkpoints/"
    universal_path_shared_two = "outputs/2024-03-10/16-42-00/maddpg_simple_reference_mlp__ba90da34_24_03_10-16_42_00/checkpoints/"
    universal_path_shared_three = "outputs/2024-03-11/10-59-22/maddpg_simple_reference_mlp__4fb2c2eb_24_03_11-10_59_22/checkpoints/"
    
    universal_path_unshared_one = "outputs/Final Models/10-31-32/maddpg_simple_reference_mlp__d5090c0f_24_01_15-10_31_32/checkpoints/"
    universal_path_unshared_two = "outputs/Final Models/01-00-18/maddpg_simple_reference_mlp__d06778fe_24_02_20-01_00_18/checkpoints/"
    universal_path_unshared_three = "outputs/23-00-33/maddpg_simple_reference_mlp__c900fd76_24_03_11-23_00_35/checkpoints/"

    noise_path_shared_one = "outputs/Final Models/00-54-38/maddpg_simple_reference_idiolect_mlp__913becc1_24_02_20-00_54_38/checkpoints/"
    noise_path_shared_two = "outputs/2024-03-11/00-18-48/maddpg_simple_reference_idiolect_mlp__118b494c_24_03_11-00_18_48/checkpoints/"
    noise_path_shared_three = "outputs/2024-03-11/18-48-55/maddpg_simple_reference_idiolect_mlp__26cf9021_24_03_11-18_48_55/checkpoints/"

    noise_path_unshared_one = "outputs/Final Models/19-38-40/maddpg_simple_reference_idiolect_mlp__18300887_24_01_14-19_38_40/checkpoints/"
    noise_path_unshared_two = "outputs/Final Models/2024-02-01/19-07-50/maddpg_simple_reference_idiolect_mlp__2aa7883e_24_02_01-19_07_50/checkpoints/"
    noise_path_unshared_three = "outputs/2024-03-12/01-09-30/maddpg_simple_reference_idiolect_mlp__9c92345a_24_03_12-01_09_30/checkpoints/"

    # Changeable params
    seeds = 5
    combs = 5
    agent = 0
    save_path = "/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/graphs/3-15-update/"

    if heatmap:
        shared_noiseless_paths = [universal_path_shared_one, universal_path_shared_two, universal_path_shared_three]
        unshared_noiseless_paths = [universal_path_unshared_one, universal_path_unshared_two, universal_path_unshared_three]
        shared_noise_paths = [noise_path_shared_one, noise_path_shared_two, noise_path_shared_three]
        unshared_noise_paths = [noise_path_unshared_one, noise_path_unshared_two, noise_path_unshared_three]

        checkpoints = range(600000, 5700000, 600000)
        
        # Heatmap in trained env
        plot_heatmap(shared_noiseless_paths, task=VmasTask.SIMPLE_REFERENCE.get_from_yaml(), shared=True, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/", scenario="shared_noiseless")
        plot_heatmap(unshared_noiseless_paths, task=VmasTask.SIMPLE_REFERENCE.get_from_yaml(), shared=False, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/", scenario="unshared_noiseless")
        plot_heatmap(shared_noise_paths, task=VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml(), shared=True, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/", scenario="shared_noise")
        plot_heatmap(unshared_noise_paths, task=VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml(), shared=False, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/", scenario="unshared_noise")
        # Other enviornment than trained on
        plot_heatmap(shared_noiseless_paths, task=VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml(), shared=True, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/other_env/", scenario="shared_noiseless")
        plot_heatmap(shared_noise_paths, task=VmasTask.SIMPLE_REFERENCE.get_from_yaml(), shared=True, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/other_env/", scenario="shared_noise")
        plot_heatmap(unshared_noiseless_paths, task=VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml(), shared=False, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/other_env/", scenario="unshared_noiseless")
        plot_heatmap(unshared_noise_paths, task=VmasTask.SIMPLE_REFERENCE.get_from_yaml(), shared=False, seeds=seeds, checkpoints=checkpoints, save_path=save_path+"heatmaps/other_env/", scenario="unshared_noise")

    elif vis:
        # Paths
        noiseless = [universal_path_shared_one, universal_path_shared_two, universal_path_unshared_one, universal_path_unshared_two]
        noisy = [noise_path_shared_one, noise_path_shared_two, noise_path_unshared_one, noise_path_unshared_two]

        # Perform Comparisons
        perform_comp(paths=noiseless, task=VmasTask.SIMPLE_REFERENCE.get_from_yaml(), vis=vis, seeds=seeds, combs=combs, agent=agent, save_path=save_path+"NoNoise/")
        perform_comp(paths=noisy, task=VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml(), vis=vis, seeds=seeds, combs=combs, agent=agent, save_path=save_path+"WithNoise/")
    
    elif log_comms:
        pol1 = get_comms(VmasTask.SIMPLE_REFERENCE.get_from_yaml(), universal_path_shared_one, universal_path_shared_two, shared=True, seed = 2, agent = 0)
        pol2 = get_comms(VmasTask.SIMPLE_REFERENCE_IDIOLECT.get_from_yaml(), noise_path_shared_one, noise_path_shared_two, shared=True, seed = 2, agent = 0)
    # for seed in range(seeds):
    #     pol1 = get_comms(VmasTask.SIMPLE_REFERENCE.get_from_yaml(), universal_path_shared_one, universal_path_shared_three, shared=True, seed = seed, agent = 0)