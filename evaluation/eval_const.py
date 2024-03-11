import eval
from benchmarl.environments import IdiolectEvoTask

def generate_data(paths, seed):
    # Get Paths
    universal = paths[0]
    idiolect = paths[1]

    # Get Stats for old environment
    universal_old, universal_old_means, universal_old_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_OLD_CONST.get_from_yaml(), universal, seed)
    idiolect_old, idiolect_old_means, idiolect_old_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_OLD_NOISE_CONST.get_from_yaml(), idiolect, seed)
    # old_evals = [universal_old, idiolect_old]
    # old_pair = [(universal_old, idiolect_old)]
    # old_graphs = [universal_old_graphs, idiolect_old_graphs]

    # titles = ["universal", "idiolect"]

    # # Initialize the dictionaries for all p-values
    # p_vals_old = {
    #     "Titles": titles,
    #     "less": [],
    #     "greater": []
    # }

    # # Populate P-Value Dictionaries
    # eval.eval_pairs(old_pair, p_vals_old)

    # # Initialize the dictionaries for all means
    # means_old = {
    #     "universal": universal_old_means,
    #     "idiolect": idiolect_old_means,
    # }
    
    # # Write results to files
    # output_folder = '/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/stats/const'
    # eval.write_csv(os.path.join(output_folder, 'p_vals_old_seed'+str(seed)+'.csv'), p_vals_old)
    # eval.write_csv(os.path.join(output_folder, 'means_old'+str(seed)+'.csv'), means_old, False)

    # # Graph Data
    # eval.graph_data(old_graphs, titles, "Old_seed"+str(seed))

if __name__ == "__main__":
    # Checkpoint paths
    universal = "evaluation/checkpoints/constant_training/universal.pt"
    noise = "evaluation/checkpoints/constant_training/idiolect.pt"

    paths = [universal, noise]

    # Seed
    seeds = 10

    # Generate Everything
    for seed in range(seeds):
        print("SEED ", seed)
        seed = seed + 10
        generate_data(paths, seed)