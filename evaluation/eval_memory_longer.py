from eval import *

def generate_data(paths, seed):
    # Get Paths
    memory5 = paths[0]
    memory10 = paths[1]
    memory15 = paths[2]

    # Get Stats for old environment
    memory5_old, memory5_old_means, memory5_old_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_OLD_MEM_BUFFER.get_from_yaml(), memory5, seed)
    memory10_old, memory10_old_means, memory10_old_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_OLD_MEM_BUFFER.get_from_yaml(), memory10, seed)
    memory15_old, memory15_old_means, memory15_old_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_OLD_MEM_BUFFER.get_from_yaml(), memory15, seed)
    old_evals = [memory5_old, memory10_old, memory15_old]
    old_pairs = list(itertools.combinations(old_evals, 2))
    old_graphs = [memory5_old_graphs, memory10_old_graphs, memory15_old_graphs]

    # Get Stats for new environment
    memory5_new, memory5_new_means, memory5_new_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_NEW_MEM_BUFFER.get_from_yaml(), memory5, seed)
    memory10_new, memory10_new_means, memory10_new_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_NEW_MEM_BUFFER.get_from_yaml(), memory10, seed)
    memory15_new, memory15_new_means, memory15_new_graphs = eval.run_benchmark(IdiolectEvoTask.SPEED_NEW_MEM_BUFFER.get_from_yaml(), memory15, seed)
    new_evals = [memory5_new, idiolect_new]
    new_pairs = list(itertools.combinations(new_evals, 2))
    new_graphs = [memory5_new_graphs, memory10_new_graphs, memory15_new_graphs]

    titles = ["memory5", "memory10", "memory15"]

    # Initialize the dictionaries for all p-values
    p_vals_old = {
        "Titles": titles,
        "less": [],
        "greater": []
    }
    p_vals_new = {
        "Titles": titles,
        "less": [],
        "greater": []
    }

    # Populate P-Value Dictionaries
    eval.eval_pairs(old_pair, p_vals_old)
    eval.eval_pairs(new_pair, p_vals_new)

    # Initialize the dictionaries for all means
    means_old = {
        "memory5": memory5_old_means,
        "memory10": memory10_old_means,
        "memory15": memory15_old_means,
    }
    means_new = {
        "memory5": memory5_new_means,
        "memory10": memory10_new_means,
        "memory15": memory15_new_means,
    }
    
    # Write results to files
    output_folder = '/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/evaluation/stats/memory_length'
    eval.write_csv(os.path.join(output_folder, 'p_vals_old_seed'+str(seed)+'.csv'), p_vals_old)
    eval.write_csv(os.path.join(output_folder, 'p_vals_new'+str(seed)+'.csv'), p_vals_new)
    eval.write_csv(os.path.join(output_folder, 'means_old'+str(seed)+'.csv'), means_old, False)
    eval.write_csv(os.path.join(output_folder, 'means_new'+str(seed)+'.csv'), means_new, False)

    # Graph Data
    eval.graph_data(old_graphs, titles, "Old_seed"+str(seed))
    eval.graph_data(new_graphs, titles, "Novel_seed"+str(seed))

if __name__ == "__main__":
    # Checkpoint paths
    memory5 = "evaluation/checkpoints/memory_length/mem5.pt"
    memory10 = "evaluation/checkpoints/memory_length/mem10.pt"
    memory15 = "evaluation/checkpoints/memory_length/mem15.pt"

    paths = [memory5, memory10, memory15]

    # Seed
    seeds = 10

    # Generate Everything
    for seed in range(seeds):
        print("SEED ", seed)
        seed = seed + 10
        generate_data(sim_paths, seed)