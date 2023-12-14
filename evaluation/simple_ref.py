from eval import use_vmas_env

scenario_name="simple_reference"
path = "/Users/sashaboguraev/Desktop/Cornell/College Scholar/BenchMARL/outputs/2023-11-20/FIrst Universal/maddpg_simple_reference_mlp__3141fed9_23_11_20-14_40_19/checkpoints/checkpoint_9900000.pt"
use_vmas_env(
    scenario=scenario_name,
    render=True,
    save_render=True,
    num_envs=1,
    n_steps=100,
    device="cpu",
    continuous_actions=True,
    PATH=path
)