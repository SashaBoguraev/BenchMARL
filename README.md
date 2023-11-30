# Emergent Idiolects 
### Forked from [FacebookResearch/BenchMARL](https://github.com/facebookresearch/BenchMARL)

#### This repository is a working collection of Sasha Boguraev's Undergraduate Honors Thesis. It draws extensively on both the [Vectorized Multi Agent Simulators](https://github.com/proroklab/VectorizedMultiAgentSimulator) and [BenchMARL](https://github.com/facebookresearch/BenchMARL). As an active fork of BenchMARL, this repo also provides full BenchMARL functionality. I take no credit for any of BenchMARL's or VMAS' implementation or functionality, and attribute all credit for those implementations to the respective authors.


## Premise for Work

This work is premised on the observation that in a population, each individual has a unique understanding of language: their personal idiolect. Without going into too much detail, this is a counterintuitive phenomena as it would seem that universal languages, ones where every speaker understands language in the same way, would be more efficient. I aim to investigate which advantages, if any, are conferred by this phenomena.

To do this, I aim to use MARL to develop populations of agents with varying styles of emergent communication, and evaluate these populations using various tests to investigate what advantages they hold.

In this repository, I provide scripts to develop various populaitons. Of these, one is a universal population, and three possess various idiolects. These idiolects comprise of:
1. Adding noise to the communication channel 
2. Using soft-attention to allow agents memory capacity
3. Combining both above methods

The provided scripts train agents within these populations using [Multi-Agent Deep Deterministic Policy Gradients](https://arxiv.org/abs/1706.02275) and BenchMARL's fine-tuned VMAS hyper-parameters. Through BenchMARL, training logging is provided via [Weights and Biases](https://wandb.ai/site). 

This repository will further include the code necessary to evaluate models, once they have been developed.

## Different Populations:

### Universal Population

`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference`

The universal population merely consists of agents trained on the VMAS simple reference task, which itself has been adapted from OpenAI's [MPEs](https://github.com/openai/multiagent-particle-envs). In this scenario, there are two agents, and three landmarks. Agents' goals consist of desiring the other agent to move to a specific location. However, goals are not observable agents. In its stead, agents can send "communications" consisting of an n-dimensional vector (base setting of 10), in an attempt to complete their goals. Our reward function is the L2 distance between their goal agent (the agent they want to move to a specific target) and goal landmark (the landmark they want the agent to move to). Agents are cooperatively rewarded by the sum of their individual rewards.

### Idiolect Population (Noise Regimen)

`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_idiolect`

In this case, we simulate idiolects by peturbing the communcation sent by agents. At the start of the simulation, agents are each assigned two parameters, &alpha; and &beta; at random. At training time, a noise vector of the same size as the communication vector, is sampled from this distribution for each agent, and added in element-wise manner to the communication.

### Idiolect Population (Memory Regimen)

`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_mem_buffer`

In this scenario, agents are provided a memory buffer of five times the size of an episode. This memory buffer gets progressively filled by the most recent observations, such that it always consists of the most recent `5*E` observations (where `E = Episode Length`). Using soft-attention, an agents current observation attends to all observations in the memory buffer, and a weighted memory vector is created, weighted by dot-product similarity to the current observation. This weighted memory vector is then appended to the observation as input to the policy network. This attention mechanism is implemented in `World.weight_mem` of `VectorizedMultiAgentSimulator/vmas/simulator/core.py`.

### Idiolect Population (Noise + Memory Regimen)

`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_noise_mem`

In this population both of the prior regimens are combined.

### Constant environments

`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_const`
`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_idiolect_const`
`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_mem_buffer_const`
`IdiolectEvo/VectorizedMultiAgentSimulator/vmas/scenarios/mpe/simple_reference_noise_mem_const`

By default, environments have landmarks which randomly change location every episode. To explore additional behavior, I provide "constant environments", which keep landmarks constant throughout training time.


## Setup

To use this repo, please clone locally to access the configs and scripts:
```bash
git clone https://github.com/SashaBoguraev/IdiolectEvo.git
pip install -e IdiolectEvo
```
Further, install the dependencies for VMAS:
```bash
cd IdiolectEvo/VectorizedMultiAgentSimulator
pip install -e .
```
These packages only include necessary packages. To include full functionality of logging and run visualization, one may want to install the following packages:
```bash
pip install wandb opencv-python moviepy
```

## Usage

To train the respective models we have provided the `train_model.sh` script. This script utilizes the following keywords to run the desired training:
1. `SCENARIO`: One of "universal", "noise", "memory" or "both", corresponding to the regimen you would like to train your population with. This is a REQUIRED keyword.
2. `ENVIRONMENT`: One of "constant" or "variable", depending on whether you want your enviornment to be constant or changing during training. This is a REQUIRED keyword.
3. `ITERS`: Number representing the number of populations you would like to train seperately. This is a REQUIRED keyword. 

*You MUST be within the IdiolectEvo directory to run these commands*

*If any variable is not specified, your model will train with whatever regimen is stored in the environment variables.*

### Example Training:

For example, to train 3 populations of a noise idiolect with variable landmark positions, you would use the following:

```bash
./train_model.sh SCENARIO="noise" ENVIRONMENT="variable" ITERS=3 
``````
