#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import World, Agent, Landmark
from vmas.simulator.scenario import BaseScenario



class FFNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, initialization = torch.nn.init.uniform_) -> None:
        """
        A multi-layer feed-forward neural network that applies a linear transformation, followed by a ReLU
        nonlinearity, at each layer.
        Parameters
        ----------
        embedding_dim : int
            Number of dimensions of an input embedding.
        hidden_dim : int
            Number of dimensions for the hidden layer(s).
        output_dim : int
            Number of dimensions for the output layer.
        num_layers : int
            Number of hidden layers to initialize.
        initialization : torch.nn.init funciton
            Distribution to initialize weights from
        """
        super().__init__()

        assert num_layers > 0

        self.input_hidden = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        initialization(self.input_hidden.weight)
        
        self.hidden_hidden = None
        
        if num_layers > 1:
            self.hidden_hidden = torch.nn.ModuleList(
                [torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for _ in range(num_layers - 1)]
            )
            for idx, layer in enumerate(self.hidden_hidden):
                initialization(layer.weight)
                
        self.hidden_output = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
        initialization(self.hidden_output.weight)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through each of the network layers using the given input observation.
        Parameters
        ----------
        obs : torch.Tensor
            Input tensor of observations of shape ``(batch_size, observation_dimension, 1)``.
        Returns
        -------
        torch.Tensor
            Output tensor resulting from forward pass of shape ``(batch_size, c_dim, 1)``.
        """
        
        out = torch.nn.functional.relu(self.input_hidden(obs))
        if self.hidden_hidden is not None:
            for hidden_hidden in self.hidden_hidden:
                
                out = torch.nn.functional.relu(hidden_hidden(out))
                
        out = self.hidden_output(out)/1000000
        return out

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim=batch_dim, device=device, dim_c=10)

        n_agents = 2
        n_landmarks = 3

        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent_{i}", collide=False, silent=False)
            world.add_agent(agent)
        # Add landmarks
        for i in range(n_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            # assign goals to agents
            for agent in self.world.agents:
                agent.goal_a = None
                agent.goal_b = None
            # want other agent to go to the goal landmark
            self.world.agents[0].goal_a = self.world.agents[1]
            self.world.agents[0].goal_b = self.world.landmarks[
                torch.randint(0, len(self.world.landmarks), (1,)).item()
            ]
            self.world.agents[1].goal_a = self.world.agents[0]
            self.world.agents[1].goal_b = self.world.landmarks[
                torch.randint(0, len(self.world.landmarks), (1,)).item()
            ]
            # random properties for agents
            for i, agent in enumerate(self.world.agents):
                agent.color = torch.tensor(
                    [0.25, 0.25, 0.25], device=self.world.device, dtype=torch.float32
                )
            # random properties for landmarks
            self.world.landmarks[0].color = torch.tensor(
                [0.75, 0.25, 0.25], device=self.world.device, dtype=torch.float32
            )
            self.world.landmarks[1].color = torch.tensor(
                [0.25, 0.75, 0.25], device=self.world.device, dtype=torch.float32
            )
            self.world.landmarks[2].color = torch.tensor(
                [0.25, 0.25, 0.75], device=self.world.device, dtype=torch.float32
            )
            # special colors for goals
            self.world.agents[0].goal_a.color = self.world.agents[0].goal_b.color
            self.world.agents[1].goal_a.color = self.world.agents[1].goal_b.color

            # Make everything for noise (need to make this not hard-coded)
            for agent in self.world.agents:
                # agent.noise = FFNN(21, 64, self.world.dim_c, 3)
                agent.memory = torch.zeros((self.world.batch_dim, 21, 500))
                # agent.noise = torch.distributions.Beta(torch.rand(1), torch.rand(1))

        # set random initial states
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device)
            for a in self.world.agents:
                if a.goal_a is None or a.goal_b is None:
                    return torch.zeros(
                        self.world.batch_dim,
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                self.rew += -torch.sqrt(
                    torch.sum(
                        torch.square(a.goal_a.state.pos - a.goal_b.state.pos), dim=-1
                    )
                )
        return self.rew

    def observation_no_mem(self, agent: Agent):
        # goal color
        goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)

        # communication of all other agents
        comm = []
        for other in self.world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
        return torch.cat(
            [
                agent.state.vel,
                *entity_pos,
                goal_color.repeat(self.world.batch_dim, 1),
                *comm,
            ],
            dim=-1,
        )
    
    def observation(self, agent: Agent):
            obs = self.observation_no_mem(agent)
            if agent.memory != None:
                out = torch.cat(
                [
                    obs,
                    self.world.weight_mem(obs, agent)
                ],
                dim=-1
                )
                agent.memory = torch.cat(
                    [agent.memory,
                        torch.unsqueeze(obs, 2)],
                        dim = -1
                )[:, :, 1:]
                return out
            else:
                return obs
