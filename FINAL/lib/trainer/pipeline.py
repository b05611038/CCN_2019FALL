import itertools
from typing import Callable, Optional, Tuple, Dict

import torch
import bindsnet
from bindsnet.pipeline.base_pipeline import BasePipeline
from bindsnet.network import Network
from bindsnet.network.nodes import AbstractInput
from bindsnet.network.monitors import Monitor
from bindsnet.pipeline.action import select_softmax

import lib
from lib.environment import GymEnvironment
from lib.agent import Agent, PongAgent
from lib.agent.snn import SNN


__all__ = ['PongPipeline']


# ref: https://github.com/BindsNET/bindsnet/blob/7ab6d8314310755c18f2a5f474b5d24a2c6a9295/bindsnet/pipeline/environment_pipeline.py#L14
class PongPipeline(BasePipeline):
    def __init__(self,
            agent: Agent,
            environment: GymEnvironment,
            action_function: Optional[Callable] = None,
            **kwargs):

        super(PongPipeline, self).__init__(agent.model, **kwargs)

        self.agent = agent
        if not isinstance(self.agent.model, SNN):
            raise TypeError('Only SNN agent can be used in PongPipeline.')

        self.network = self.agent.model
        self.device = self.agent.device
        self.network = self.network.to(self.device)
        
        self.env = environment
        if action_function is None:
            self.action_function = select_softmax
        else:
            self.action_function = action_function

        self.accumulated_reward = 0.0
        self.reward_list = []

        self.output = kwargs.get('output', None)
        self.render_interval = kwargs.get('render_interval', None)
        self.reward_delay = kwargs.get('reward_delay', None)
        self.time = kwargs.get('time', int(network.dt))
        self.skip_first_frame = kwargs.get('skip_first_frame', True)

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)
        
        # Set up for multiple layers of input layers.
        self.inputs = [
            name
            for name, layer in network.layers.items()
            if isinstance(layer, AbstractInput)
        ]

        self.action = None

        self.voltage_record = None
        self.threshold_value = None

        self.first = True        

        if self.output is not None:
            self.network.add_monitor(
                Monitor(self.network.layers[self.output], ["s"]), self.output
            )

            self.spike_record = {
                self.output: torch.zeros((self.time, len(self.agent.valid_action)))
            }

    def train(self, num_episodes):
        episode = 0
        while episode < num_episodes:
            self.episode(episode)
            episode += 1

        return None

    def episode(self, iter_num, train = True, test_seed = None):
        self.reset_state_variables()
        if not train:
            if test_seed is None:
                self.env.seed(0)
            else:
                self.env.seed(test_seed)

        for _ in itertools.count():
            obs, reward, done, info = self.env_step()

            self.step((obs, reward, done, info), **kwargs)

            if done:
                break

        print(
            f"Episode: {iter_num} - "
            f"accumulated reward: {self.accumulated_reward:.2f}"
        )

        self.agent.model = self.network

        return self.accumulated_reward

    def env_step(self) -> Tuple[torch.Tensor, float, bool, Dict]:
        '''
        still need to add self.agent.insert_memory into step
        '''
        # Render game.
        if (
            self.render_interval is not None
            and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action based on output neuron spiking.
        preprocessed = self.agent.preprocess(self.output)
        preprocessed = preprocessed.to(self.device)
        self.action = self.action_function(self, output = preprocessed)

        # Run a step of the environment.
        obs, reward, done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward

        info["accumulated_reward"] = self.accumulated_reward

        return obs, reward, done, info

    def step_(
        self, gym_batch: Tuple[torch.Tensor, float, bool, Dict], **kwargs
    ) -> None:
        obs, reward, done, info = gym_batch

        # Place the observations into the inputs.
        obs_shape = [1] * len(obs.shape[1: ])
        inputs = {k: obs.repeat(self.time, *obs_shape) for k in self.inputs}

        # Run the network on the spike train-encoded inputs.
        self.network.run(inputs = inputs, time = self.time, reward = reward, **kwargs)

        if self.output is not None:
            self.spike_record[self.output] = (
                self.network.monitors[self.output].get("s").float()
            )

        if done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward = self.accumulated_reward,
                    steps = self.step_count,
                    **kwargs,
                )
            self.reward_list.append(self.accumulated_reward)

        return None

    def reset_state_variables(self) -> None:
        obs = self.env.reset()
        self.agent.insert_memory(obs)
        self.network.reset_state_variables()
        self.accumulated_reward = 0.0
        self.step_count = 0
        return obs

    def init_fn(self) -> None:
        pass

    def plots(self) -> None:
        pass


