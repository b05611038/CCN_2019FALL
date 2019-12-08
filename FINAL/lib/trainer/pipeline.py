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
        self.time = kwargs.get('time', int(self.agent.model.dt))
        self.skip_first_frame = kwargs.get('skip_first_frame', True)

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)
        
        # Set up for multiple layers of input layers.
        self.inputs = [
            name
            for name, layer in self.network.layers.items()
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
                self.output: torch.zeros((self.time, len(self.agent.action)))
            }

    def train(self, num_episodes):
        for ep in range(num_episodes):
            self.episode(ep)
        return None

    def episode(self, iter_num, train = True, test_seed = None, **kwargs):
        if train:
            print('Start episode: %d ...' % iter_num)

        self.reset_state_variables()
        if not train:
            if test_seed is None:
                test_seed = 0
            self.env.seed(test_seed)

        for frame_iter in itertools.count():
            obs, reward, done, info = self.env_step()

            self.step((obs, reward, done, info), **kwargs)

            if frame_iter % 100 == 0 and frame_iter != 0:
                print('Game frame: %d.' % frame_iter)

            if done:
                break

        if train:
            print(
                f"Episode: {iter_num} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )

        return self.accumulated_reward

    def env_step(self) -> Tuple[torch.Tensor, float, bool, Dict]:
        # Render game.
        if (
            self.render_interval is not None
            and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action based on output neuron spiking.
        self.action = self.action_function(self, output = self.output)

        # Run a step of the environment.
        obs, reward, done, info = self.env.step(self.action, tensor = True)

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
        preprocessed = self.agent.preprocess(obs)
        shape = [1] * len(preprocessed.shape)
        inputs = {k: preprocessed.repeat(self.time, *shape).unsqueeze(0) for k in self.inputs}

        inputs['Input Layer'] = torch.randn(100, 1, 84, 84)

        # Run the network on the spike train-encoded inputs.
        self.network.run(inputs = inputs, time = self.time, reward = reward, **kwargs)
        self.agent.model = self.network
        self.agent.insert_memory(obs)

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

    def plots(self, batch, step_out) -> None:
        pass


