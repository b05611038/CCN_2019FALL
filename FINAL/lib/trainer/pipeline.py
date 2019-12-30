import itertools
import warnings
from typing import Any, Callable, Optional, Tuple, Dict

import torch
from torch.utils.data import DataLoader 

import bindsnet
from bindsnet.pipeline.base_pipeline import BasePipeline, recursive_to
from bindsnet.network import Network
from bindsnet.network.nodes import AbstractInput
from bindsnet.network.monitors import Monitor

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
        self.action_function = action_function

        self.accumulated_reward = 0.0
        self.reward_list = []

        self.output = kwargs.get('output', None)
        self.render_interval = kwargs.get('render_interval', None)
        self.reward_delay = kwargs.get('reward_delay', None)
        self.time = kwargs.get('time', int(self.agent.model.dt))
        self.skip_first_frame = kwargs.get('skip_first_frame', True)
        self.replay_buffer = kwargs.get('replay_buffer', None)
        if self.replay_buffer is None:
            warnings.warn('Please use replay buffer to handle sparse rewarding condition.')

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

    def update_episode_memory(self, **kwargs):
        if self.replay_buffer is not None:
            self.network.learning = True

            self.replay_buffer.make(self.replay_buffer.maximum)
            loader = DataLoader(self.replay_buffer, batch_size = 1, shuffle = False)

            for mini_iter, (observation, reward) in enumerate(loader):
                observation = observation.to(self.device)
                reward = reward.to(self.device)

                shape = [1] * len(observation.size()) 
                inputs = {k: observation.unsqueeze(0).repeat(self.time, *shape).unsqueeze(2) for k in self.inputs}

                self.network.run(inputs = inputs, time = self.time, reward = reward, **kwargs) 

                if mini_iter % 100 == 0 and mini_iter != 0:
                    print('Update iter: %d' % mini_iter)

            self.agent.model = self.network
            self.network.learning = False

        return None

    def episode(self, iter_num, train = True, test_seed = None, **kwargs):
        if train:
            print('Start episode: %d ...' % iter_num)

        self.reset_state_variables()
        if not train:
            if test_seed is None:
                test_seed = 0

            self.env.seed(test_seed)
        else:
            self.replay_buffer.new_episode()

        self.mini_counter = 0
        for frame_iter in itertools.count():
            obs, reward, done, info = self.env_step()

            self.step((obs, reward, done, info), train, **kwargs)

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

    def step(self, batch: Any, train, **kwargs) -> Any:
        # language=rst
        """
        Single step of any pipeline at a high level.
        :param batch: A batch of inputs to be handed to the ``step_()`` function.
                      Standard in subclasses of ``BasePipeline``.
        :return: The output from the subclass's ``step_()`` method, which could be
            anything. Passed to plotting to accommodate this.
        """
        self.step_count += 1

        batch = recursive_to(batch, self.device)
        step_out = self.step_(batch, train, **kwargs)

        if (
            self.print_interval is not None
            and self.step_count % self.print_interval == 0
        ):
            print(
                f"Iteration: {self.step_count} (Time: {time.time() - self.clock:.4f})"
            )
            self.clock = time.time()

        self.plots(batch, step_out)

        if self.save_interval is not None and self.step_count % self.save_interval == 0:
            self.network.save(self.save_dir)

        if self.test_interval is not None and self.step_count % self.test_interval == 0:
            self.test()

        return step_out

    def step_(
        self, gym_batch: Tuple[torch.Tensor, float, bool, Dict], train, **kwargs
    ) -> None:
        obs, reward, done, info = gym_batch

        # Place the observations into the inputs.
        preprocessed = self.agent.preprocess(obs)
        shape = [1] * len(preprocessed.shape)
        inputs = {k: preprocessed.repeat(self.time, *shape).unsqueeze(1) for k in self.inputs}

        if self.replay_buffer is not None and train:
            self.replay_buffer.insert(preprocessed)
            self.mini_counter += 1
            if reward != 0:
                self.replay_buffer.insert_reward(reward, self.mini_counter, done)
                self.mini_counter = 0

        # Run the network on the spike train-encoded inputs.
        self.network.run(inputs = inputs, time = self.time, reward = reward, batch_size = 20, **kwargs)
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


