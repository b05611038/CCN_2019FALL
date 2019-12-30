import torch
import numpy as np

from ..trainer.pipeline import PongPipeline


__all__ = ['select_softmax']


def select_softmax(pipeline: PongPipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action using softmax function based on spiking from a network layer.
    :param pipeline: EnvironmentPipeline with environment that has an integer action
        space and :code:`spike_record` set.
    :return: Action sampled from softmax over activity of similarly-sized output layer.
    Keyword arguments:
    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs["output"]
    except KeyError:
        raise KeyError('select_softmax() requires an "output" layer argument.')

    assert (
        pipeline.network.layers[output].n == pipeline.env.action_space.n
    ), "Output layer size is not equal to the size of the action space."

    assert hasattr(
        pipeline, "spike_record"
    ), "EnvironmentPipeline is missing the attribute: spike_record."

    spikes = torch.sum(pipeline.spike_record[output], dim=0)
    probabilities = torch.softmax(spikes, dim=0)
    return torch.multinomial(probabilities, num_samples=1).item()

 
