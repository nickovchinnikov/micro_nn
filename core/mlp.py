from typing import List
import random

from .value import Value


class Module:
    r"""
    A class representing a module in a neural network.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> List["Value"]:
        r"""
        Get the parameters of the module.

        Returns:
            List[Value]: The parameters of the module.
        """
        raise NotImplementedError


class Neuron(Module):
    r"""
    A class representing a single neuron in a neural network.
    """

    def __init__(self):
        # Weights by default are None will be initialized on the first call
        self.w = None
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: List[float]) -> "Value":
        r"""
        Compute the output of the neuron given the input x.

        Args:
            x (List[float]): The input to the neuron.

        Returns:
            Value: The output of the neuron.
        """
        self.w = (
            [Value(random.uniform(-1, 1)) for _ in range(len(x))]
            if self.w is None
            else self.w
        )

        out = sum((w * x for w, x in zip(self.w, x)), self.b).tanh()
        return out

    def parameters(self) -> List["Value"]:
        r"""
        Get the parameters of the neuron.

        Returns:
            List[Value]: The parameters of the neuron.
        """
        return self.w + [self.b]


class Layer(Module):
    r"""
    A class representing a layer of neurons in a neural network.

    Args:
        n_neurons (int): The number of neurons in the layer.
    """

    def __init__(self, n_neurons: int):
        self.neurons = [Neuron() for _ in range(n_neurons)]

    def __call__(self, x: List[float]) -> List["Value"]:
        r"""
        Compute the output of the layer given the input x.

        Args:
            x (List[float]): The input to the layer.

        Returns:
            List[Value]: The output of the layer.
        """
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List["Value"]:
        r"""
        Get the parameters of the layer.

        Returns:
            List[Value]: The parameters of the layer.
        """
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP(Module):
    r"""
    A class representing a multi-layer perceptron.

    Args:
        n_inputs (int): The number of inputs to the network.
        n_outputs (List[int]): The number of outputs from each layer.
    """

    def __init__(self, n_inputs: int, n_outputs: List[int]):
        self.inputs = [Layer(n_inputs)]
        self.outputs = [Layer(n) for n in n_outputs]

        self.layers = self.inputs + self.outputs

    def __call__(self, x: List[float]) -> List["Value"]:
        r"""
        Compute the output of the MLP given the input x.

        Args:
            x (List[float]): The input to the network.

        Returns:
            List[Value]: The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List["Value"]:
        r"""
        Get the parameters of the MLP.

        Returns:
            List[Value]: The parameters of the MLP.
        """
        return [param for layer in self.layers for param in layer.parameters()]

    def step(self, lr: float):
        r"""
        Perform a step of gradient descent on the parameters of the MLP.

        Args:
            lr (float): The learning rate.
        """
        for param in self.parameters():
            param.data += -lr * param.grad
