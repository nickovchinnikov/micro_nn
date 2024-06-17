import math
from typing import List, Tuple, Union


class Value:
    r"""
    A class representing a scalar value and its gradient.
    """

    def __init__(
        self,
        data: Union[int, float, "Value"],
        _prev: Tuple["Value", "Value"] = (),
        _op: str = "",
        label: str = "",
    ):
        r"""
        Initialize a Value object.

        Args:
            data (Union[int, float, Value]): The scalar value or another Value object.
            _prev (Tuple[Value, Value], optional): The previous Value objects that this Value depends on. Defaults to ().
            _op (str, optional): The operation that produced this Value. Defaults to "".
            label (str, optional): A label for this Value. Defaults to "".
        """
        self.grad = 0.0
        self._backward = lambda: None

        if isinstance(data, Value):
            self.__dict__ = data.__dict__
        else:
            self.data = data
            self._prev = set(_prev)
            self._op = _op

        self.label = label

    def __repr__(self) -> str:
        r"""
        Return a string representation of the Value.

        Returns:
            str: A string representation of the Value.
        """
        return f"Value(data={self.data})"

    @staticmethod
    def _scalar_gate(other: Union[int, float]) -> Union[int, float]:
        r"""
        Create a scalar from an integer or float.

        Args:
            other (Union[int, float]): The scalar to convert.

        Returns:
            Union[int, float]: The scalar.
        """
        assert isinstance(other, (int, float)), "Support only int or float."
        return other

    @staticmethod
    def _value_gate(other: Union[int, float, "Value"]) -> "Value":
        r"""
        Create a Value object from an integer or float.

        Args:
            other (Union[int, float, Value]): The value to convert.

        Returns:
            Value: A new Value object.
        """
        if not isinstance(other, Value):
            other = Value._scalar_gate(other)
            other = Value(other)
        return other

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        r"""
        Add another Value to this Value and return a new Value representing the sum.

        Args:
            other (Union[int, float, "Value"]): The value to add.

        Returns:
            Value: A new Value representing the sum.
        """
        other = self._value_gate(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Union[int, float]) -> "Value":
        r"""
        Handle addition with self on the right-hand side.

        Args:
            other (Value): The Value to add.

        Returns:
            Value: A new Value representing the sum.
        """
        other = Value._scalar_gate(other)

        return self + other

    def __neg__(self) -> "Value":
        r"""
        Negate the Value.

        Returns:
            Value: A new Value representing the negation.
        """
        return self * -1

    def __sub__(self, other: "Value") -> "Value":
        r"""
        Subtract another Value from this Value and return a new Value representing the difference.

        Args:
            other (Value): The Value to subtract.

        Returns:
            Value: A new Value representing the difference.
        """
        return self + (-other)

    def __rsub__(self, other):
        r"""
        Handle subtraction with self on the right-hand side.

        Args:
            other (Value): The Value to subtract.

        Returns:
            Value: A new Value representing the difference.
        """
        return -self + other

    def __mul__(self, other: "Value") -> "Value":
        r"""
        Multiply this Value by another Value and return a new Value representing the product.

        Args:
            other (Value): The Value to multiply.

        Returns:
            Value: A new Value representing the product.
        """
        other = self._value_gate(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        r"""
        Handle multiplication with self on the right-hand side.

        Args:
            other (Value): The Value to multiply.

        Returns:
            Value: A new Value representing the product.
        """
        return self * other

    def __truediv__(self, other: "Value") -> "Value":
        r"""
        Divide this Value by another Value and return a new Value representing the quotient.

        Args:
            other (Value): The Value to divide by.

        Returns:
            Value: A new Value representing the quotient.
        """
        other = self._value_gate(other)

        return self * other**-1

    def __pow__(self, other: Union[int, float]) -> "Value":
        r"""
        Raise this Value to the power of another Value and return a new Value representing the result.

        Args:
            other (Union[int, float]): The exponent.

        Returns:
            Value: A new Value representing the result.
        """
        other = Value._scalar_gate(other)

        x, y = self.data, other
        out = Value(x**y, (self,), f"**{y}")

        def _backward():
            # d(x**y)/dx = y * x**(y-1)
            self.grad += y * (x ** (y - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        r"""
        Compute the exponential of this Value and return a new Value representing the result.

        Returns:
            Value: A new Value representing the result.
        """
        x = self.data

        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        r"""
        Compute the hyperbolic tangent of this Value and return a new Value representing the result.

        Returns:
            Value: A new Value representing the result.
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def _topo_sort(self) -> List["Value"]:
        r"""
        Perform a topological sort on the graph of Values.

        Returns:
            List[Value]: A list of Values in topological order.
        """
        visited = set()
        topo: List["Value"] = []

        def _sort(graph):
            if graph not in visited:
                visited.add(graph)
                for node in graph._prev:
                    _sort(node)
                topo.append(graph)

        _sort(self)
        return topo

    def backward(self):
        r"""
        Perform backpropagation to compute the gradients of all Values in the graph.

        This method performs a topological sort on the graph of Values and then computes the gradient of each Value in reverse topological order. The gradient of this Value is set to 1.0 before backpropagation begins.
        """
        self.grad = 1.0
        for node in reversed(self._topo_sort()):
            node._backward()

    def zero_grad(self):
        r"""
        Set the gradients of all Values in the graph to zero.

        This method sets the gradient of each Value to zero. This is typically used at the start of each training iteration to ensure that gradients from previous iterations do not affect the current iteration.
        """
        visited = set()

        def _zero_grad(node: "Value"):
            if node not in visited:
                visited.add(node)
                node.grad = 0.0
                for n in node._prev:
                    _zero_grad(n)

        _zero_grad(self)
