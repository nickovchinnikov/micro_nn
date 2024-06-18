# %%
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, List


from tracer import draw_dot
# %matplotlib inline

random.seed(0)
np.random.seed(0)

# Check styles plt.style.available
plt.style.use("seaborn-v0_8-whitegrid")


# %%
def f(x):
    return 3 * x**2 - 4 * x + 5


# %%
f(3)

# %%
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
plt.show()

# %%
# Bump the x value by a small amount h
h = 0.000001

# Derivative: f'(x) = (f(x + h) - f(x)) / h
df_dx = lambda f, x: (f(x + h) - f(x)) / h

x = 3
df_dx(f, x)

# %%
# 3 variables function
a = 2.0
b = -3.0
c = 10.0

d = a * b + c

print(d)

# %%
h = 0.0001

a = 2.0
b = -3.0
c = 10.0

# 3 variables function
f_abc = lambda a=a, b=b, c=c: a * b + c

# Partial derivatives of f_abc with respect to a
dfabc_da = lambda a, b, c: (f_abc(a + h, b, c) - f_abc(a, b, c)) / h

# Partial derivatives of f_abc with respect to b
dfabc_db = lambda a, b, c: (f_abc(a, b + h, c) - f_abc(a, b, c)) / h

# Partial derivatives of f_abc with respect to c
dfabc_dc = lambda a, b, c: (f_abc(a, b, c + h) - f_abc(a, b, c)) / h

print(f"f_abc(2.0, -3.0, 10.0) = {f_abc()}")
print(f"dfabc_da(2.0, -3.0, 10.0) = {dfabc_da(a, b, c)}")
print(f"dfabc_db(2.0, -3.0, 10.0) = {dfabc_db(a, b, c)}")
print(f"dfabc_dc(2.0, -3.0, 10.0) = {dfabc_dc(a, b, c)}")


# %%
class Value:
    def __init__(
        self,
        data: Union[int, float, "Value"],
        _prev: Tuple["Value", "Value"] = (),
        _op: str = "",
        label: str = "",
    ):
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
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value") -> "Value":
        return self + (-other)

    def __mul__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        return self * other**-1

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(other, (int, float)), "Support only int or float."

        x, y = self.data, other
        out = Value(x**y, (self,), f"**{y}")

        def _backward():
            # d(x**y)/dx = y * x**(y-1)
            self.grad += y * (x ** (y - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data

        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def _topo_sort(self) -> List["Value"]:
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
        self.grad = 1.0
        for node in reversed(self._topo_sort()):
            node._backward()

    def zero_grad(self):
        self.grad = 0.0
        for node in self._topo_sort():
            node.grad = 0.0


# %%
a = Value(2.0, label="a")
b = Value(4.0, label="b")
a / b, a - b

# %%

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = Value(a * b, label="e")

d = Value(e + c, label="d")

f = Value(-2.0, label="f")

L = Value(d * f, label="L")

# Same result
d, a * b + c

# %%
# Backpropagation
L.grad = 1.0  # dL/dL = 1
d.grad = f.data  # dL/dd = f
f.grad = d.data  # dL/df = d
c.grad = d.grad
e.grad = d.grad
a.grad = e.grad * b.data
b.grad = e.grad * a.data

# %%
draw_dot(L)


# %%


def lol():
    h = 0.0001
    h_v = Value(h, label="h")

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = Value(a * b, label="e")

    d = Value(e + c, label="d")

    f = Value(-2.0, label="f")
    L1 = Value(d * f, label="L").data

    #############

    a = Value(2.0, label="a") + h_v
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = Value(a * b, label="e")

    d = Value(e + c, label="d")

    f = Value(-2.0, label="f")
    L2 = Value(d * f, label="L").data

    dL = (L2 - L1) / h_v.data
    print(dL)


lol()

# %%
xs = np.arange(-5, 5, 0.2)
ys = np.tanh(xs)

plt.plot(xs, ys)
plt.show()

# %%
# Neural network

# Inputs (features)
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

x1w1 = Value(x1 * w1, label="x1w1")
x2w2 = Value(x2 * w2, label="x2w2")

x1w1_x2w2 = Value(x1w1 + x2w2, label="x1w1_x2w2")

y = Value(x1w1_x2w2 + b, label="y")

output = Value(y.tanh(), label="output")

# %%
draw_dot(output)

# %%
# Backpropagation
output.grad = 1.0
y.grad = 1 - output.data**2  # d(tanh(x))/dx = 1 - tanh(x)^2, tanh(x) = output.data

# Because y = x1w1 + x2w2 + b
x1w1_x2w2.grad = y.grad
b.grad = y.grad

x1w1.grad = x1w1_x2w2.grad
x2w2.grad = x1w1_x2w2.grad

# Because x1w1 = x1 * w1
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

# %%
# Backpropagation with __backward method
# Neural network

# Inputs (features)
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

x1w1 = Value(x1 * w1, label="x1w1")
x2w2 = Value(x2 * w2, label="x2w2")

x1w1_x2w2 = Value(x1w1 + x2w2, label="x1w1_x2w2")

y = Value(x1w1_x2w2 + b, label="y")

output = Value(y.tanh(), label="output")

# %%
output.grad = 1.0

# %%
output._backward()

# %%
y._backward()

# %%
b._backward()

# %%
x1w1_x2w2._backward()

# %%
x1w1._backward()
x2w2._backward()

# %%
draw_dot(output)

# %%
output

# %%
# Build a topological sort of the graph


def topo_sort(graph):
    visited = set()
    topo = []

    def _sort(graph):
        if graph not in visited:
            visited.add(graph)
            for node in graph._prev:
                _sort(node)
            topo.append(graph)

    _sort(graph)
    return topo


topo_sort(output)


# %%
# Backpropagation with main backward method
# Neural network

# Inputs (features)
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

x1w1 = Value(x1 * w1, label="x1w1")
x2w2 = Value(x2 * w2, label="x2w2")

x1w1_x2w2 = Value(x1w1 + x2w2, label="x1w1_x2w2")

y = Value(x1w1_x2w2 + b, label="y")

output = Value(y.tanh(), label="output")
output.backward()

# %%

draw_dot(output)

# %%
# Compare with the tanh derivative piece-by-piece
# Neural network

# Inputs (features)
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

x1w1 = Value(x1 * w1, label="x1w1")
x2w2 = Value(x2 * w2, label="x2w2")

x1w1_x2w2 = Value(x1w1 + x2w2, label="x1w1_x2w2")

y = Value(x1w1_x2w2 + b, label="y")

output = Value(y.tanh(), label="output")
output.backward()

draw_dot(output)

# %%
# Neural network 2 piece-by-piece
# Inputs (features)
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

x1w1 = Value(x1 * w1, label="x1w1")
x2w2 = Value(x2 * w2, label="x2w2")

x1w1_x2w2 = Value(x1w1 + x2w2, label="x1w1_x2w2")

y = Value(x1w1_x2w2 + b, label="y")

e = (2 * y).exp()
output = (e - 1) / (e + 1)

output.backward()

draw_dot(output)

# %%
a = Value(3.0, label="a")
b = Value(a + a, label="b")
b.backward()

draw_dot(b)

# %%
a = Value(-2.0, label="a")
b = Value(3.0, label="b")

d = Value(a * b, label="d")
e = Value(a + b, label="e")
f = Value(d * e, label="f")

f.backward()

draw_dot(f)


# %%
class Neuron:
    def __init__(self):
        # Weights by default are None will be initialized on the first call
        self.w = None
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: List[float]) -> "Value":
        self.w = (
            [Value(random.uniform(-1, 1)) for _ in range(len(x))]
            if self.w is None
            else self.w
        )

        out = sum((w * x for w, x in zip(self.w, x)), self.b).tanh()
        return out

    def parameters(self) -> List["Value"]:
        return self.w + [self.b]


class Layer:
    def __init__(self, n_neurons: int):
        self.neurons = [Neuron() for _ in range(n_neurons)]

    def __call__(self, x: List[float]) -> List["Value"]:
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List["Value"]:
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    def __init__(self, n_inputs: int, n_outputs: List[int]):
        self.inputs = [Layer(n_inputs)]
        self.outputs = [Layer(n) for n in n_outputs]

        self.layers = self.inputs + self.outputs

    def __call__(self, x: List[float]) -> List["Value"]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List["Value"]:
        return [param for layer in self.layers for param in layer.parameters()]

    def step(self, lr: float):
        for param in self.parameters():
            param.data += -lr * param.grad


x = [2.0, 3.0, -1.0]

neuron = MLP(3, [4, 4, 1])
neuron(x)

# %%
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]  # inputs

ys = [1.0, -1.0, -1.0, 1.0]  # targets

# %%
desired_loss, max_steps, step = 0.000001, 1000, 0

while step < max_steps:
    y_pred = [neuron(x) for x in xs]

    loss = sum([(ypred - ygt) ** 2 for ygt, ypred in zip(ys, y_pred)])

    loss.zero_grad()
    loss.backward()

    neuron.step(0.2)

    print(f"step: {step}; loss: {loss.data}; pred: {y_pred}")
    step += 1

    if loss.data <= desired_loss:
        break

# %%
# Loss
y_pred = [neuron(x) for x in xs]
loss = sum([(ypred - ygt) ** 2 for ygt, ypred in zip(ys, y_pred)])
loss

# %%
# Predictions
y_pred

# %%
# Backpropagation
loss.backward()
# Step
for param in neuron.parameters():
    param.data += -0.01 * param.grad

loss.zero_grad()

# %%
draw_dot(loss)

# %%
neuron.layers[0].neurons[0].w[0].grad

# %%
neuron.layers[0].neurons[0].w[0].data

# %%
neuron.layers[0].neurons[0].w[0].data

# %%
