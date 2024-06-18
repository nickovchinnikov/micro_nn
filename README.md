# Micro NN

### Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd)

### Changes:

You can create a `Value` with label based on the operation result:

```python
a = Value(-2.0, label="a")
b = Value(3.0, label="b")

d = Value(a * b, label="d")
e = Value(a + b, label="e")
f = Value(d * e, label="f")
```

In `Neuron` you don't need to init weights dims, initialization based on the input size:

```python
x = [2.0, 3.0, -1.0]

neuron = MLP(3, [4, 4, 1])
neuron(x)
```