import unittest
from core.mlp import Neuron, Layer, MLP
from core.value import Value


class TestNeuron(unittest.TestCase):
    def test_neuron_output(self):
        input = [0.5, 0.5]

        neuron = Neuron()
        output = neuron(input)

        self.assertIsInstance(output, Value)
        self.assertIsInstance(output.data, float)

        parameters = neuron.parameters()

        # 2 weights + 1 bias
        self.assertEqual(len(input) + 1, len(parameters))


class TestLayer(unittest.TestCase):
    def test_layer_output(self):
        input = [0.5, 0.5, 0.5]

        layer = Layer(len(input))
        self.assertEqual(len(layer.neurons), len(input))

        output = layer(input)

        self.assertIsInstance(output, list)
        self.assertEqual(len(output), len(input))
        self.assertIsInstance(output[0], Value)


class TestMLP(unittest.TestCase):
    def test_mlp_output(self):
        mlp = MLP(3, [2, 1])
        self.assertEqual(len(mlp.layers), 3)

        output = mlp([0.5, 0.5, 0.5])
        self.assertIsInstance(output, Value)

    def test_mlp_parameters(self):
        mlp = MLP(3, [2, 1])

        mlp([0.5, 0.5, 0.5])

        parameters = mlp.parameters()
        self.assertIsInstance(parameters, list)

        # 3 neurons in the input layer, 2 in the first hidden layer, 1 in the output layer
        # 3 neurons * (3 weights + 1 bias) = 12
        # 2 neurons * (3 weights + 1 bias) = 8
        # 1 neuron * (2 weights and 1) = 3
        self.assertEqual(len(parameters), 23)

    def test_mlp_training_step(self):
        mlp = MLP(3, [2, 1])

        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]  # inputs

        ys = [1.0, -1.0, -1.0, 1.0]  # targets

        y_pred = [mlp(x) for x in xs]

        parameters_before = [param.data for param in mlp.parameters()]

        loss = sum([(ypred - ygt) ** 2 for ygt, ypred in zip(ys, y_pred)])

        loss.zero_grad()
        loss.backward()

        mlp.step(0.2)

        parameters_after = [param.data for param in mlp.parameters()]
        self.assertNotEqual(parameters_before, parameters_after)


if __name__ == "__main__":
    unittest.main()
