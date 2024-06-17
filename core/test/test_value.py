import unittest

from core.value import Value


class TestValue(unittest.TestCase):
    def test__scalar_gate(self):
        # Test with int
        int_value = Value._scalar_gate(5)
        self.assertIsInstance(int_value, int)
        self.assertEqual(int_value, 5)

        # Test with float
        float_value = Value._scalar_gate(5.5)
        self.assertIsInstance(float_value, float)
        self.assertEqual(float_value, 5.5)

        # Test with unsupported types
        value = Value(10)
        with self.assertRaises(AssertionError):
            Value._scalar_gate(value)

        with self.assertRaises(AssertionError):
            Value._value_gate("unsupported")

    def test__value_gate(self):
        # Test with int
        int_value = Value._value_gate(5)
        self.assertIsInstance(int_value, Value)
        self.assertEqual(int_value.data, 5)

        # Test with float
        float_value = Value._value_gate(5.5)
        self.assertIsInstance(float_value, Value)
        self.assertEqual(float_value.data, 5.5)

        # Test with Value
        value = Value(10)
        same_value = Value._value_gate(value)
        self.assertEqual(same_value, value)

        # Test with unsupported type
        with self.assertRaises(AssertionError):
            Value._value_gate("unsupported")

    def test_add(self):
        a = Value(1)
        b = Value(2)
        c = a + b
        self.assertEqual(c.data, 3)

        # Check the backward pass
        c.grad = 1
        c._backward()  # call the _backward method
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_value_wrap(self):
        a = Value(1)
        b = 2
        c = Value(a + b, label="c")
        self.assertEqual(c.data, 3)
        self.assertEqual(c.label, "c")

    def test_radd(self):
        a = Value(2)
        b = 1
        c = b + a
        self.assertEqual(c.data, 3)

        # Check the backward pass
        c.grad = 1
        c._backward()  # call the _backward method
        self.assertEqual(a.grad, 1)

    def test_neg(self):
        a = Value(2)
        b = -a
        self.assertEqual(b.data, -2)

    def test_sub(self):
        a = Value(1)
        b = Value(2)
        c = a - b
        self.assertEqual(c.data, -1)

    def test_rsub(self):
        a = Value(2)
        b = 1
        c = b - a
        self.assertEqual(c.data, -1)

    def test_mul(self):
        a = Value(3)
        b = Value(2)
        c = a * b
        self.assertEqual(c.data, 6)

        # Check the backward pass
        c.grad = 1
        c._backward()  # call the _backward method
        self.assertEqual(a.grad, 2)
        self.assertEqual(b.grad, 3)

    def test_rmul(self):
        a = Value(2)
        b = 3
        c = b * a
        self.assertEqual(c.data, 6)

    def test_div(self):
        a = Value(3)
        b = Value(2)
        c = a / b
        self.assertEqual(c.data, 1.5)

    def test_pow(self):
        a = Value(2)
        b = a**3
        self.assertEqual(b.data, 8)

        # Check the backward pass
        b.grad = 1
        b._backward()  # call the _backward method
        self.assertEqual(a.grad, 3 * a.data**2)

    def test_exp(self):
        a = Value(2)
        b = a.exp()
        self.assertAlmostEqual(b.data, 7.38905609893065)

        # Check the backward pass
        b.grad = 1
        b._backward()
        self.assertAlmostEqual(a.grad, 7.38905609893065)

    def test_tanh(self):
        a = Value(2)
        b = a.tanh()
        self.assertAlmostEqual(b.data, 0.964027580075817)

        # Check the backward pass
        b.grad = 1
        b._backward()

        t = b.data
        dxdt = (1 - t**2) * b.grad

        self.assertAlmostEqual(a.grad, dxdt)

    def test_backward(self):
        a = Value(2)
        b = Value(3)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3)
        self.assertEqual(b.grad, 2)

    def test_backward2(self):
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

        self.assertAlmostEqual(x1.grad, -1.5)
        self.assertAlmostEqual(x2.grad, 0.5)
        self.assertAlmostEqual(w1.grad, 1.0)
        self.assertAlmostEqual(w2.grad, 0.0)
        self.assertAlmostEqual(b.grad, 0.5)
        self.assertAlmostEqual(x1w1.grad, 0.5)
        self.assertAlmostEqual(x2w2.grad, 0.5)
        self.assertAlmostEqual(x1w1_x2w2.grad, 0.5)
        self.assertAlmostEqual(y.grad, 0.5)
        self.assertAlmostEqual(output.grad, 1.0)

    def test_zero_grad(self):
        a = Value(2)
        b = Value(3)
        c = a * b
        c.backward()
        a.zero_grad()
        self.assertEqual(a.grad, 0)

    def test_zero_grad2(self):
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

        # Compute gradients
        output.backward()

        self.assertAlmostEqual(x1.grad, -1.5)
        self.assertAlmostEqual(x2.grad, 0.5)
        self.assertAlmostEqual(output.grad, 1.0)

        # Reset gradients
        output.zero_grad()

        self.assertAlmostEqual(x1.grad, 0)
        self.assertAlmostEqual(x2.grad, 0)
        self.assertAlmostEqual(w1.grad, 0)
        self.assertAlmostEqual(w2.grad, 0)
        self.assertAlmostEqual(b.grad, 0)
        self.assertAlmostEqual(x1w1.grad, 0)
        self.assertAlmostEqual(x2w2.grad, 0)
        self.assertAlmostEqual(x1w1_x2w2.grad, 0)
        self.assertAlmostEqual(y.grad, 0)
        self.assertAlmostEqual(output.grad, 0)


if __name__ == "__main__":
    unittest.main()
