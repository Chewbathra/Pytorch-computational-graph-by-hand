import numpy as np

from variable import Variable


class _Function:
    def __init__(self, name, x, y=None):
        self.name = name
        self.x = x
        self.y = y

    def forward(self):
        self.x.add_child(self)
        if self.y is not None:
            self.y.add_child(self)
        result_variable = Variable(self.result)
        result_variable.grad_fn = self
        return result_variable

    def backward(self, grad, retain_graph):
        self._backward(grad)
        self.x.update_grad(self.dx, child=self, retain_graph=retain_graph)
        if self.y is not None:
            self.y.update_grad(self.dy, child=self, retain_graph=retain_graph)

        self.x.backward(retain_graph=retain_graph)
        if self.y is not None:
            self.y.backward(retain_graph=retain_graph)


class Add(_Function):
    """Adition of two elements."""

    def __init__(self, x, y):
        super().__init__("Add", x, y)
        self.result = x.data + y.data

    def _backward(self, grad):
        self.dx = grad # 1 * grad
        self.dy = grad # 1 * grad


class Sub(_Function):
    """Substraction of two elements."""

    def __init__(self, x, y):
        super().__init__("Sub", x, y)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = x.data - y.data
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = grad # 1 * grad
        self.dy = -grad #-1 * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Mul(_Function):
    """Element-wise multiplication."""

    def __init__(self, x, y):
        super().__init__("Mul", x, y)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = x.data * y.data
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = self.y.data * grad
        self.dy = self.x.data * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Div(_Function):
    """Element-wise divide."""

    def __init__(self, x, y):
        super().__init__("Div", x, y)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = x.data / y.data
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = 1/self.y.data * grad
        self.dy = -(self.x.data / self.y.data**2) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class MatMul(_Function):
    """Matrice multiplication."""

    def __init__(self, x, y):
        super().__init__("MatMul", x, y)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.matmul(x.data, y.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = grad.dot(self.y.data.T)
        self.dy = self.x.data.T.dot(grad)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Exp(_Function):
    """Exponential function."""

    def __init__(self, x):
        super().__init__("Exp", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.exp(x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = self.result * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Log(_Function):
    """Logarithmic function."""

    def __init__(self, x):
        super().__init__("Log", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.log(x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = 1 / self.x.data * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Sin(_Function):
    """Sinus function."""

    def __init__(self, x):
        super().__init__("Sin", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.sin(x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = np.cos(self.x.data) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Cos(_Function):
    """Cosinus function."""

    def __init__(self, x):
        super().__init__("Cos", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.cos(x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = -np.sin(self.x.data) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Tan(_Function):
    """Tangent function."""

    def __init__(self, x):
        super().__init__("Tan", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.tan(x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = (1 / np.cos(self.x.data)**2) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


# ACTIVATIONS

class Sigmoid(_Function):
    """Sigmoid."""

    def __init__(self, x):
        super().__init__("Sigmoid", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = 1 / (1 + np.exp(-x.data))
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = self.result*(1-self.result) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Tanh(_Function):
    """Tanh."""

    def __init__(self, x):
        super().__init__("Tanh", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.tanh(x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = (1-np.tanh(self.x.data)**2) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################


class Softmax(_Function):
    """Softmax."""

    def __init__(self, x, dim):
        super().__init__("Softmax", x)
        self.dim = dim
        x_norm = x.data - np.max(x.data)
        exp = np.exp(x_norm)
        self.result = exp / np.sum(exp, axis=dim, keepdims=True)

    def _backward(self, grad):
        # q_i(delta_{i,j} - q_j)
        if self.dim == 0:
            res = self.result.T
            (N, D) = res.shape
            grad = grad.T
        elif self.dim == 1:
            res = self.result
            (N, D) = res.shape
        else:
            raise NotImplementedError(
                "Backward for dim > 1 not implemented, Sorry :(")

        self.dx = res[:, None, :]
        self.dx = np.tensordot(self.dx, self.dx, axes=((1), (1)))
        self.dx = self.dx.swapaxes(1, 2)[np.arange(N), np.arange(N)]

        diag = np.tile(np.eye(D), (N, 1)).reshape(N, D, D)
        diag = res[:, :, None] * diag

        self.dx -= diag
        self.dx *= -1

        # chain rule
        self.dx = grad.dot(self.dx)[np.arange(N), np.arange(N)]
        if self.dim == 0:
            self.dx = self.dx.T


class ReLu(_Function):
    """ReLu."""

    def __init__(self, x):
        super().__init__("ReLu", x)
        #######################################################################
        # TODO: Implement the forward pass and put the result in self.result.
        # The notbook provide you the formulas for this operation.
        #######################################################################
        self.result = np.maximum(0, x.data)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def _backward(self, grad):
        #######################################################################
        # TODO: Implement the derivative dx for this opetation and add the
        # result of the chain rule on self.dx.
        #######################################################################
        self.dx = (self.x.data > 0) * grad
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################
