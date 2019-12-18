from functions import *


class Functional:
    # operations
    def add(self, x, y):
        return Add(x, y).forward()

    def sub(self, x, y):
        return Sub(x, y).forward()

    def mul(self, x, y):
        return Mul(x, y).forward()

    def matmul(self, x, y):
        return MatMul(x, y).forward()

    def div(self, x, y):
        return Div(x, y).forward()

    def exp(self, x):
        return Exp(x).forward()

    def log(self, x):
        return Log(x).forward()

    def sin(self, x):
        return Sin(x).forward()

    def cos(self, x):
        return Cos(x).forward()

    def tan(self, x):
        return Tan(x).forward()

    # activations
    def sigmoid(self, x):
        return Sigmoid(x).forward()

    def tanh(self, x):
        return Tanh(x).forward()

    def relu(self, x):
        return ReLu(x).forward()

    def softmax(self, x, dim):
        return Softmax(x, dim).forward()


F = Functional()

if __name__ == "__main__":
    pass
