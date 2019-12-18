import copy
import numpy as np


class Variable:
    def __init__(self, data):
        self.__class__ = Variable
        self.data = np.array(data, ndmin=2)
        # fix shapes due to ndmin
        if np.array(data).shape != self.data.shape:
            self.data = self.data.T
        self.shape = self.data.shape
        self.grad = None
        self.grad_fn = None
        self.children = []
        self.retained_values = {}
        self._freed = False
        self._fn = ""

    def item(self):
        """If Variable is a scalar returns it."""
        if self.shape == (1, 1):
            return self.data[0, 0]
        else:
            raise ValueError("only one element tensors can be converted to Python scalars")

    def add_fn(self, fn):
        """Add the function that is at the origin of this Variable."""
        self.grad_fn = fn

    def add_child(self, child):
        """Add a new child to children list, child is an operation where self is parent."""
        self.children.append(child)

    def remove_child(self, child):
        """Remove child from the list of children."""
        self.children.remove(child)

    def update_retained_values(self):
        """Updates retained_values, which is a copy of children and grad_fn.
        retained_values is used when retain_graph is set to True to not erase
        the real children list and grad_fn."""
        if self.retained_values == {}:
            self.retained_values = {
                "children": self.children[:],
                "grad_fn": self.grad_fn
            }

    def zero_grad(self):
        """Sets the grad to zero."""
        self.grad = np.zeros(self.shape)

    def set_defaults(self):
        """Sets the variable to its defauls options (keeping only the data)."""
        self.grad = None
        self.grad_fn = None
        self.children = []
        self.retained_values = {}
        self._freed = False
        self._fn = ""

    def _update_grad_help(self, variable, grad, child, retain_graph):
        """Help function for special cases like .sum(), .mean(), .t(), ..."""
        if "_variable" in variable.__dict__.keys():
            grad = self._update_grad_help(variable._variable,
                                          grad,
                                          child,
                                          retain_graph)
            if variable._fn == "sum":
                grad = grad.sum(variable._artefact)
                grad = grad.reshape(grad.shape[0], 1 if len(grad.shape) < 2 else grad.shape[1])
                grad_to_update = np.ones(variable._variable.shape) * grad
            elif variable._fn == "mean":
                grad = grad.mean(variable._artefact)
                grad = grad.reshape(grad.shape[0], 1 if len(grad.shape) < 2 else grad.shape[1])
                grad_to_update = np.ones(variable._variable.shape) * grad
            elif variable._fn == "transpose":
                grad_to_update = grad.T
            elif variable._fn == "items":
                grad_to_update = np.zeros(variable._variable.shape)
                grad_to_update[variable._artefact] = 1
                grad_to_update = grad_to_update * grad
            else:
                raise ValueError("The function is not Known !")

            if variable._variable.grad is None:
                variable._variable.grad = grad_to_update
            else:
                variable._variable.grad += grad_to_update
        return grad

    def update_grad(self, grad, child, retain_graph=False):
        """Updates the gradients of self.

        Args:
            - grad (array): the new gradients to update 
            - child (function): the child from where the gradients come
            - retain_graph (bool): specify if we keep the graph for later or not 
        """

        # for transpose, sum and mean
        grad = self._update_grad_help(self, grad, child, retain_graph)
        grad = np.ones(self.shape) * grad

        if grad.shape[0] != self.shape[0]:
            grad = grad.sum(0)[None, :]

        if grad.shape != self.shape:
            raise ValueError("Shape of gradients and shape of data missmatch.",
                             "\n\tShape of gradients: {}".format(grad.shape),
                             "\n\tShape of data: {}".format(self.shape))
        if self.grad is None:
            #######################################################################
            # TODO: Update the current grad (self.grad), if the previous value
            # is None. What should be the update ?
            #######################################################################
            self.grad = grad
            #######################################################################
            # --------------------------- END OF YOUR CODE ------------------------
            #######################################################################
        else:
            #######################################################################
            # TODO: Update the current grad(self.grad), if the previous value
            # is not None. What should be the update ?
            #######################################################################
            self.grad = self.grad + grad
            #######################################################################
            # --------------------------- END OF YOUR CODE ------------------------
            #######################################################################

        if retain_graph:
            self.update_retained_values()
            self.retained_values["children"].remove(child)
        else:
            self.remove_child(child)

    def backward(self, retain_graph=False):
        """Starts the backward pass.

        If None of the tests are triggered this should call the backward of the
        operation that has made this variable.

        Args:
            - retain_graph (bool): specify if you want to keep the graph for 
            later use.

        """
        if self.grad_fn is not None:
            # create local children and grad_fn accordind to retain graph or not
            if retain_graph:
                self.update_retained_values()
                grad_fn = self.retained_values["grad_fn"]
                children = self.retained_values["children"]
            else:
                grad_fn = self.grad_fn
                children = self.children
            if self.grad is None:
                if self.shape != (1, 1):
                    raise RuntimeError(
                        "grad can be implicitly created only for scalar outputs")
                self.grad = np.ones(self.shape)
                if self._fn == "items":
                    self.grad = np.zeros(self._variable.shape)
                    self.grad[self._artefact] = 1
                children = []
            if not len(children):
                #######################################################################
                # TODO: Call the backward of the operation that has build this Variable
                #######################################################################
                self.grad_fn.backward(self.grad, retain_graph)
                #######################################################################
                # --------------------------- END OF YOUR CODE ------------------------
                #######################################################################
                if not retain_graph:
                    self.grad_fn = None
        else:
            # check if we are in a leaf
            if self._freed:
                raise RuntimeError(
                    "Trying to backward through the graph a second time,"
                    "but the buffers have already been freed.")

    def clone(self):
        """."""
        var_cloned = copy.deepcopy(self)
        var_cloned.__dict__ = copy.deepcopy(self.__dict__)
        return var_cloned

    def sum(self, dim=None):
        """."""
        var = Variable(self.data.sum(axis=dim))
        var.grad_fn = self.grad_fn
        var._fn = "sum"
        var._artefact = dim
        var._variable = self
        self.add_child(var)
        return var

    def mean(self, dim=None):
        """."""
        var = Variable(self.data.mean(axis=dim))
        var.grad_fn = self.grad_fn
        var._fn = "mean"
        var._artefact = dim
        var.grad = np.ones(self.shape) / self.data.size
        var._variable = self
        self.add_child(var)
        return var

    def t(self):
        """."""
        var = Variable(self.data.T)
        var.grad_fn = self.grad_fn
        var._fn = "transpose"
        var._variable = self
        self.add_child(var)
        return var

    def __add__(self, other):
        """."""
        from functional import F
        return F.add(self, other)

    def __sub__(self, other):
        """."""
        from functional import F
        return F.sub(self, other)

    def __mul__(self, other):
        """."""
        from functional import F
        return F.mul(self, other)

    def __truediv__(self, other):
        """."""
        from functional import F
        return F.div(self, other)

    def __setitem__(self, pos, item):
        """."""
        self.data[pos] = item

    def __getitem__(self, pos):
        """."""
        if self.shape[0] == 1 and type(pos) == int:
            pos = (0, pos)
        var = Variable(self.data[pos])
        var.grad_fn = self.grad_fn
        var._fn = "items"
        var._artefact = pos
        var._variable = self
        self.add_child(var)
        return var

    def __str__(self):
        """Converts the class to string (e.g. to print the class)."""
        data_str = ",\n         ".join(str(self.data).split("\n"))
        grad_fn_str = ""
        if self.grad_fn is not None:
            grad_fn_str = ", grad_fn=<{}Backward>".format(self.grad_fn.name)
        return "Variable({}{})".format(data_str, grad_fn_str)

    def __repr__(self):
        """Uses the string representation of the class when called 'in command line mode'."""
        return self.__str__()
