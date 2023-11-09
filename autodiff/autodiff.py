from collections import defaultdict
import numpy as np

class Var:
    '''
    A variable which contains a value and its local gradients
    Variables make up a computational graph
    '''

    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients


    def __add__(self, other):
        if isinstance(other, Var):
            return _add(self, other)
        elif isinstance(other, int) or isinstance(other, float):
            return _add(self, Var(other))
        elif isinstance(other, np.ndarray):
            return _var_numpy_add(self, other)
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for '+'")

    def __radd__(self, other):
        return self.__add__(other)


    def __sub__(self, other):
        if isinstance(other, Var):
            return _add(self, _neg(other))
        elif isinstance(other, int) or isinstance(other, float):
            return _add(self, _neg(Var(other)))
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for '-'")

    def __rsub__(self, other):
        return self.__sub__(other)


    def __mul__(self, other):
        if isinstance(other, Var):
            return _mul(self, other)
        elif isinstance(other, int) or isinstance(other, float):
            return _mul(self, Var(other))
        # elif isinstance(other, np.ndarray):
        #     return to_var(other) * self
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for '*'")
            
    def __rmul__(self, other):
        return self.__mul__(other)
        
    
    def __pow__(self, other):
        if isinstance(other, Var):
            return _pow(self, other) 
        elif isinstance(other, int) or isinstance(other, float):
            return _pow(self, Var(other))
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for '**'")


    def __truediv__(self, other):
        if isinstance(other, Var):
            return _mul(self, _inv(other))
        elif isinstance(other, int) or isinstance(other, float):
            return _mul(self, _inv(Var(other)))
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for '/'")

    def __rtruediv__(self, other):
        return self.__truediv__(other)


def _add(a: Var, b: Var):
    value = a.value + b.value
    local_gradients = (
        (a, 1),
        (b, 1)
    )
    return Var(value, local_gradients)

def _neg(a: Var):
    value = -1 * a.value
    local_gradients = (
        (a, -1),
    )
    return Var(value, local_gradients)

def _pow(a: Var, power: Var):
    value = a.value**power.value
    local_gradients = (
        (a, power.value * a.value**(power.value-1)),
    )
    return Var(value, local_gradients)

def _inv(a: Var):
    value = 1.0 / a.value
    local_gradients = (
        (a, -1 / a.value**2),
    )
    return Var(value, local_gradients)

def _mul(a: Var, b: Var):
    value = a.value * b.value
    local_gradients = (
        (a, b.value),
        (b, a.value),
    )
    return Var(value, local_gradients)

def sin(a: Var):
    value = np.sin(a.value)
    local_gradients = (
        (a, np.cos(a.value)),
    )
    return Var(value, local_gradients)

def log(a: Var):
    value = np.log(a.value)
    local_gradients = (
        (a, 1.0 / a.value),
    )
    return Var(value, local_gradients)

def exp(a: Var):
    value = np.exp(a.value)
    local_gradients = (
        (a, value),
    )
    return Var(value, local_gradients)

def _var_numpy_add(v: Var, a: np.ndarray):
    a_copy = a.copy()
    for i in range(len(a_copy)):
        a_copy[i] = _add(a_copy[i], v)
    return a_copy
    # return np.vectorize(lambda x: x + v.value)(a_copy)

def grad(variable: Var):
    '''
    Compute the first derivatives of 'variable' with respect
    to the child variables
    '''

    gradients = defaultdict(lambda: 0)

    def compute(variable, path_value):
        '''
        Computes gradients by recursively multiplying local
        derivates along a path, and then summing paths together
        '''

        # base case is when no local_gradients is empty
        for child_variable, local_gradient in variable.local_gradients:
            path_to_child_val = path_value * local_gradient
            gradients[child_variable] += path_to_child_val
            compute(child_variable, path_to_child_val)
    compute(variable, path_value=1)

    return gradients

# Super basic vectorization
# Converts a numpy array into an array of Var objects
to_var = np.vectorize(lambda x : Var(x))

# Converts array of Var to values
to_vals = np.vectorize(lambda variable : variable.value)

