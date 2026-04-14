# THEME: NN AND BACKPROPOGATION
# PART 0: BACKPROPOGATION AUTOGRAD ENGINE (class Value)

# Import libraries
import math
import random

# Wrapper for computational graph
class Value:
    
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_prev)
        self._op = _op
        
    def __repr__(self):
        return f'Value({self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self,other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
        
    def __radd__(self, other):
        return self + other
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self,other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
        
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        return self + (-other)
        
    def __pow__(self, other):
        assert isinstance(other, (int,float)), 'only int,float in pow'
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += out.grad * other * self.data**(other-1)
        out._backward = _backward
        return out
        
    def __truediv__(self, other):
        return self * other**-1
        
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1-out.data**2)
        out._backward = _backward
        return out
        
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        def funtopo(n): # topological sort
            if n not in visited:
                visited.add(n)
                for prev in n._prev:
                    funtopo(prev)
                topo.append(n)
        funtopo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()