# THEME: NEURAL NETWORKS INTRODUCTION AT SCALAR LEVEL
# PART 0: BACKPROPOGATION AUTOGRAD ENGINE IN NEURAL NETWORKS
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
        # Topological sort in forward pass manner
        def funtopo(n):
            if n not in visited:
                visited.add(n)
                for prev in n._prev:
                    funtopo(prev)
                topo.append(n)
        funtopo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# PART 1: NN MLP LIBRARY ON TOP VALUE AUTOGRAD ENGINE
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters (self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, listout):
        sz = [nin] + listout
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(listout))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
# PART 2: MLP TRAINING
# Prepare the model, inputs, labels
model = MLP(3,[4,4,1])
xs = [
    [1.0,-1.0,0.0],
    [0.3,-0.6,0.5],
    [0.5,-1.0,1.2],
    [0.1,-0.6,0.5],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Gradient descent: forward, backward, update, mse loss
for k in range (300):
    ypred = [model(x) for x in xs]
    loss = sum((ypred-ys)**2 for ypred, ys in zip(ypred, ys))
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():
        p.data += -0.1 * p.grad
    if k % 100 == 0 or (k+1)==300:
        print(f"{k:3} | loss = {loss.data}")

# Results that should fit ys(labels)
print(ypred)
