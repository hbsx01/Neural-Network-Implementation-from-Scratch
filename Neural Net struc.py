class Operation(object):

    def __init__(self):
        return
    
    def __call__(self, x):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError
        
        
class Identity(Operation):

    def __call__(self, z):

        self.dims = z.shape
        y = copy.deepcopy(z)
        return y
    
    def derivative(self, s=None):

        # Compute the derivatives
        if s is None:
            return np.ones(self.dims)
        else:
            return s


class Softmax(Operation):

    def __call__(self, z):
        v = np.exp(z)
        # Compute denominator (sum along rows)
        denom = np.sum(v, axis=1)
        # Softmax formula, duplicating the denom across rows
        self.y = v/np.tile(denom[:,np.newaxis], [1,np.shape(v)[1]])
        # Store self.y so you can use it in derivative
        return self.y

    def derivative(self, s):

        idx = np.nonzero(s)[1]  # Find one-hot categories

        # Create empty copies to populate
        s_gamma = np.zeros_like(s)
        y_gamma = np.zeros_like(self.y)
        kronecker = np.zeros_like(s)

        # Compute dy_k/dz_j 
        for j,gamma in enumerate(idx):
            s_gamma[j,:] = s[j,gamma]
            y_gamma[j,:] = self.y[j,gamma]
            kronecker[j,gamma] = 1.
        dydz = s_gamma*y_gamma*(kronecker-self.y)
        return dydz
    
    
class MSE(Operation):

    def __call__(self, y, t):

        # MSE formula
        self.n_samples = np.shape(t)[0]
        L = np.sum((y-t)**2)/2./self.n_samples
        self.dL = (y-t) / self.n_samples
        return L

    def derivative(self):

        # Compute the gradient of MSE w.r.t. network output
        return self.dL

        
class CategoricalCE(Operation):

    def __call__(self, y, t):
        
        self.t = t
        self.y = y
        return -np.sum(t * np.log(y)) / len(t)
        
    def derivative(self):

        return -self.t/self.y / len(self.t)
    
class CrossEntropy(Operation):

    def __call__(self, y, t):
        L = 0 
        return L

    def derivative(self):

        return 0.

    
    
class Layer(object):

    def __init__(self):
        return

    def __call__(self, x):
        raise NotImplementedError


class Population(Layer):


    def __init__(self, nodes, act=Identity()):
        self.nodes = nodes
        self.z = None
        self.h = None
        self.act = act
        self.params = []

    def __call__(self, x=None):
        if x is not None:
            self.z = x
            self.h = self.act(x)
        return self.h


class Connection(Layer):


    def __init__(self, from_nodes=1, to_nodes=1):
        super().__init__()

        self.W = np.random.randn(from_nodes, to_nodes) / np.sqrt(from_nodes)
        self.b = np.zeros(to_nodes)
        self.params = [self.W, self.b]

    def __call__(self, x=None):
        if x is None:
            print('Should not call Connection without arguments.')
            return
        P = len(x)
        if P>1:
            return x@self.W + np.outer(np.ones(P), self.b)
        else:
            return x@self.W + self.b


class DenseLayer(Layer):


    def __init__(self, from_nodes=1, to_nodes=1, act=Logistic()):
        self.L1 = Connection(from_nodes=from_nodes, to_nodes=to_nodes)
        self.L2 = Population(to_nodes, act=act)

    def __call__(self, x=None):
        if x is None:
            return self.L2.h
        else:
            # Calculate and return the operation of the two layers, L1 and L2
            return self.L2(self.L1(x))

class Network(object):


    def __init__(self):
        self.lyr = []
        self.loss = None

    def add_layer(self, L):

        self.lyr.append(L)

    def __call__(self, x):

        for l in self.lyr:
            x = l(x)
        return x


    def backprop(self, t, lrate=1.):
        
        # Set up top gradient
        dEdh = np.zeros_like(self.lyr[-1]())

        # Work our way down through the layers
        for i in range(len(self.lyr)-1, 0, -1):

            # References to the layer below, and layer above
            pre = self.lyr[i-1]   # layer below, (i-1)
            post = self.lyr[i]    # layer above, (i)
            #   post.L1.W contains the connection weights
            #   post.L1.b contains the biases
            #   post.L2.z contains the input currents
            #   post.L2.h contains the upper layer's activities

            # Compute dEdz from dEdh
            dEdz = dEdh

            # Parameter gradients
            dEdW = np.zeros_like(post.L1.W)
            dEdb = np.zeros_like(post.L1.b)

            # Project gradient through connection, to layer below
            dEdh = np.zeros_like(pre.h)

            # Update weight parameters using learn which is PAJN for me rn 
        
        
    def learn(self, ds, lrate=1., epochs=10):
        
        loss_history = []  # for plotting
        for epoch in range(epochs):
            if epoch%100==0:
                cost = 0.
                loss_history.append(cost)
                print(f'{epoch}: cost = {cost}')

        return np.array(loss_history)  
            
net = Network()

#layers
input_layer = Population(2)
h1 = DenseLayer(from_nodes=2, to_nodes=30, act=Logistic())
h2 = DenseLayer(from_nodes=30, to_nodes=10, act=Logistic())


# Logistic + CrossEntropy
#output_layer = DenseLayer(from_nodes=10, to_nodes=1, act=Logistic())
#net.loss = CrossEntropy()

# Softmax + Categorical CE
output_layer = DenseLayer(from_nodes=10, to_nodes=2, act=Softmax())
net.loss = CategoricalCE()

# Adding layers to the network, from bottom to top
net.add_layer(input_layer)
net.add_layer(h1)
net.add_layer(h2)
net.add_layer(output_layer)


# Train the network
loss_history = net.learn(ds, epochs=5000);

# Plot the progress of the cost
plt.plot(loss_history);
plt.xlabel('Epoch')
plt.ylabel('Cost');

# Sanity check, to see if output matches targets
y = net(ds.inputs())
print(f'Outputs:\n{y[:5,:]}')
print(f'Targets:\n{ds.targets()[:5,:]}')
ds.plot(labels=y)

# Accuracy of our model
def accuracy(y, t):
    '''
     ac = accuracy(y, t)
     
     Calculates the fraction of correctly classified samples.
     A sample is classified correctly if the largest element
     in y corresponds to where the 1 is in the target.
     
     Inputs:
       y  a batch of outputs, with one sample per row
       t  the corresponding batch of targets
       
     Output:
       ac the fraction of correct classifications (0<=ac<=1)
    '''
    true_class = np.argmax(t, axis=1)       # vector of indices for true class
    estimated_class = np.argmax(y, axis=1)  # vector of indices for estimated class
    errors = sum(true_class==estimated_class)  # add up how many times they match
    acc = errors / len(ds)    # divide by the total number of samples
    return acc

ac = accuracy(net(ds.inputs()), ds.targets())
print(f"Your model's training accuracy = {ac*100}%")
