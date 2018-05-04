import torch
import numpy as np
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


# Model params
g_input_size = 1     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 1    # size of generated output vector
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size


print_interval = 200


# ### Uncomment only one of these
(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
# (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


#   Used to generate original data that the generator going to mimic
#   Is a Guassian Distribution
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.FloatTensor(np.random.normal(mu, sigma, (1, n)))


#   Input data to the generator
#   Is a uniform distribution
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)


#   Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)


#   Discriminator
#   Output a scalar: 0 means fake; 1 means real
class Discrimnator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discrimnator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        x = F.sigmoid(self.map3(x))
        return x


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return  [np.mean(d), np.std(d)]


''' Finally, the training loop alternates between two modes: 
    first training D on real data vs. fake data, with accurate 
    labels (think of this as Police Academy); and then training 
    G to fool D, with inaccurate labels (this is more like those
    preparation montages from Ocean’s Eleven). It’s a fight between
    good and evil, people.
'''

def train():
    return None


D = Discrimnator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
d_sampler = get_distribution_sampler(4, 1.25)
gi_sampler = get_generator_input_sampler()
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=2e-4, betas=(0.9, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.9, 0.999))

if __name__ == '__main__':
    for epoch in range(10000):
        for d_index in range(1):    # Trian Discriminator
            #   1. Train D on real+fake
            D.zero_grad()

            #   1A: Train D on real
            d_real_data = Variable(d_sampler(d_input_size))
            d_real_decision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # One = True
            d_real_error.backward() # Compute/store gradients, but don't change params

            #   1B: Train D on fake
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input.detach())   # Detach to avoid training G on these lables
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros((1))))    # Zero = Fake
            d_fake_error.backward()
            d_optimizer.step() #Only optimize D's parameters; changes based on stored gradients from backward

        for g_index in range(1):    # Train Generator
            #   2. Train G on D's response (but do not train D on these labels)
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))

            g_error.backward()
            g_optimizer.step()  #Only optimize G's parameters

        if epoch % print_interval == 0:
            print("%s: D: %s/%s G: %s (Real: %s, Fake: %s)" % (epoch,
                                                               extract(d_real_error)[0],
                                                               extract(d_fake_error)[0],
                                                               extract(g_error)[0],
                                                               stats(extract(d_real_data)),
                                                               stats(extract(d_fake_data))))

