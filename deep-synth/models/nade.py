import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Distribution
from torch.distributions import Categorical

"""
A Discrete NADE, as a torch Distribution object
"""
class DiscreteNADEDistribution(Distribution):

    def __init__(self, data_size, data_domain_sizes, hidden_size, w, v, c):
        self.data_size = data_size
        self.data_domain_sizes = data_domain_sizes
        self.hidden_size = hidden_size
        self.w = w
        self.v = v
        self.c = c
        self.temperature = 1

    def set_temperature(self, temp):
        self.temperature = temp

    def log_prob(self, x):
        log_probs = []

        batch_size = x.size()[0]
        
        # Make this have same batch size as x
        a = self.c
        a = a.unsqueeze(0)
        a = a.expand(batch_size, a.size()[1])

        for i in range(0, self.data_size):
            h = F.sigmoid(a)
            probs = F.softmax(self.v[i](h), 1)
            dist = Categorical(probs)
            val = x[:, i]
            lp = dist.log_prob(val)
            log_probs.append(lp.unsqueeze(1))
            normalized_val = val.float() / self.data_domain_sizes[i]  # Normalize to [0,1]
            normalized_val = normalized_val.unsqueeze(1)
            if i < self.data_size - 1:
                a = self.w[i](normalized_val) + a


        log_prob = torch.sum(torch.cat(log_probs, 1), 1)    # First dimension is batch dim

        return log_prob


    def sample(self):
        return self.sample_n(1)

    def sample_n(self, n):
        outputs = []

        batch_size = n
        
        # Make this have batch size
        a = self.c
        a = a.unsqueeze(0)
        a = a.expand(batch_size, a.size()[1])

        for i in range(0, self.data_size):
            h = F.sigmoid(a)
            logits = self.v[i](h)
            probs = F.softmax(logits / self.temperature, 1)
            dist = Categorical(probs)
            val = dist.sample().unsqueeze(1)
            outputs.append(val)
            normalized_val = val.float() / self.data_domain_sizes[i]  # Normalize to [0,1]
            if i < self.data_size - 1:
                a = self.w[i](normalized_val) + a

        outputs = torch.cat(outputs, 1)                        # First dimension is batch dim

        return outputs


"""
A Discrete NADE, as a nn Module
"""
class DiscreteNADEModule(nn.Module):

    def __init__(self, data_size, data_domain_sizes, hidden_size):
        super(DiscreteNADEModule, self).__init__()

        self.data_size = data_size
        self.data_domain_sizes = data_domain_sizes
        self.hidden_size = hidden_size

        # The initial bias
        self.c = Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

        # Initialize one linear module for every step of the first layer
        # (Need ModuleList to make automatic parameter registration work)
        self.w = nn.ModuleList()
        for i in range(0, data_size - 1):
            self.w.append(nn.Linear(1, hidden_size, bias=False))

        # Initialize one linear module for every step of the second layer
        self.v = nn.ModuleList()
        for i in range(0, data_size):
            domain_size = data_domain_sizes[i]
            self.v.append(nn.Linear(hidden_size, domain_size, bias=True))

        self.temperature = 1

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.c.data.uniform_(-stdv, stdv)

    def set_temperature(self, temp):
        self.temperature = temp

    def forward(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        nade = DiscreteNADEDistribution(self.data_size, self.data_domain_sizes, self.hidden_size,
            self.w, self.v, self.c)
        return nade.log_prob(x)

    def sample(self):
        nade = DiscreteNADEDistribution(self.data_size, self.data_domain_sizes, self.hidden_size,
            self.w, self.v, self.c)
        nade.set_temperature(self.temperature)
        return nade.sample()

    def sample_n(self, n):
        nade = DiscreteNADEDistribution(self.data_size, self.data_domain_sizes, self.hidden_size,
            self.w, self.v, self.c)
        nade.set_temperature(self.temperature)
        return nade.sample_n(n)


# ##### TEST
# n = DiscreteNADEModule(10, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5)
# # inp = torch.autograd.Variable(torch.LongTensor(4, 10).fill_(1))
# # lp = n(inp)
# # print(lp)
# out = n.sample_n(4)
# print(out)
# ##### END TEST
