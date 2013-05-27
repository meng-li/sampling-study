import numpy as np
import pylab as p

proposal_mean = [0.0, 0.0]
proposal_std  = [1.0, 1.0]

target_x_0 = [-1.0, 1.0]
target_x_1 = [-1.0, 1.0]

def proposal_sample(size):

    x_0 = np.random.normal(proposal_mean[0], proposal_std[0], size)
    x_1 = np.random.normal(proposal_mean[1], proposal_std[1], size)

    return np.array( [x_0, x_1] )

def target_sample(size):
    x_0 = np.random.uniform(target_x_0[0], target_x_0[1], size)
    x_1 = np.random.uniform(target_x_1[0], target_x_1[1], size)

    return np.array( [x_0, x_1] )


proposal = proposal_sample(200)
target   = target_sample(200)

p.scatter(proposal[0], proposal[1], color='blue')
p.scatter(target[0], target[1], color='red')

p.grid(True)
p.xlim(target_x_0[0] - 2, target_x_0[1] + 2)
p.ylim(target_x_1[0] - 2, target_x_1[1] + 2)
p.show()
