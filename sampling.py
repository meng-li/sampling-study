import itertools
import numpy as np
import pylab as p
import scipy as s
import scipy.stats as stats

proposal_mean = [1.0, 1.0]
proposal_std  = [2.0, 2.0]

target_x_0 = [-1.0, 1.0]
target_x_1 = [-1.0, 1.0]

def proposal_sample(size):

    x_0 = np.random.normal(proposal_mean[0], proposal_std[0], size)
    x_1 = np.random.normal(proposal_mean[1], proposal_std[1], size)

    return np.array( [x_0, x_1] ).T

def proposal_pdf(point):
    prob =  stats.norm.pdf(point[0], proposal_mean[0], proposal_std[0])
    prob *= stats.norm.pdf(point[1], proposal_mean[1], proposal_std[1])
    return prob

def target_pdf(point):
    if point[0] > target_x_0[1] or point[0] < target_x_0[0]:
        return 0
    if point[1] > target_x_1[1] or point[1] < target_x_1[0]:
        return 0

    prob = 1.0 / (target_x_0[1] - target_x_0[0])
    prob /= target_x_1[1] - target_x_1[0]

    return prob

def target_sample(size):
    x_0 = np.random.uniform(target_x_0[0], target_x_0[1], size)
    x_1 = np.random.uniform(target_x_1[0], target_x_1[1], size)

    return np.array( [x_0, x_1] )


def rejection_sampling_K():
    vertexes = itertools.product( target_x_0, target_x_1 )
    vertexes = list(vertexes)

    probs_target = map(target_pdf, vertexes)
    probs_proposal = map(proposal_pdf, vertexes)

    K = min( [k/v for k, v in zip(probs_target, probs_proposal)] )
 
    return K


def rejection_sampling(sample, k):
    '''takes a sample from the poposal distribution and produces a sample from the target distribution'''
    accepted = []
    rejected = []

    for i in xrange(sample.shape[0]):
        z_0 = sample[i, :]

        prob_proposal = proposal_pdf(z_0)
        prob_target   = target_pdf(z_0)

        u_0 = np.random.uniform(0, k * prob_proposal)
       
        if u_0 < prob_target:
            #accept
            accepted.append(z_0)
        else:
            #reject
            rejected.append(z_0)


    return np.array(accepted), np.array(rejected)
        

def plot_target():
    vertexes = np.array( [[target_x_0[0], target_x_1[0]], 
                         [target_x_0[0], target_x_1[1]],
                         [target_x_0[1], target_x_1[1]],
                         [target_x_0[1], target_x_1[0]],
                         [target_x_0[0], target_x_1[0]]]
                       )

    p.plot(map(lambda x: x[0], vertexes), map(lambda x: x[1], vertexes), 'red')

def plot_rejection():
    rejection_K = rejection_sampling_K()
    sample = proposal_sample(1000)
    accepted, rejected = rejection_sampling(sample, rejection_K )
    print 'mean of the proposal sample', sample.mean(0)
    print 'mean of the resulting sample', accepted.mean(0)

    plot_target()

    p.scatter(rejected[:, 0], rejected[:, 1], color='blue')
    p.scatter(accepted[:, 0], accepted[:, 1], color='green')

    p.grid(True)
    p.xlim(target_x_0[0] - 3, target_x_0[1] + 3)
    p.ylim(target_x_1[0] - 3, target_x_1[1] + 3)
    p.show()


if __name__ == '__main__':
    plot_rejection()
