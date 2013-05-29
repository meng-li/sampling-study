import itertools
import numpy as np
import pylab as p
import scipy as s
import scipy.stats as stats

#We use two dimensional (multivariate) Normal distribution (http://en.wikipedia.org/wiki/Multivariate_normal_distribution) 
#as a proposal distribution with proposal_mean as it's center and proposal_std as it's diagonal covariance matrix

proposal_mean = [1.0, 1.0]
proposal_std  = [1.5, 2.0]

#The target distribution is the uniform distribution over rectangle, bounded between
# x_0 >= target_x_0[0] && x_0 <= target_x_0[1]
# x_1 >= target_x_1[0] && x_1 <= target_x_1[1]

target_x_0 = [-1.0, 1.0]
target_x_1 = [-1.0, 1.0]

def proposal_sample(size, center = None):
    '''Returns a sample from the proposal distribution. The returned sample is a numpy array of (`size`, 2) shape. If `center` is not None, the center of the proposal distribution is shifted to `center`, otherwise `proposal_mean` is used.'''
    if center == None:
        center = proposal_mean

    x_0 = np.random.normal(center[0], proposal_std[0], size)
    x_1 = np.random.normal(center[1], proposal_std[1], size)

    return np.array( [x_0, x_1] ).T

def proposal_pdf(point):
    '''Returns the proposal distribution's pdf (probability density function) at the `point`.'''
    prob =  stats.norm.pdf(point[0], proposal_mean[0], proposal_std[0])
    prob *= stats.norm.pdf(point[1], proposal_mean[1], proposal_std[1])
    return prob

def target_pdf(point):
    '''Returns the target distribution's pdf at the `point`.'''
    if point[0] > target_x_0[1] or point[0] < target_x_0[0]:
        return 0
    if point[1] > target_x_1[1] or point[1] < target_x_1[0]:
        return 0

    prob = 1.0 / (target_x_0[1] - target_x_0[0])
    prob /= target_x_1[1] - target_x_1[0]

    return prob

def target_sample(size):
    '''Returns a sample from the target distribution (used only for the illustration). The returned sample is a numpy array of (`size`, 2) shape.'''
    x_0 = np.random.uniform(target_x_0[0], target_x_0[1], size)
    x_1 = np.random.uniform(target_x_1[0], target_x_1[1], size)

    return np.array( [x_0, x_1] )

def target_sample_x_0(x_1):
    '''Generates a value of the first coordinate from the target distribution conditioned by the second coordinate: x_0 ~ P(x_0 | x_1). Used by Gibbs sampling.'''
    x_0 = np.random.uniform(target_x_0[0], target_x_0[1], 1)
    return x_0

def target_sample_x_1(x_0):
    '''Generates a value of the second coordinate from the target distribution conditioned by the first coordinate: x_1 ~ P(x_1 | x_0). Used by Gibbs sampling.'''
    x_1 = np.random.uniform(target_x_1[0], target_x_1[1], 1)
    return x_1

def rejection_sampling_K():
    '''In order to perform rejection sampling, one needs to define a constant K such that q(x) * K >= p(x) for all x: p(x) > 0, where q(x) and p(x) are the proposal and the target distributions. In this function we estimate it as the maximal ratio of p(x) / q(x) over the vertexes of the target distribution's support (rectangle).'''

    vertexes = itertools.product( target_x_0, target_x_1 )
    vertexes = list(vertexes)

    probs_target = map(target_pdf, vertexes)
    probs_proposal = map(proposal_pdf, vertexes)

    K = max( [k/v for k, v in zip(probs_target, probs_proposal)] )
 
    return K


def rejection_sampling(sample, k):
    '''The function implements the rejection sampling technique. `sample` is a sample from the proposal distribution and `k` takes a sample from the proposal distribution and returns two sub-samples: points that are accepted and those that are rejected. The first sub-sample represents a sample from the target distribution.'''
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
        
def importance_sampling(sample):
    '''This function implements the importance sampling technique. `sample` is a sample from the proposal distribution. The function returns a vector of weights. These weights can be used to estimate some statistics of the target distribution using the sample `sample` from the proposal distribution.'''
    weights = []

    for i in xrange(sample.shape[0]):
        z_0 = sample[i, :]

        prob_proposal = proposal_pdf(z_0)
        prob_target   = target_pdf(z_0)

        weights.append( prob_target / prob_proposal )

    return np.array( weights )

def sampling_resampling(iterations, proposal_size, target_size):
    '''The function implements the sampling-importance-resampling technique. This algorithms runs in iterations, on each iteration `proposal_size` points from the proposal distribution are sampled. After that they are re-weighted just as in importance sampling, but these weights are used to re-sample `target_size` points. The latter are used as a sample from the target distribution. This function returns two numpy arrays: the resulting sample of shape (iterations * target_size, 2) and a sample of points that are not resampled on the second (resampling) step.'''
    target_sample = []
    source_sample = []

    for _ in xrange(iterations):
        sample = proposal_sample(proposal_size)
        weights = []

        for i in xrange(sample.shape[0]):
            z_0 = sample[i, :]

            source_sample.append(z_0)

            prob_proposal = proposal_pdf(z_0)
            prob_target   = target_pdf(z_0)

            weights.append( prob_target / prob_proposal )

        weights = np.array( weights )
        weights /= weights.sum()

        to_sample = np.random.multinomial(target_size, weights, size=1)[0, :]
        for i in xrange(to_sample.shape[0]):
            count = to_sample[i]
            point = [sample[i, 0], sample[i, 1]]

            target_sample += [ point ] * count 

    return np.array(target_sample), np.array(source_sample)

def metropolis_sampling(size):
    '''This function implements the Metropolis sampling technique. Returns two samples: the resulting sample in a numpy array (size, 2) shape and sample of points that were rejected (for the purpose of illustration).'''
    z_0 = (target_x_0[0] + target_x_0[1], target_x_1[0] + target_x_1[1])

    accepted, rejected = [], []

    for _ in xrange(size):
        z_1 = proposal_sample(1, z_0)[0]


        acceptance_prob = min(1.0, target_pdf(z_1) / target_pdf(z_0) )
        u = np.random.uniform(0, 1)

        if u < acceptance_prob:
            z_0 = z_1
        else:
            rejected.append(z_1)

        accepted.append(z_0)

    return np.array(accepted), np.array(rejected)

def gibbs_sampling(size):
    '''This function implements the Gibbs sampling technique. Returns a sample from the target distribution in a numpy array of (size, 2) shape.'''
    x_0 = target_x_0[0] + target_x_0[1]
    x_1 = target_x_1[0] + target_x_1[1]

    sample = [ (x_0, x_1) ]

    for _ in xrange(size / 2):
        
        x_0 = target_sample_x_0(x_1)
        sample.append( (x_0, x_1) )

        x_1 = target_sample_x_0(x_0)
        sample.append( (x_0, x_1) )

    return np.array(sample)


def plot_target():
    '''Plots a boundaries of the target distribution's support rectangle.'''
    vertexes = np.array( [[target_x_0[0], target_x_1[0]], 
                         [target_x_0[0], target_x_1[1]],
                         [target_x_0[1], target_x_1[1]],
                         [target_x_0[1], target_x_1[0]],
                         [target_x_0[0], target_x_1[0]]]
                       )

    p.plot(map(lambda x: x[0], vertexes), map(lambda x: x[1], vertexes), 'red')

def demo_rejection():
    size = 1000

    rejection_K = rejection_sampling_K()
    sample = proposal_sample(size)
    accepted, rejected = rejection_sampling(sample, rejection_K )
    print 'mean of the proposal sample', sample.mean(0)
    print 'mean of the resulting sample', accepted.mean(0)
    print 'accepted %s, rejected %s, total %s'%(accepted.shape[0], rejected.shape[0], size)

    plot_target()

    p.scatter(rejected[:, 0], rejected[:, 1], color='blue')
    p.scatter(accepted[:, 0], accepted[:, 1], color='green')

    p.grid(True)
    p.xlim(target_x_0[0] - 3, target_x_0[1] + 3)
    p.ylim(target_x_1[0] - 3, target_x_1[1] + 3)
    p.show()


def demo_importance():
    size = 1000

    sample = proposal_sample(size)
    weights = importance_sampling(sample)
    
    print 'mean of the proposal sample', sample.mean(0)
    print 'mean of the resulting sample', (sample * np.array([weights, weights]).T).mean(0)

def demo_sampling_resampling():
    iterations = 10
    proposal_size = 100
    target_size = 10
    
    
    resample, sample = sampling_resampling(iterations, proposal_size, target_size)

    print 'mean of the proposal sample', sample.mean(0)
    print 'mean of the resulting sample', resample.mean(0)
    print 'resample size %s, total %s'%(resample.shape[0], sample.shape[0])

    plot_target()

    p.scatter(resample[:, 0], resample[:, 1], color='green')

    p.grid(True)
    p.xlim(target_x_0[0] - 3, target_x_0[1] + 3)
    p.ylim(target_x_1[0] - 3, target_x_1[1] + 3)
    p.show()

def demo_metropolis():
    size = 10000
    sieve = [10 *k for k in xrange(1000)]

    accepted, rejected = metropolis_sampling(size)
    accepted = accepted[sieve, :]

    print 'mean of the resulting sample', accepted.mean(0)
    print 'accepted %s, rejected %s, total %s'%(accepted.shape[0], rejected.shape[0], size)

    plot_target()

    p.scatter(rejected[:, 0], rejected[:, 1], color='blue')
    p.scatter(accepted[:, 0], accepted[:, 1], color='green')

    p.grid(True)
    p.xlim(target_x_0[0] - 3, target_x_0[1] + 3)
    p.ylim(target_x_1[0] - 3, target_x_1[1] + 3)
    p.show()

def demo_gibbs():
    target_size = 1000
    sample = gibbs_sampling(target_size)

    print 'mean of the resulting sample', sample.mean(0)

    plot_target()

    p.scatter(sample[:, 0], sample[:, 1], color='green')

    p.grid(True)
    p.xlim(target_x_0[0] - 3, target_x_0[1] + 3)
    p.ylim(target_x_1[0] - 3, target_x_1[1] + 3)
    p.show()


if __name__ == '__main__':
    print 'target distribution is the uniform over the square with x_0 in [%s, %s] and x_1 in [%s, %s]'%(target_x_0[0], target_x_0[1], target_x_1[0], target_x_1[1])
    #demo_rejection()
    demo_importance()
    #demo_sampling_resampling()
    #demo_metropolis()
    #demo_gibbs()
