import numpy as np

# Model #
# --------------------------------------------------------- #

# random initial probability, (pi) -> p initial
weatherProb = np.array([0.6, 0.4])

#  conditional probability of weather evidences
#  observations through umbrella, Emission probability
weatherObservationProb = np.array([[0.9, 0.2], [0.1, 0.8]])

# transition probabilities
#                       today
#    yesterday          sunny       rainy
#    rainy              0.3         0.7
#    sunny              0.7         0.3

# Transmission probabilities
weatherTransitionProb = np.array([[0.3, 0.7], [0.7, 0.3]])


# --------------------------------------------------------- #


# Direct Sampling #
# --------------------------------------------------------- #


# at least 15 sequences of length 20.
# p% probability of returning 1.
# 1 -> sunny, 2-> rainy
# I'm not sure about where p value comes from. In the paper there's no such thing.
def sample(p):
    return np.random.binomial(1, p)


# length 20
def dayGenerator():
    return [sample(weatherProb[0]) for i in range(0, 20)]


# 15 sequences
def sequenceGenerator():
    return [dayGenerator() for i in range(0, 15)]


# --------------------------------------------------------- #


"""
Parameters of def forward()
pi -> Weather probability -> Random Initial Probability that I've made up... 
A -> Weather transmission probabilities 
B -> Weather observations probabilities -> Emission probabilities
"""


def forward(obs_seq, pi, A, B):
    T = len(obs_seq)  # length of the observed sequence. OUT -> 20
    N = A.shape[0]  # length of the weather transmission probabilities. OUT -> 2 [[0.3 0.7][0.7 0.3]]

    # print("Length of observed sequence is ->", T)
    # print("Length of transmission probabilities table is ->", N)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs_seq[0]]  # initializing the first value to the alpha(zeroes) array
    for t in range(1, T):
        alpha[t] = np.inner(alpha[t - 1], A) * B[:, obs_seq[t]]
    return alpha


def likelihood(alpha):
    # returns the sum of 2 values in alpha[-1] tuple. Which is the last tuple in the alpha[] 2d-array.

    return alpha[-1].sum()


Seq = np.array(sequenceGenerator())
# returns just the first Nth sequence -> Seq[N]
alpha = forward(Seq[0], weatherProb, weatherTransitionProb, weatherObservationProb)


# returns probability of all sequences
def accumulator():
    return [likelihood((forward(Seq[i], weatherProb, weatherTransitionProb, weatherObservationProb))) for i in
            range(0, len(Seq))]


# another accumulator that prints the occurrence probabilities of all the sequences
# Use it to display occurrence probability for each sequence
def accumulator_all():
    each_prob = accumulator()
    for i in range(0, len(each_prob)):
        print("Likelihood for this sequence -> ", Seq[i], "to be occured is ->", each_prob[i])


# Use this one to display the occurrence probability of just the first Sequence.
# print("Likelihood for this sequence -> ", Seq[0], "to be occured is ->", likelihood(alpha))

# Use this to display the occurrence probability of entire Sequence.
# accumulator_all()

