import numpy as np
from tqdm import tqdm

def forward(pi, A, O, observations):
    
    """
    Compute the forward probability matrix and scaling factor for a given HMM model and a sequence of observations.

    Args:
        pi: The initial probability distribution of the hidden states.
        A: The transition probability matrix between hidden states.
        O: The observation probability matrix.
        observations: The sequence of observations.

    Returns:
        alpha: The forward probability matrix.
        za: The scaling factor.
    """
    
    N = len(observations)
    S = len(pi)
    alpha = np.zeros((N, S))

    # base case
    alpha[0, :] = pi * O[:,observations[0]]
        
    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * O[s2, observations[i]]
                    
    return (alpha, np.sum(alpha[N-1,:]))

def backward(pi, A, O, observations):
    
    """
    Compute the backward probability matrix and scaling factor for a given HMM model and a sequence of observations.

    Args:
        pi: The initial probability distribution of the hidden states.
        A: The transition probability matrix between hidden states.
        O: The observation probability matrix.
        observations: The sequence of observations.

    Returns:
        beta: The backward probability matrix.
        zb: The scaling factor.
    """
        
    N = len(observations)
    S = len(pi)
    beta = np.zeros((N, S))
        
    # base case
    beta[N-1, :] = 1
        
    # recursive case
    for i in range(N-2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i+1, s2] * A[s1, s2] * O[s2, observations[i+1]]
        
    return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))

def baum_welch(o, N, iterations = 1000):
    
    """
    Trains a Hidden Markov Model using the Baum-Welch algorithm.

    Args:
    stock_prices: The sequence of stock prices.
        n_states: The number of hidden states.
        rand_seed: The random seed for the random number generator.
        iterations: The number of iterations to run the algorithm.

    Returns:
        pi: The initial probability distribution of the hidden states.
        A: The transition probability matrix between hidden states.
        O: The observation probability matrix.
    """
              
    T = len(o[0])
    M = int(max(o[0])) + 1 # Number of possible observations

    # initialise pi, A, O randomly

    pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
    pi=1.0/N*np.ones(N)-pi_randomizer

    a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
    a=1.0/N*np.ones([N,N])-a_randomizer

    b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
    b = 1.0/M*np.ones([N,M])-b_randomizer

    pi, A, O = np.copy(pi), np.copy(a), np.copy(b) # take copies, as we modify them
    S = pi.shape[0]
    
    training = o
    # do several steps of EM hill climbing
    for _ in tqdm(range(iterations)):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)
            
        for observations in training:   
            # compute forward-backward matrices
            alpha, za = forward(pi, A, O, observations)
            beta, zb = backward(pi, A, O, observations)
                
            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0,:] * beta[0,:] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i-1,s1] * A[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
                                                                        
        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
    return pi, A, O

def viterbi(n_states, starting_matrix, transition_matrix, observation_matrix, observation_sequence):
    
    """
    n_states: number of states
    starting_matrix: starting probability matrix
    transition_matrix: transition probability matrix
    observation_matrix: observation probability matrix
    observation_sequence: sequence of observations
    """
    
    n_observations = len(observation_sequence)
    delta = np.zeros((n_states, n_observations))
    delta[:, 0] = starting_matrix * observation_matrix[:, observation_sequence[0]]
    psi = np.zeros((n_states, n_observations))
    for t in range(1, n_observations):
        for j in range(n_states):
            delta[j, t] = np.max(delta[:, t-1] * transition_matrix[:, j]) * observation_matrix[j, observation_sequence[t]]
            psi[j, t] = np.argmax(delta[:, t-1] * transition_matrix[:, j])
    return delta, psi