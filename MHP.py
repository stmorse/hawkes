"""
Author: Steven Morse
Email: steventmorse@gmail.com
License: MIT License (see LICENSE in top folder)

Implementation of MAP EM algorithm for Hawkes process as described in:
https://arxiv.org/abs/2005.06542
https://stmorse.github.io/docs/orc-thesis.pdf

@article{xu2020modeling,
  title={Modeling human dynamics and lifestyle using digital traces},
  author={Xu, Sharon and Morse, Steven and Gonz{\'a}lez, Marta C},
  journal={arXiv preprint arXiv:2005.06542},
  year={2020}
}

NOTE: This is an update of the previous repo, improving several matrix
manipulations in the train method and adding more commenting.
"""

import time as T

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class MHP:
    def __init__(self, alpha=[[0.5]], mu=[0.1], omega=1.0):
        self.alpha = np.array(alpha)
        self.mu = np.array(mu)
        self.omega = float(omega)
        self.dim = self.mu.shape[0]

    def check_stability(self):
        '''Check stability of process (max alpha eigenvalue < 1)'''
        w, _ = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w.real))
        print(f'Max eigenvalue: {me:1.5f}')
        if me >= 1.:
            print('(WARNING) Unstable.')
        else:
            print('Appears stable')

    def get_rate(self, data, ct, d):
        """Return rate at time ct in dimension d"""
        seq = np.array(data)
        if not np.all(ct > seq[:,0]): 
            seq = seq[seq[:,0] < ct]
        return (
            self.mu[d] + np.sum([
                self.alpha[d,int(j)] * self.omega * np.exp(-self.omega*(ct-t)) 
                for t,j in seq
            ])
        )

    def generate(self, horizon=10, data=None):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        data = np.array([[0,0]]) if data is None else np.array(data)

        # total base rate and initial event time s
        Istar = np.sum(self.mu)
        s = np.random.exponential(scale=1./Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim), 
                              1, 
                              p=(self.mu / Istar))[0]
        data = np.append(data, [[s, n0]], axis=0)

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        while True:
            # get the last event time and attribution
            tj, uj = data[-1,0], int(data[-1,1])

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar 
                # (inclusive of last event)
                Istar = np.sum(lastrates) + \
                        self.omega * np.sum(self.alpha[:,uj])

            # generate new event
            s += np.random.exponential(scale=1./Istar)

            # calc rates at time s 
            # (use trick to take advantage of rates at last event, 
            # see thesis or paper linked at top of file)
            rates = (
                self.mu + np.exp(-self.omega * (s - tj)) * 
                (self.alpha[:,uj].flatten() * self.omega + lastrates - self.mu)
            )

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            n0 = np.random.choice(
                np.arange(self.dim+1), 
                1, 
                p=(np.append(rates, diff) / Istar)
            )[0]
            
            if n0 < self.dim:
                data = np.append(data, [[s, n0]], axis=0)
                # update lastrates
                lastrates = rates.copy()
            else:
                decIstar = True

            # if past horizon, done
            if s >= horizon:
                return data
            
    def train(self, seq, 
              Ahat=None, mhat=None, omega=None, 
              smx=None, tmx=None, regularize=False, 
              Tm=-1, maxiter=100, epsilon=0.01, stopping_criterion='iterations',
              verbose=True
    ):
        '''
        Implements MAP EM (from https://stmorse.github.io/docs/orc-thesis.pdf). 
        
        Optionally regularize with `smx` and `tmx` matrix (shape=(dim,dim)).
        In general, the `tmx` matrix is a pseudocount of parent events from column j,
        and the `smx` matrix is a pseudocount of child events from column j -> i.
        
        Parameters
        ----------
        seq : array-like (N, 2)
            Sequence of events, where each row is (time, event_id).
        Ahat : array-like, optional (dim, dim)
            Estimate of triggering kernel.  If not specified, will use the value from initialization.
        mhat : array-like, optional (dim,)
            Estimate of background rates.  If not specified, will use the value from initialization.
        omega : float, optional
            Fixed omega (not learned).  If not specified, will use the value from initialization.
        smx : array-like, optional (dim, dim)
            Regularization matrix for child events.  Must be specified if `regularize` is True.
        tmx : array-like, optional (dim, dim)
            Regularization matrix for parent events.  Must be specified if `regularize` is True.
        regularize : bool, optional
            Whether to use regularization.  Default is False.
        Tm : float, optional
            Maximum time horizon of sequence.  If not specified, will use the last time stamp.
            This only affects the log-likelihood for convergence testing, and is not critical.
        maxiter : int, optional
            Maximum number of iterations.  Default is 100.
        epsilon : float, optional
            Convergence threshold.  Default is 0.01.
        stopping_criterion : str, optional
            Convergence criterion.  Default is 'iterations'. 
            Options are 'iterations' (requires maxiter) or 'll' (requires epsilon)
        verbose : bool, optional
            Whether to print progress.  Default is True.
        return_p : bool, optional
            Whether to return the p_ii and p_ij matrices.  Default is False.

        Returns
        -------
        array-like (dim, dim)
            Post-training estimate of triggering kernel.
        array-like (dim,)
            Post-training estimate of background rates.
        array-like (N,)
            Post-training estimate of p_ii.
        array-like (N, N)
            Post-training estimate of p_ij.
        '''

        # check that if regularize=True, smx and tmx are specified
        # if not, turn off regularization and warn
        if regularize and (smx is None or tmx is None):
            print('Regularize is on but priors are not set. Turning off regularization.')
            regularize = False
        
        # use stored values unless something passed
        Ahat = Ahat if Ahat is not None else self.alpha
        mhat = mhat if mhat is not None else self.mu
        omega = omega if omega is not None else self.omega

        N = len(seq)
        dim = mhat.shape[0]
        Tm = float(seq[-1,0]) if Tm < 0 else float(Tm)
        sequ = seq[:,1].astype(int)

        p_ii = np.random.uniform(0.01, 0.99, size=N)
        p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

        t0 = T.time()

        # PRECOMPUTATIONS
        if verbose: print(f'Doing precomputations ... {T.time()-t0:.3f}')

        # element-wise: diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:,0]]).T, metric = 'euclidean')
        diffs[np.triu_indices(N)] = 0

        # element-wise: kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega*np.exp(-omega*diffs)

        # rowidx, colidx to allow numpy fancy indexing on Ahat
        colidx = np.tile(sequ.reshape((1,N)), (N,1))
        rowidx = np.tile(sequ.reshape((N,1)), (1,N))

        # indicator matrix for events
        # S[i,j] = 1 if event i is of type j
        S = np.zeros((N, dim), dtype=np.float32)
        S[np.arange(N), sequ] = 1

        # during training we need to compute the sum of the Gt kernel
        # if we approximate G(T-t) = 1, then we can compute the sum of the kernel
        # as the number of events of each type that have occurred before time t
        p_ones = np.ones((N, N))
        p_ones[np.triu_indices(N)] = 0
        seqcnts = (S.T @ p_ones) @ S   # (dim, dim)
        seqcnts[np.where(seqcnts == 0)] = 1  # hack to avoid div by zero

        k = 0
        old_LL = -10000   # log likelihood
        while k < maxiter:
            # compute A_{u_i,u_j} * G_{t_i,t_j}
            Auu = Ahat[rowidx, colidx]
            ag = np.multiply(Auu, kern)
            ag[np.triu_indices(N)] = 0

            # compute m_{u_i}
            mu = mhat[sequ]

            # compute total rates of u_i at time i
            rates = mu + np.sum(ag, axis=1)

            # compute matrix of p_ii and p_ij  (keep separate for later computations)
            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1,N)))
            p_ii = np.divide(mu, rates)

            # compute mhat:  mhat_u = (\sum_{u_i=u} p_ii) / T
            mhat = np.array([np.sum(p_ii[np.where(seq[:,1]==i)]) \
                             for i in range(dim)]) / Tm
            
            if regularize:
                Ahat = np.divide((S.T @ p_ij @ S) + (smx - 1), seqcnts + tmx)
            else:
                Ahat = np.divide(S.T @ p_ij @ S, seqcnts)

            if k % 10 == 0:
                term1 = np.sum(np.log(rates))
                term2 = Tm * np.sum(mhat)
                term3 = np.sum([
                    np.sum([Ahat[u,int(seq[j,1])] for j in range(N)]) 
                    for u in range(dim)
                ])
                new_LL = (1./N) * (term1 - term2 - term3)
                if verbose:
                    print(f'Iter {k} (LL: {new_LL}) ... {T.time()-t0:.3f}')
                if stopping_criterion == 'll' and np.abs(old_LL - new_LL) < epsilon:
                    print(f'Reached stopping criterion ... {T.time()-t0:.3f}')
                    break
                old_LL = new_LL

            k += 1

        print(f'Reached max iter {maxiter} (LL: {new_LL}) ... {T.time()-t0:.3f}')

        self.Ahat = Ahat
        self.mhat = mhat
        
        return Ahat, mhat, p_ii, p_ij

# -----------
# Plotting utility functions
# -----------

def plot_events_and_rates(mhp=None, data=None, horizon=None):

    horizon = np.amax(data[:,0]) if horizon is None else horizon
    dim = mhp.dim

    f, axarr = plt.subplots(
        dim*2, 1, 
        sharex='col', 
        gridspec_kw={'height_ratios':sum([[3,1] for i in range(dim)],[])}, 
        figsize=(8, dim*2)
    )

    xs = np.linspace(0, horizon, int(horizon*10))
    for i in range(dim):
        row = i * 2

        # plot rate
        r = [mhp.get_rate(data, ct, i) for ct in xs]
        axarr[row].plot(xs, r, 'k-')
        axarr[row].set_ylim([-0.01, np.amax(r)+(np.amax(r)/2.)])
        axarr[row].set_ylabel('$\lambda(t)_{%d}$' % i, fontsize=14)
        r = []

        # plot events
        subseq = data[data[:,1]==i][:,0]
        axarr[row+1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha=0.2)
        axarr[row+1].yaxis.set_visible(False)
        axarr[row+1].set_xlim([0, horizon])

    plt.tight_layout()

def plot_events(data, horizon=-1, labeled=True):
    if horizon < 0:
        horizon = np.amax(data[:,0])
    dim = int(np.amax(data[:,1])) + 1

    fig, ax = plt.subplots(1, 1, figsize=(10,2))

    for i in range(dim):
        subseq = data[data[:,1]==i][:,0]
        plt.plot(subseq, np.zeros(len(subseq)) - i, 'bo', alpha=0.2)

    if labeled:
        ax.set_yticklabels('')
        ax.set_yticks(-np.arange(0, dim), minor=True)
        ax.set_yticklabels([r'$e_{%d}$' % i for i in range(dim)], minor=True)
    else:
        ax.yaxis.set_visible(False)

    ax.set_xlim([0,horizon])
    ax.set_ylim([-dim, 1])
    ax.set_xlabel('t')
    # plt.tight_layout()