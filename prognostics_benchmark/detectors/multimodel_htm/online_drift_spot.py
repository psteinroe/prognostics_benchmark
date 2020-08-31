from math import log
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'


def backMean(X, d):
    M = []
    w = X[:d].sum()
    M.append(w / d)
    for i in range(d, len(X)):
        w = w - X[i - d] + X[i]
        M.append(w / d)
    return np.array(M)


def rootsFinder(fun, jac, bounds, npoints, method):
    """
    Find possible roots of a scalar function

    Parameters
    ----------
    fun : function
        scalar function
    jac : function
        first order derivative of the function
    bounds : tuple
        (min,max) interval for the roots search
    npoints : int
        maximum number of roots to output
    method : str
        'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

    Returns
    ----------
    numpy.array
        possible roots of the function
    """
    if method == 'regular':
        step = (bounds[1] - bounds[0]) / (npoints + 1)
        X0 = np.arange(bounds[0] + step, bounds[1], step)
    elif method == 'random':
        X0 = np.random.uniform(bounds[0], bounds[1], npoints)

    def objFun(X, f, jac):
        g = 0
        j = np.zeros(X.shape)
        i = 0
        for x in X:
            fx = f(x)
            g = g + fx ** 2
            j[i] = 2 * fx * jac(x)
            i = i + 1
        return g, j

    minimize_bounds = None
    if bounds[0] < bounds[1]:  # more fail save
        minimize_bounds = [bounds] * len(X0)
    opt = minimize(lambda X: objFun(X, fun, jac), X0,
                   method='L-BFGS-B',
                   jac=True, bounds=minimize_bounds)

    X = opt.x
    np.round(X, decimals=5)
    return np.unique(X)


def log_likelihood(Y, gamma, sigma):
    """
    Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

    Parameters
    ----------
    Y : numpy.array
        observations
    gamma : float
        GPD index parameter
    sigma : float
        GPD scale parameter (>0)
    Returns
    ----------
    float
        log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
    """
    n = Y.size
    if gamma != 0:
        tau = gamma / sigma
        L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
    else:
        L = n * (1 + log(Y.mean()))
    return L


class OnlineDriftSpot:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    depth : int
        Number of observations to compute the moving average

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values. Is reset during the initialization which occurs after the probation period.

    Nt : int
        number of observed peaks

    Nt : bool
        If true, returns alarms. If False, always returns false for alarm

    W : np.array
        Actual normal window
    """

    def __init__(self, probationary_period, depth_ratio, q=1e-4, update_threshold_on_alarm=True, verbose=False):
        self.proba = q
        self.extreme_quantile = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0
        self.depth = int(probationary_period * depth_ratio)

        self.W = None
        self.update_threshold_on_alarm = update_threshold_on_alarm

        self.verbose = verbose

        # For initialization period
        self.is_initialized = False
        self.probationary_period = probationary_period
        self._init_data = np.zeros(probationary_period)
        self._iteration = -1  # we start counting at 0

    def add(self, value):
        self._iteration = self._iteration + 1

        if self._iteration <= self.probationary_period - 1:
            self._init_data[self._iteration] = value
            return False, value
        elif self.is_initialized is False:
            try:
                self._initialize(self._init_data)
            except Exception as e:
                if self.verbose:
                    print('Unable to initialize SPOT: {}. Skipping iteration...'.format(e))
                self._init_data = np.append(self._init_data, value)
                return False, value

        is_alarm = False

        Mi = self.W.mean()
        # If the observed value exceeds the current threshold (alarm case)
        if (value - Mi) > self.extreme_quantile:
            # raise alarm
            is_alarm = True
            if self.update_threshold_on_alarm is True:
                self._handle_peak(value=value, Mi=Mi)
        # case where the value exceeds the initial threshold but not the alarm ones
        elif (value - Mi) > self.init_threshold:
            self._handle_peak(value=value, Mi=Mi)
        else:
            self.n += 1
            self.W = np.append(self.W[1:], value)

        # Return boolean indicating if the given value causes an alarm and the current threshold
        return is_alarm, self.extreme_quantile + Mi

    def _handle_peak(self, value, Mi):
        self.peaks = np.append(self.peaks, value - Mi - self.init_threshold)
        self.Nt += 1
        self.n += 1
        # and we update the thresholds

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)  # + Mi
        self.W = np.append(self.W[1:], value)

    def _initialize(self, init_data):
        """
        Run the calibration (initialization) step
        """
        if self.verbose:
            print('Initializing SPOT...')
        n_init = init_data.size - self.depth

        M = backMean(init_data, self.depth)
        T = init_data[self.depth:] - M[:-1]  # new variable

        S = np.sort(T)  # we sort X to get the empirical quantile
        # t is fixed for the whole algorithm, but if the probationary period is too small, we ensure
        # that there are at least two peaks.
        self.init_threshold = S[min(len(S) - 3, int(0.98 * n_init))]

        # initial peaks
        self.peaks = T[T > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if self.verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if self.verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        # initialize actual normal window
        self.W = init_data[-self.depth:]

        self.is_initialized = True

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
            numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)
        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = rootsFinder(lambda t: w(self.peaks, t),
                                 lambda t: jac_w(self.peaks, t),
                                 (a + epsilon, -epsilon),
                                 n_points, 'regular')

        right_zeros = rootsFinder(lambda t: w(self.peaks, t),
                                  lambda t: jac_w(self.peaks, t),
                                  (b, c),
                                  n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = 0
        try:
            ll_best = log_likelihood(self.peaks, gamma_best, sigma_best)
        except ValueError:
            # We can run into math domain errors due to the "fix" we use during init
            pass

        # we look for better candidates
        for z in zeros:
            try:
                gamma = u(1 + z * self.peaks) - 1
                sigma = gamma / z
                ll = log_likelihood(self.peaks, gamma, sigma)
                if ll > ll_best:
                    gamma_best = gamma
                    sigma_best = sigma
                    ll_best = ll
            except ValueError:
                # we expect math domain errors but can simply skip them
                # because we are just searching for the best candidate
                continue

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
            GPD parameter
        sigma : float
            GPD parameter
        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results given by the run

        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
            (default = True) If True, alarms are plotted.
        Returns
        ----------
        list
            list of the plots

        """
        data = run_results['data']
        x = range(data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, data, color=air_force_blue)
        fig = [ts_fig]

        #        if 'upper_thresholds' in K:
        #            thup = run_results['upper_thresholds']
        #            uth_fig, = plt.plot(x,thup,color=deep_saffron,lw=2,ls='dashed')
        #            fig.append(uth_fig)
        #
        #        if 'lower_thresholds' in K:
        #            thdown = run_results['lower_thresholds']
        #            lth_fig, = plt.plot(x,thdown,color=deep_saffron,lw=2,ls='dashed')
        #            fig.append(lth_fig)

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            if len(alarm) > 0:
                plt.scatter(alarm, data[alarm], color='red')

        plt.xlim((0, data.size))

        return fig
