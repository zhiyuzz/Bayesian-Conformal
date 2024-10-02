import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


class EmpiricalQuantilePredictor:
    def __init__(self):
        self.observations =[]
    
    def update(self, new_data):
        self.observations.append(new_data)

    def output(self, quantile):
        if not self.observations:
            return quantile
        else:
            return np.quantile(self.observations, quantile, method='inverted_cdf')


class SingleOGD:
    # 1D unprojected OGD with a single confidence level, on the quantile loss
    # the learning rate is 1/sqrt(t) * lr_scalar
    def __init__(self, quantile, lr_scalar=1, initial=0.5):
        self.quantile = quantile
        self.scalar = lr_scalar
        self.iterate = initial
        self.t = 1

    def update(self, new_data):
        if self.iterate >= new_data:
            gradient = 1 - self.quantile
        else:
            gradient = - self.quantile
        self.iterate = self.iterate - gradient * self.scalar / np.sqrt(self.t)
        self.t += 1
    
    def output(self):
        return self.iterate


class MultiOGD:
    def __init__(self, n_quantile=50):
        self.algorithms = {}
        self.n_quantile = n_quantile
        self.quantile_seq = np.linspace(0, 1, n_quantile)
        for n in range(n_quantile):
            self.algorithms[n] = SingleOGD(self.quantile_seq[n], initial=self.quantile_seq[n])

    def update(self, new_data):
        for n in range(self.n_quantile):
            self.algorithms[n].update(new_data)

    def output(self, quantile):
        ind = np.argmin(np.abs(self.quantile_seq - quantile))
        return self.algorithms[ind].output()


# The following baseline is the MVP algorithm from "Practical Adversarial Multivalid Conformal Prediction", by Bastani et al.(2022). 
# We use the original implementation from the authors, https://github.com/ProgBelarus/MultiValidPrediction/blob/main/src/MultiValidPrediction.py
# The only modification is to add the random seed as an input. 

class MultiValidPrediction:
    def __init__(self, delta=0.1, n_buckets=50, groups=[(lambda x : True)], eta=0.5, r=1000, normalize_by_counts=True, seed=None):
        # coverage parameter, want to be 1 - delta covered
        self.delta = delta
        # how many buckets do you want?
        self.n_buckets = n_buckets
        # groups, given as a collection of True/False outputting functions
        self.groups = groups
        self.n_groups = len(groups)
        # eta, should be set externally based on n_groups, n_buckets, and T
        self.eta = eta
        # nuisance parameter
        self.r = r
        # do you normalize computation by bucket-group counts? 
        self.normalize_by_counts = normalize_by_counts

        # NEW: initialize a random number generator with the specified seed
        if seed == None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        # actual thresholds played
        self.thresholds = []
        # scores encountered
        self.scores = []
        # feature vectors encountered
        self.xs = []

        # for each round: 1 = miscovered, 0 = covered
        self.err_seq = []
        # vvals[i][g] = v_value on bucket i and group g so far
        self.vvals = np.zeros((self.n_buckets, self.n_groups), dtype=np.float64)
        # bg_miscoverage[i][g] = how many times was (i, g) miscovered so far?
        self.bg_miscoverage = np.zeros((self.n_buckets, self.n_groups), dtype=int)
        # bg_counts[i][g] = how many times did (i, g) come up so far?
        self.bg_counts = np.zeros((self.n_buckets, self.n_groups), dtype=int)


    def predict(self, x):
        curr_groups = [i for i in range(self.n_groups) if (self.groups[i])(x)]
        if len(curr_groups) == 0: # arbitrarily return threshold 0 for points with zero groups
          return 0

        all_c_neg = True # are all c values nonpositive?
        cmps_prev = 0.0
        cmps_curr = 0.0
        overcalibr_log_prev = 0.0
        overcalibr_log_curr = 0.0

        for i in range(self.n_buckets):
            # compute normalized bucket-group counts
            norm_fn = lambda x: np.sqrt((x+1)*(np.log2(x+2)**2))
            bg_counts_norm = 1./norm_fn(self.bg_counts[i, curr_groups])
            
            # compute sign of cvalue for bucket i
            a = self.eta * self.vvals[i, curr_groups]
            if self.normalize_by_counts:
                a *= bg_counts_norm
            mx = np.max(a)
            mn = np.min(a)

            if self.normalize_by_counts:
                overcalibr_log_curr  =  mx + logsumexp( a - mx, b=bg_counts_norm)
                undercalibr_log_curr = -mn + logsumexp(-a + mn, b=bg_counts_norm)
            else:
                overcalibr_log_curr  =  mx + logsumexp( a - mx)
                undercalibr_log_curr = -mn + logsumexp(-a + mn)
            cmps_curr = overcalibr_log_curr - undercalibr_log_curr

            if cmps_curr > 0:
                all_c_neg = False
            
            if (i != 0) and ((cmps_curr >= 0 and cmps_prev <= 0) or (cmps_curr <= 0 and cmps_prev >= 0)):
                cvalue_prev = np.exp(overcalibr_log_prev) - np.exp(undercalibr_log_prev)
                cvalue_curr = np.exp(overcalibr_log_curr) - np.exp(undercalibr_log_curr)

                Z = np.abs(cvalue_prev) + np.abs(cvalue_curr)
                prob_prev = 1 if Z == 0 else np.abs(cvalue_curr)/Z
                if self.rng.random() <= prob_prev:
                    return (1.0 * i) / self.n_buckets - 1.0 /(self.r * self.n_buckets)
                else:
                    return 1.0 * i / self.n_buckets

            cmps_prev = cmps_curr
            overcalibr_log_prev = overcalibr_log_curr
            undercalibr_log_prev = undercalibr_log_curr

        return (1.0 if all_c_neg else 0.0)

    def update(self, x, threshold, score):
        curr_groups = [i for i in range(self.n_groups) if (self.groups[i])(x)]
        if len(curr_groups) == 0: # don't update on points with zero groups
          return

        self.thresholds.append(threshold)
        self.scores.append(score)
        self.xs.append(x)

        bucket = min(int(threshold * self.n_buckets + 0.5/self.r), self.n_buckets - 1)
        err_ind = int(score > threshold)
      
        # update vvals
        self.vvals[bucket, curr_groups] += self.delta - err_ind # (1-err_ind) - (1-delta)
        # update stats
        self.bg_counts[bucket, curr_groups] += 1
        self.bg_miscoverage[bucket, curr_groups] += err_ind
        self.err_seq.append(err_ind)

    def coverage_stats(self):
        return self.thresholds, self.err_seq
    

class MultiMVP:
    def __init__(self, n_quantile=50, n_buckets=50, groups=[(lambda x : True)], eta=0.5, r=1000, normalize_by_counts=True, seed=None):
        self.algorithms = {}
        self.n_quantile = n_quantile
        self.quantile_seq = np.linspace(0, 1, n_quantile)
        self.predictions = np.zeros(n_quantile)
        for n in range(n_quantile):
            self.algorithms[n] = MultiValidPrediction(1-self.quantile_seq[n], n_buckets, groups,eta, r, normalize_by_counts, seed)

    def update(self, x, new_data):
        for n in range(self.n_quantile):
            self.algorithms[n].update(x, self.predictions[n], new_data)

    def output(self, x, quantile):
        for n in range(self.n_quantile):
            self.predictions[n] = self.algorithms[n].predict(x)
        ind = np.argmin(np.abs(self.quantile_seq - quantile))
        return self.predictions[ind]