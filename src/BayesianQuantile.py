import numpy as np
from scipy import optimize


class BayesianQuantilePredictor:
    # assume the observed data is normalized to the interval [0,1]
    # we use the step size 1/sqrt(t) and the uniform prior

    def __init__(self, exact=True, n_bins=50):
        # exact=True corresponds to the exact version
        # exact=False corresponds to the quantized version

        self.t = 1
        if exact:
            # initialize a dictionary to store the unique historical observations and their counts
            self.observations = {}
            self.exact = True
        else:
            # initialize the histogram
            self.observations = np.zeros(n_bins)
            self.bins = np.linspace(0, 1, n_bins + 1)
            self.bins_center = np.zeros(n_bins)
            for n in range(n_bins):
                self.bins_center[n] = (self.bins[n] + self.bins[n+1]) / 2
            self.bins_center = np.concatenate((np.array([0]),self.bins_center))
            self.bins_center = np.concatenate((self.bins_center, np.array([1])))
            self.exact = False
    
    def update(self, new_data):
        # note that new_data has to belong to the interval [0,1]
        if self.exact:
            if new_data in self.observations:
                # if the new data exists in previous observations, increase its count by 1
                self.observations[new_data] += 1
            else:
                # if the new data does not exist, add it with a count of 1
                self.observations[new_data] = 1
        else:
            # add one count to the bin corresponding to the new data
            if new_data == 1:
                self.observations[-1] += 1
            else:
                ind = np.digitize(new_data, self.bins) - 1
                self.observations[ind] += 1
        self.t += 1

    def regularized_cdf(self, x, confidence):
        if self.exact:
            total_count = 0
            for existing_value in self.observations.keys():
                if existing_value <= x:
                    total_count += self.observations[existing_value]
        else:
            if x == 1:
                total_count = np.sum(self.observations)
            else:
                ind = np.digitize(x, self.bins_center) - 1
                total_count = np.sum(self.observations[:ind])
        step_size = 1 / np.sqrt(self.t)
        return step_size * x + (1 - step_size) * total_count / (self.t - 1) - confidence

    def output(self, quantile):
        if self.t == 1:
            return quantile
        else:
            if self.regularized_cdf(0, quantile) >= 0:
                return 0
            elif self.regularized_cdf(1, quantile) <= 0:
                return 1
            else:
                return optimize.bisect(self.regularized_cdf, 0, 1, args=(quantile))


class DiscountedBayesian:
    # discounted version of the above algorithm
    # we only implement the discounting on the quantized version

    def __init__(self, n_bins=50, discount=0.99, lr_scalar=1):
        self.t = 1
        self.discount = discount
        self.lr = lr_scalar * np.sqrt(1 - discount) / (discount + np.sqrt(1 - discount))

        self.observations = np.ones(n_bins) / n_bins
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.bins_center = np.zeros(n_bins)
        for n in range(n_bins):
            self.bins_center[n] = (self.bins[n] + self.bins[n+1]) / 2
        self.bins_center = np.concatenate((np.array([0]),self.bins_center))
        self.bins_center = np.concatenate((self.bins_center, np.array([1])))

    
    def update(self, new_data):
        self.observations = self.observations * self.discount
        if new_data == 1:
            self.observations[-1] += 1 - self.discount
        else:
            ind = np.digitize(new_data, self.bins) - 1
            self.observations[ind] += 1 - self.discount
        self.t += 1

    def regularized_cdf(self, x, confidence):
        if x == 1:
            total_weight = 1
        else:
            ind = np.digitize(x, self.bins_center) - 1
            total_weight = np.sum(self.observations[:ind])
        return self.lr * x + (1 - self.lr) * total_weight - confidence

    def output(self, quantile):
        if self.t == 1:
            return quantile
        else:
            if self.regularized_cdf(0, quantile) >= 0:
                return 0
            elif self.regularized_cdf(1, quantile) <= 0:
                return 1
            else:
                return optimize.bisect(self.regularized_cdf, 0, 1, args=(quantile))
            

class BayesianQuantilePredictorScaled:
    # assume the observed data is normalized to the interval [0,1]

    def __init__(self, exact=True, n_bins=50):
        # exact=True corresponds to the exact version
        # exact=False corresponds to the quantized version

        self.t = 1
        if exact:
            # initialize a dictionary to store the unique historical observations and their counts
            self.observations = {}
            self.exact = True
        else:
            # initialize the histogram
            self.observations = np.zeros(n_bins)
            self.bins = np.linspace(0, 1, n_bins + 1)
            self.bins_center = np.zeros(n_bins)
            for n in range(n_bins):
                self.bins_center[n] = (self.bins[n] + self.bins[n+1]) / 2
            self.bins_center = np.concatenate((np.array([0]),self.bins_center))
            self.bins_center = np.concatenate((self.bins_center, np.array([1])))
            self.exact = False
    
    def update(self, new_data):
        # note that new_data has to belong to the interval [0,1/2]
        if self.exact:
            if new_data in self.observations:
                # if the new data exists in previous observations, increase its count by 1
                self.observations[new_data] += 1
            else:
                # if the new data does not exist, add it with a count of 1
                self.observations[new_data] = 1
        else:
            # add one count to the bin corresponding to the new data
            if new_data == 1:
                self.observations[-1] += 1
            else:
                ind = np.digitize(new_data, self.bins) - 1
                self.observations[ind] += 1
        self.t += 1

    def regularized_cdf(self, x, confidence):
        if self.exact:
            total_count = 0
            for existing_value in self.observations.keys():
                if existing_value <= x:
                    total_count += self.observations[existing_value]
        else:
            if x == 1:
                total_count = np.sum(self.observations)
            else:
                ind = np.digitize(x, self.bins_center) - 1
                total_count = np.sum(self.observations[:ind])
        step_size = 1 / np.sqrt(self.t)
        return step_size * np.maximum(2*x, 1) + (1 - step_size) * total_count / (self.t - 1) - confidence

    def output(self, quantile):
        if self.t == 1:
            return quantile
        else:
            if self.regularized_cdf(0, quantile) >= 0:
                return 0
            elif self.regularized_cdf(1, quantile) <= 0:
                return 1
            else:
                return optimize.bisect(self.regularized_cdf, 0, 1, args=(quantile))