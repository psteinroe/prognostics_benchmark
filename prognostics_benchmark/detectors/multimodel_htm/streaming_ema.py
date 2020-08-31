class StreamingExponentialMovingAverage:

    def __init__(self, alpha, mean=0):
        self._alpha = alpha
        self._mean = mean

        self._weights = [1]

    def _beta(self):
        return 1 - self._alpha

    def update(self, new_value):
        redistributed_mean = self._beta() * self._mean

        mean_increment = self._alpha * new_value

        new_mean = redistributed_mean + mean_increment

        self._mean = new_mean

        updated_weights = [w * self._beta() for w in self._weights]

        self._weights = updated_weights

        self._weights.append(self._alpha)

        return new_mean
