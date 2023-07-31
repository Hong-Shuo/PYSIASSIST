class FeatureExtractor():
    def __init__(self, input, sample_frequency=4096):
        self.fs = sample_frequency
        self.input = np.array(input.reshape(-1, 3, self.fs))
        self.min_max_scaler = MinMaxScaler()
        self.fourier_transform = self.__discrete_fourier_transform()
        self.power_spectrum = self.__power_spectrum()

    def __discrete_fourier_transform(self):
        transform = np.fft.rfft(self.input)
        transform = np.abs(transform)
        return transform

    def __power_spectrum(self):
        PS = self.fourier_transform ** 2
        return PS

    def extract_features(self):
        required_shape = (
        self.fourier_transform.shape[0], self.fourier_transform.shape[1], self.fourier_transform.shape[2])
        Dim_1_transform = self.fourier_transform.reshape(-1, required_shape[2]).get()
        Dim_1_power_spectrum = self.power_spectrum.reshape(-1, required_shape[2]).get()
        scaled_fourier_transform = self.min_max_scaler.fit_transform(Dim_1_transform)
        scaled_PSD = self.min_max_scaler.fit_transform(Dim_1_power_spectrum)
        scaled_fourier_transform, scaled_PSD = scaled_fourier_transform.reshape(required_shape[0], required_shape[1],
                                                                                required_shape[2]), scaled_PSD.reshape(
            required_shape[0], required_shape[1], required_shape[2])
        features = np.concatenate((np.array(scaled_fourier_transform), np.array(scaled_PSD)))
        feature = features.transpose(1, 0, 2)
        return features.reshape(-1, 3, 2, scaled_PSD.shape[-1])


a = FeatureExtractor()