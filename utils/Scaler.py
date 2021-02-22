import time
import warnings

import numpy as np
import torch
import json
from utils.Logger import create_logger

logger = create_logger(__name__)


class Scaler:
    """
    operates on one or multiple existing datasets and applies operations
    """

    def __init__(self):
        self.mean_ = None
        self.mean_of_square_ = None
        self.std_ = None

    def mean(self, data, axis=-1):
        """
        Compute the mean incrementally.

        Args:
            data: batch to calculate the mean of.
            axis: axis or axes along which the means are computed.
                  (default value = -1, which means have at the end a mean vector of the last dimension)

        Return:
            mean: arithmetic mean along the specified axis.
        """

        if axis == -1:
            mean = data
            while len(mean.shape) != 1:
                mean = np.mean(mean, axis=0, dtype=np.float64)
        else:
            mean = np.mean(data, axis=axis, dtype=np.float64)
        return mean

    def variance(self, mean, mean_of_square):
        """
        Compute variance thanks to mean and mean of square

        Args:
            mean: mean_of_square
            mean_of_square: mean_of_square

        Return:
            variance

        """
        return mean_of_square - mean ** 2

    def means(self, dataset):
        """
        Splits a dataset in to train test validation.

        Args:
            dataset: dataset, from DataLoad class, each sample is an (X, y) tuple.
        """

        logger.info("computing mean")
        start = time.time()

        shape = None

        counter = 0
        for sample in dataset:
            if type(sample) in [tuple, list] and len(sample) == 2:
                batch_x, _ = sample
            else:
                batch_x = sample
            if type(batch_x) is torch.Tensor:
                batch_x_arr = batch_x.numpy()
            else:
                batch_x_arr = batch_x
            data_square = batch_x_arr ** 2
            counter += 1

            if shape is None:
                shape = batch_x_arr.shape
            else:
                if not batch_x_arr.shape == shape:
                    raise NotImplementedError(
                        "Not possible to add data with different shape in mean calculation yet"
                    )

            # assume first item will have shape info
            if self.mean_ is None:
                self.mean_ = self.mean(batch_x_arr, axis=-1)
            else:
                self.mean_ += self.mean(batch_x_arr, axis=-1)

            if self.mean_of_square_ is None:
                self.mean_of_square_ = self.mean(data_square, axis=-1)
            else:
                self.mean_of_square_ += self.mean(data_square, axis=-1)

        self.mean_ /= counter
        self.mean_of_square_ /= counter

        # ### To be used if data different shape, but need to stop the iteration before.
        # rest = len(dataset) - i
        # if rest != 0:
        #     weight = rest / float(i + rest)
        #     X, y = dataset[-1]
        #     data_square = X ** 2
        #     mean = mean * (1 - weight) + self.mean(X, axis=-1) * weight
        #     mean_of_square = mean_of_square * (1 - weight) + self.mean(data_square, axis=-1) * weight

        logger.info("time to compute means: " + str(time.time() - start))
        return self

    def std(self, variance):
        return np.sqrt(variance)

    def calculate_scaler(self, dataset):
        """
        Calculate mean and standard deviation of the dataset
        Args:
            dataset: dataset to calculate mean and standard deviation of
        Return:
            self.mean: mean
            self.std: standard deviation
        """
        self.means(dataset)
        variance = self.variance(self.mean_, self.mean_of_square_)
        self.std_ = self.std(variance)

        return self.mean_, self.std_

    def normalize(self, batch):
        
        if type(batch) is torch.Tensor:
            batch_ = batch.numpy()
            batch_ = (batch_ - self.mean_) / self.std_
            return torch.Tensor(batch_)
        else:
            return (batch - self.mean_) / self.std_

    def state_dict(self):
        if type(self.mean_) is not np.ndarray:
            raise NotImplementedError(
                "Save scaler only implemented for numpy array means_"
            )

        dict_save = {
            "mean_": self.mean_.tolist(),
            "mean_of_square_": self.mean_of_square_.tolist(),
        }
        return dict_save

    def save(self, path):
        dict_save = self.state_dict()
        with open(path, "w") as f:
            json.dump(dict_save, f)

    def load(self, path):
        with open(path, "r") as f:
            dict_save = json.load(f)

        self.load_state_dict(dict_save)

    def load_state_dict(self, state_dict):
        self.mean_ = np.array(state_dict["mean_"])
        self.mean_of_square_ = np.array(state_dict["mean_of_square_"])
        variance = self.variance(self.mean_, self.mean_of_square_)
        self.std_ = self.std(variance)


class ScalerPerAudio:
    """Normalize inputs one by one
    Args:
        normalization: str, in {"global", "per_channel"}
        type_norm: str, in {"mean", "max"}
    """

    def __init__(self, normalization="global", type_norm="mean"):
        self.normalization = normalization
        self.type_norm = type_norm

    def normalize(self, spectrogram):
        """Apply the transformation on data
        Args:
            spectrogram: np.array, the data to be modified, assume to have 3 dimensions

        Returns:
            np.array
            The transformed data
        """
        if type(spectrogram) is torch.Tensor:
            tensor = True
            spectrogram = spectrogram.numpy()
        else:
            tensor = False

        if self.normalization == "global":
            axis = None
        elif self.normalization == "per_band":
            axis = 0
        else:
            raise NotImplementedError("normalization is 'global' or 'per_band'")

        if self.type_norm == "standard":
            res_data = (spectrogram - spectrogram[0].mean(axis)) / (
                spectrogram[0].std(axis) + np.finfo(float).eps
            )
        elif self.type_norm == "max":
            res_data = spectrogram[0] / (
                np.abs(spectrogram[0].max(axis)) + np.finfo(float).eps
            )
        elif self.type_norm == "min-max":
            res_data = (spectrogram - spectrogram[0].min(axis)) / (
                spectrogram[0].max(axis)
                - spectrogram[0].min(axis)
                + np.finfo(float).eps
            )
        else:
            raise NotImplementedError(
                "No other type_norm implemented except {'standard', 'max', 'min-max'}"
            )
        if np.isnan(res_data).any():
            res_data = np.nan_to_num(res_data, posinf=0, neginf=0)
            warnings.warn(
                "Trying to divide by zeros while normalizing spectrogram, replacing nan by 0"
            )

        if tensor:
            res_data = torch.Tensor(res_data)
        return res_data

    def state_dict(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def load_state_dict(self, state_dict):
        pass
