import warnings

import librosa
import numpy as np
import torch


class Transform:
    def transform_data(self, data):
        # Mandatory to be defined by subclasses
        raise NotImplementedError("Abstract object")

    def transform_label(self, label):
        # Do nothing, to be changed in subclasses if needed
        return label

    def _apply_transform(self, sample_no_index):
        data, label = sample_no_index
        if (
            type(data) is tuple
        ):  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                data[k] = self.transform_data(data[k])
            data = tuple(data)
        else:
            data = self.transform_data(data)
        label = self.transform_label(label)
        return data, label

    def __call__(self, sample):
        """Apply the transformation
        Args:
            sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if (
            type(sample[1]) is int
        ):  # Means there is an index, may be another way to make it cleaner
            sample_data, index = sample
            sample_data = self._apply_transform(sample_data)
            sample = sample_data, index
        else:
            sample = self._apply_transform(sample)
        return sample


class ApplyLog(Transform):
    """
    Convert ndarrays in sample to Tensors.
    """

    def transform_data(self, data):
        """
        Apply the transformation on data
        Args:
            data: np.array, the data to be modified
        Returns:
            np.array: The transformed data
        """
        return librosa.amplitude_to_db(data.T).T


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.
    The sequence should be on axis -2.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    shape = x.shape
    if shape[-2] <= max_len:
        padded = max_len - shape[-2]
        padded_shape = ((0, 0),) * len(shape[:-2]) + ((0, padded), (0, 0))
        x = np.pad(x, padded_shape, mode="constant")
    else:
        x = x[..., :max_len, :]
    return x


class PadOrTrunc(Transform):
    """
    Pad or truncate a sequence given a number of frames
    """

    def __init__(self, nb_frames, apply_to_label=False):
        """
        Inizialization of PadOrTrunc instance
        Args:
            nb_frames: int, the number of frames to match
        """
        self.nb_frames = nb_frames
        self.apply_to_label = apply_to_label

    def transform_label(self, label):
        if self.apply_to_label:
            return pad_trunc_seq(label, self.nb_frames)
        else:
            return label

    def transform_data(self, data):
        """Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return pad_trunc_seq(data, self.nb_frames)


class AugmentGaussianNoise(Transform):
    """
    Pad or truncate a sequence given a number of frames
    """

    def __init__(self, mean=0.0, std=None, snr=None):
        """
        Inizialization of AugmentGaussianNoise instance
        Args:
            mean: float, mean of the Gaussian noise to add
            std: float, std of the Gaussian noise to add
            snr: noise dictionary parameters
        """
        self.mean = mean
        self.std = std
        self.snr = snr

    @staticmethod
    def gaussian_noise(features, snr):
        """Apply gaussian noise on each point of the data

        Args:
            features: numpy.array, features to be modified
            snr: float, average snr to be used for data augmentation
        Returns:
            numpy.ndarray
            Modified features
        """
        # If using source separation, using only the first audio (the mixture) to compute the gaussian noise,
        # Otherwise it just removes the first axis if it was an extended one
        if len(features.shape) == 3:
            feat_used = features[0]
        else:
            feat_used = features
        std = np.sqrt(np.mean((feat_used ** 2) * (10 ** (-snr / 10)), axis=-2))
        try:
            noise = np.random.normal(0, std, features.shape)
        except Exception as e:
            warnings.warn(
                f"the computed noise did not work std: {std}, using 0.5 for std instead"
            )
            noise = np.random.normal(0, 0.5, features.shape)

        return features + noise

    def transform_data(self, data):
        """Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            (np.array, np.array)
            (original data, noisy_data (data + noise))
        """
        if self.std is not None:
            noisy_data = data + np.abs(np.random.normal(0, 0.5 ** 2, data.shape))
        elif self.snr is not None:
            noisy_data = self.gaussian_noise(data, self.snr)
        else:
            raise NotImplementedError("Only (mean, std) or snr can be given")
        return data, noisy_data


class ToTensor(Transform):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __init__(self, unsqueeze_axis=None):
        """
        Inizialization of ToTensor instance.
        Args:
        unsqueeze_axis: int, (Default value = None) add an dimension to the axis mentioned.
                        Useful to add a channel axis to use CNN.
        """
        self.unsqueeze_axis = unsqueeze_axis

    def transform_data(self, data):
        """
        Apply the transformation on data
        Args:
            data: np.array, the data to be modified
        Returns:
            res_data: np.array, The transformed data
        """
        res_data = torch.from_numpy(data).float()
        if self.unsqueeze_axis is not None:
            res_data = res_data.unsqueeze(self.unsqueeze_axis)
        return res_data

    def transform_label(self, label):
        return torch.from_numpy(label).float()  # float otherwise error


class Normalize(Transform):
    """
    Normalize inputs
    """

    def __init__(self, scaler):
        """
        Inizialization of Normalize class
        Args:
            scaler: Scaler object, the scaler to be used to normalize the data
        """
        self.scaler = scaler

    def transform_data(self, data):
        """
        Apply the transformation on data
        Args:
            data: np.array, the data to be modified
        Returns:
            np.array, The transformed data
        """
        return self.scaler.normalize(data)


class CombineChannels(Transform):
    """
    Combine channels when using source separation (to remove the channels with low intensity)
    """

    def __init__(self, combine_on="max", n_channel_mix=2):
        """
        Inizialization of CombineChannels instance
        Args:
            combine_on: str, in {"max", "min"}, the channel in which to combine the channels with the smallest energy
            n_channel_mix: int, the number of lowest energy channel to combine in another one
        """
        self.combine_on = combine_on
        self.n_channel_mix = n_channel_mix

    def transform_data(self, data):
        """
        Apply the transformation on data
        Args:
            data: np.array, the data to be modified, assuming the first values are the mixture,
                and the other channels the sources

        Returns:
            np.array: The transformed data
        """
        mix = data[:1]  # :1 is just to keep the first axis
        sources = data[1:]
        channels_en = (sources ** 2).sum(-1).sum(-1)  # Get the energy per channel
        indexes_sorted = channels_en.argsort()
        sources_to_add = sources[indexes_sorted[:2]].sum(0)
        if self.combine_on == "min":
            sources[indexes_sorted[2]] += sources_to_add
        elif self.combine_on == "max":
            sources[indexes_sorted[-1]] += sources_to_add
        return np.concatenate((mix, sources[indexes_sorted[2:]]))


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        """
        Inizialization of Compose class.
         Args:
            transforms: list of Transform objects, list of transforms to compose.
            Example of transform: ToTensor()
        """
        self.transforms = transforms

    def add_transform(self, transform):
        """
        Add a transformer to the list
        Args:
            transform: transformer to be added to the list
        Return:
            Compose instance of list of transformers
        """
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string


def get_transforms(
    frames=None,
    scaler=None,
    add_axis=0,
    noise_dict_params=None,
    combine_channels_args=None,
):
    """
        The functions gather all the transformers that are applied to the data.

    Args:
        frames: number of frames to consider
        scaler: scaler
        add_axis: dimensin to add
        noise_dict_params: dictionary with noise parameters
        combine_channel_args: combine channel arguments

    Return:
        Compose(transf): composition of transformer appended to the list.

    """
    transf = []
    unsqueeze_axis = None

    if add_axis is not None:
        unsqueeze_axis = add_axis

    if combine_channels_args is not None:
        transf.append(CombineChannels(*combine_channels_args))

    if noise_dict_params is not None:
        transf.append(AugmentGaussianNoise(**noise_dict_params))

    transf.append(ApplyLog())

    if frames is not None:
        transf.append(PadOrTrunc(nb_frames=frames))

    transf.append(ToTensor(unsqueeze_axis=unsqueeze_axis))

    if scaler is not None:
        transf.append(Normalize(scaler=scaler))

    return Compose(transf)
