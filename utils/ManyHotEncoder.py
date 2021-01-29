import numpy as np
import pandas as pd
from dcase_util.data import DecisionEncoder


class ManyHotEncoder:
    """ "
    Adapted after DecisionEncoder.find_contiguous_regions method in
    https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

    Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
    Multiple 1 can appear on the same line, it is for multi label problem.
    """

    def __init__(self, labels=None, n_frames=None):
        """
            Initialization of ManyHotEncoder instance.
        Args:
            labels: list, the classes which will be encoded
            n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
        """
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.n_frames = n_frames

    def encode_weak(self, labels):
        """Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        if type(labels) is str:
            if labels == "empty":
                y = np.zeros(len(self.labels)) - 1
                return y
            else:
                labels = labels.split(",")
        if type(labels) is pd.DataFrame:
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]
        y = np.zeros(len(self.labels))
        for label in labels:
            if not pd.isna(label):
                i = self.labels.index(label)
                y[i] = 1
        return y

    def encode_strong_df(self, label_df):
        """
            Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                    If only filename (no onset offset) is specified, it will return the event on all the frames
                    onset and offset should be in frames
        Returns:
            y: numpy.array, Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert (
            self.n_frames is not None
        ), "n_frames need to be specified when using strong encoder"
        if type(label_df) is str:
            if label_df == "empty":
                y = np.zeros((self.n_frames, len(self.labels))) - 1
                return y
        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        onset = int(row["onset"])
                        offset = int(row["offset"])
                        y[
                            onset:offset, i
                        ] = 1  # means offset not included (hypothesis of overlapping frames, so ok)

        elif type(label_df) in [
            pd.Series,
            list,
            np.ndarray,
        ]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(
                    label_df.index
                ):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(label_df["onset"])
                        offset = int(label_df["offset"])
                        y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label is not "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] is not "":
                        i = self.labels.index(event_label[0])
                        onset = int(event_label[1])
                        offset = int(event_label[2])
                        y[onset:offset, i] = 1

                else:
                    raise NotImplementedError(
                        "cannot encode strong, type mismatch: {}".format(
                            type(event_label)
                        )
                    )

        else:
            raise NotImplementedError(
                "To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                "columns, or it is a list or pandas Series of event labels, "
                "type given: {}".format(type(label_df))
            )
        return y

    def decode_weak(self, labels):
        """Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels):
        """
        Decode the encoded strong labels

        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            result_labels: list, Decoded labels, list of list: [[label, onset offset], ...]
        """
        result_labels = []
        for i, label_column in enumerate(labels.T):
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)

            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append([self.labels[i], row[0], row[1]])
        return result_labels

    def state_dict(self):
        return {"labels": self.labels, "n_frames": self.n_frames}

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        n_frames = state_dict["n_frames"]
        return cls(labels, n_frames)
