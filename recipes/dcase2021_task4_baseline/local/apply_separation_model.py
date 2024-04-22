# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# taken from https://github.com/google-research/sound-separation

import glob
import os
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf
import torch
import torchaudio
import tqdm


class SeparationModel(object):
    """Tensorflow audio separation model."""

    def __init__(self, checkpoint_path, metagraph_path):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            new_saver = tf.train.import_meta_graph(metagraph_path)
            new_saver.restore(self.sess, checkpoint_path)

            self.input_placeholder = self.graph.get_tensor_by_name(
                "input_audio/receiver_audio:0"
            )
            self.output_tensor = self.graph.get_tensor_by_name("denoised_waveforms:0")

    def separate(self, mixture_waveform):
        """Separates a mixture waveform into sources.

        Args:
          mixture_waveform: numpy.ndarray of shape (num_samples,).

        Returns:
          numpy.ndarray of (num_sources, num_samples) of source estimates.
        """
        if mixture_waveform.ndim == 1:
            mixture_waveform_input = np.reshape(mixture_waveform, (1, 1, -1))
        elif mixture_waveform.ndim == 2:
            batch = mixture_waveform.shape[0]
            mixture_waveform_input = np.reshape(mixture_waveform, (batch, 1, -1))

        separated_waveforms = self.sess.run(
            self.output_tensor,
            feed_dict={self.input_placeholder: mixture_waveform_input},
        )
        return separated_waveforms


def separate_folder(model, in_dir, out_dir, regex="*wav"):
    """
    Separates the audio files contained in the in_dir folder and saves them in out_dir folder.

    Args:
        model (tf.Graph): restored pre-trained separation model
        in_dir (str): path to audio directory (audio to be resampled)
        out_dir (str): path to audio resampled directory
        regex (str, optional): regular expression for extension of file. Defaults to "*.wav".
    """
    compute = True
    files = glob.glob(os.path.join(in_dir, regex))
    if os.path.exists(out_dir):
        out_files = glob.glob(os.path.join(out_dir, regex))
        if len(files) == len(out_files):
            compute = False

    if compute:
        for f in tqdm.tqdm(files):
            audio, orig_fs = torchaudio.load(f)
            audio = audio.numpy()
            audio = np.mean(audio, 0)

            separated = model.separate(audio)
            separated = torch.from_numpy(separated)
            batch, sources, samples = separated.shape

            separated = separated.reshape(batch * sources, samples)

            os.makedirs(
                Path(os.path.join(out_dir, Path(f).relative_to(Path(in_dir)))).parent,
                exist_ok=True,
            )
            torchaudio.save(
                os.path.join(out_dir, Path(f).relative_to(Path(in_dir))),
                separated,
                orig_fs,
            )
    return compute
