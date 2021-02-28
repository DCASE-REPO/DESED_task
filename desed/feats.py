import torch
import torchaudio


class Fbanks(torch.nn.Module):
    def __init__(
        self,
        n_mels,
        n_fft,
        hop_length,
        win_length=None,
        f_min=0,
        f_max=None,
        power=2,
        normalized=False,
        fs=16000,
        take_log=True,
    ):
        super(Fbanks, self).__init__()

        if f_max is None:
            f_max = fs / 2
        self.mels = torchaudio.transforms.MelSpectrogram(
            fs,
            n_fft,
            win_length,
            hop_length,
            f_min,
            f_max,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
        )
        self.take_log = take_log

    def forward(self, x):

        mels = self.mels(x)

        if self.take_log:
            mels = torch.clamp(
                20 * torch.log10(torch.clamp(mels, min=1e-8)), max=80, min=-80
            )

        return mels
