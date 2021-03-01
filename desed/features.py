import torch
import torchaudio


class Fbanks(torch.nn.Module):
    def __init__(
        self,
        n_mels,
        n_fft,
        hop_length,
        window_length=None,
        fmin=0,
        fmax=None,
        power=2,
        normalized=False,
        fs=16000,
        take_log=True,
    ):
        super(Fbanks, self).__init__()

        if fmax is None:
            fmax = fs / 2
        self.mels = torchaudio.transforms.MelSpectrogram(
            fs,
            n_fft,
            window_length,
            hop_length,
            fmin,
            fmax,
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
