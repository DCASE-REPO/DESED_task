import torch
from asteroid_filterbanks import MelGramFB, Encoder
from scipy.signal import windows


class Fbanks(torch.nn.Module):
    def __init__(
        self,
        n_mels=40,
        n_filters=400,
        kernel_size=400,
        stride=200,
        window=None,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        log=True,
        **kwargs
    ):
        super().__init__()
        self.log = log
        if window is not None:
            window = torch.from_numpy(windows.get_window(window, n_filters))
        self.melg = Encoder(
            MelGramFB(
                n_filters,
                kernel_size,
                stride,
                window,
                sample_rate,
                n_mels,
                f_min,
                f_max,
                **kwargs
            )
        )

    def forward(self, x):

        if x.ndim == 1:
            x = x.reshape(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.melg(x)
        if self.log:
            x = self.take_log(x)
        return x

    @classmethod
    def take_log(cls, x):
        return torch.clamp(20 * torch.log10(torch.clamp(x, min=1e-8)), max=80, min=-80)
