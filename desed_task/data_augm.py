import torch
import random


def frame_shift(mels, labels, net_pooling=4):
    bsz, n_bands, frames = mels.shape
    shifted = []
    new_labels = []
    for bindx in range(bsz):
        shift = int(random.gauss(0, 90))
        shifted.append(torch.roll(mels[bindx], shift, dims=-1))
        shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
        new_labels.append(torch.roll(labels[bindx], shift, dims=-1))
    return torch.stack(shifted), torch.stack(new_labels)


def mixup(audio_tensors, labels, alpha=2, beta=8, mixup_type="soft"):

    assert mixup_type in ["soft", "hard"]

    audio1, audio2 = torch.chunk(audio_tensors, 2, dim=0)
    lab1, lab2 = torch.chunk(labels, 2, dim=0)

    gains = (
        torch.tensor([random.betavariate(alpha, beta) for x in range(audio2.shape[0])])
        .to(audio1)
        .reshape(-1, 1, 1)
    )
    mixed = audio1 * gains + audio2 * (1 - gains)
    if mixup_type == "soft":
        new_labels = torch.clamp(lab1 * gains + lab2 * (1 - gains), min=0, max=1)
    elif mixup_type == "hard":
        new_labels = torch.clamp(lab1 + lab2, min=0, max=1)
    else:
        raise NotImplementedError

    return mixed, new_labels


def add_noise(mels, snrs=(6, 30)):
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand(
            (mels.shape[0],), device=mels.device
        ).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)  # linear domain
    sigma = torch.std(mels, dim=(1, 2), keepdim=True) / snr
    mels = mels + torch.randn(mels.shape, device=mels.device) * sigma

    return mels
