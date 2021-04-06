import torch

def log_mse_loss(separated, source, max_snr=30.0, bias_ref_signal=None):
  """Negative log MSE loss, the negated log of SNR denominator."""
  err_pow = torch.sum((source - separated)**2, dim=-1)
  snrfactor = 10.**(-max_snr / 10.)
  if bias_ref_signal is None:
    ref_pow = torch.sum((source)**2, dim=-1)
  else:
    ref_pow = torch.sum((bias_ref_signal)**2, dim=-1)
  bias = snrfactor * ref_pow
  return 10. * torch.log10(bias + err_pow + 1e-12)
