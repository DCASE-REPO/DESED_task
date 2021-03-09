import tqdm
import torch

class TorchScaler(torch.nn.Module):
    """
    This torch module implements scaling for input tensors, both instance based
    and dataset-wide statistic based.
    """

    def __init__(self, statistic="dataset", normtype="mvn", dims=(1, 2), eps=1e-8):
        super(TorchScaler, self).__init__()
        assert statistic in ["dataset", "instance"]
        assert normtype in ["mvn", "mn", "minmax"]
        if statistic == "dataset" and normtype == "minmax":
            raise NotImplementedError(
                "statistic==dataset and normtype==minmax is not currently implemented."
            )
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def fit(self, dataloader, transform_func=lambda x: x[0]):
        indx = 0
        for batch in tqdm.tqdm(dataloader):

            feats = transform_func(batch)
            if indx == 0:
                mean = torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared = (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            else:
                mean += torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared += (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            indx += 1

        mean /= indx + 1
        mean_squared /= indx + 1

        self.register_buffer("mean", mean)
        self.register_buffer("mean_squared", mean_squared)

    def forward(self, tensor):
        if self.statistic == "dataset":
            assert hasattr(self, "mean") and hasattr(
                self, "mean_squared"
            ), "TorchScaler should be fit before used if statistics=dataset"
            assert tensor.ndim == self.mean.ndim, "Pre-computed statistics "
            if self.normtype == "mn":
                return tensor - self.mean
            elif self.normtype == "mvn":
                var = self.mean_squared - self.mean ** 2
                return (tensor - self.mean) / (var + self.eps)
            else:
                raise NotImplementedError

        else:
            if self.normtype == "mn":
                return tensor - torch.mean(tensor, self.dims, keepdim=True)
            elif self.normtype == "mvn":
                return (tensor - torch.mean(tensor, self.dims, keepdim=True)) / (
                    torch.var(tensor, self.dims, keepdim=True) + self.eps
                )
            elif self.normtype == "minmax":
                return (tensor - torch.amin(tensor, dim=self.dims, keepdim=True)) / (
                    torch.amax(tensor, dim=self.dims, keepdim=True)
                    - torch.amin(tensor, dim=self.dims, keepdim=True)
                    + self.eps
                )
