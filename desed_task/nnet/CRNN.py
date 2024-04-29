import warnings

import torch
import torch.nn as nn
import torchaudio.transforms

from .CNN import CNN
from .RNN import BidirectionalGRU


class CRNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        cnn_integration=False,
        freeze_bn=False,
        use_embeddings=False,
        embedding_size=527,
        embedding_type="global",
        frame_emb_enc_dim=512,
        aggregation_type="global",
        specaugm_t_p=0.2,
        specaugm_t_l=5,
        specaugm_f_p=0.2,
        specaugm_f_l=10,
        dropstep_recurrent=0.0,
        dropstep_recurrent_len=5,
        **kwargs,
    ):
        """
            Initialization of CRNN model

        Args:
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            attention: bool, adding attention layer or not
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            cnn_integration: bool, integration of cnn
            freeze_bn:
            **kwargs: keywords arguments for CNN.
        """
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.aggregation_type = aggregation_type
        self.nclass = nclass
        self.dropstep_recurrent = dropstep_recurrent
        self.dropstep_recurrent_len = dropstep_recurrent_len

        self.specaugm_t_p = specaugm_t_p
        self.specaugm_t_l = specaugm_t_l
        self.specaugm_f_p = specaugm_f_p
        self.specaugm_f_l = specaugm_f_l

        n_in_cnn = n_in_channel

        if cnn_integration:
            n_in_cnn = 1

        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)

        if isinstance(self.nclass, (tuple, list)) and len(self.nclass) > 1:
            # multiple heads
            self.dense = torch.nn.ModuleList([])
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)
            for current_classes in self.nclass:
                self.dense.append(nn.Linear(n_RNN_cell * 2, current_classes))
                if self.attention:
                    self.dense_softmax.append(
                        nn.Linear(n_RNN_cell * 2, current_classes)
                    )

        else:
            if isinstance(self.nclass, (tuple, list)):
                self.nclass = self.nclass[0]
            self.dense = nn.Linear(n_RNN_cell * 2, self.nclass)
            self.sigmoid = nn.Sigmoid()

            if self.attention:
                self.dense_softmax = nn.Linear(n_RNN_cell * 2, self.nclass)
                self.softmax = nn.Softmax(dim=-1)

        if self.use_embeddings:
            if self.aggregation_type == "frame":
                self.frame_embs_encoder = nn.GRU(
                    batch_first=True,
                    input_size=embedding_size,
                    hidden_size=512,
                    bidirectional=True,
                )
                self.shrink_emb = torch.nn.Sequential(
                    torch.nn.Linear(2 * frame_emb_enc_dim, nb_in),
                    torch.nn.LayerNorm(nb_in),
                )
                self.cat_tf = torch.nn.Linear(2 * nb_in, nb_in)
            elif self.aggregation_type == "global":
                self.shrink_emb = torch.nn.Sequential(
                    torch.nn.Linear(embedding_size, nb_in), torch.nn.LayerNorm(nb_in)
                )
                self.cat_tf = torch.nn.Linear(2 * nb_in, nb_in)
            elif self.aggregation_type == "interpolate":
                self.cat_tf = torch.nn.Linear(nb_in + embedding_size, nb_in)
            elif self.aggregation_type == "pool1d":
                self.cat_tf = torch.nn.Linear(nb_in + embedding_size, nb_in)
            else:
                self.cat_tf = torch.nn.Linear(2 * nb_in, nb_in)

    def _get_logits_one_head(
        self, x, pad_mask, dense, dense_softmax, classes_mask=None
    ):
        strong = dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if classes_mask is not None:
            classes_mask = ~classes_mask[:, None].expand_as(strong)
        if self.attention in [True, "legacy"]:
            sof = dense_softmax(x)  # [bs, frames, nclass]
            if not pad_mask is None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention

            if classes_mask is not None:
                # mask the invalid classes, cannot attend to these
                sof = sof.masked_fill(classes_mask, -1e30)
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)

        if classes_mask is not None:
            # mask invalid
            strong = strong.masked_fill(classes_mask, 0.0)
            weak = weak.masked_fill(classes_mask[:, 0], 0.0)

        return strong.transpose(1, 2), weak

    def _get_logits(self, x, pad_mask, classes_mask=None):
        out_strong = []
        out_weak = []
        if isinstance(self.nclass, (tuple, list)):
            # instead of masking the softmax we can have multiple heads for each dataset:
            # maestro_synth, maestro_real and desed.
            # not sure which approach is better. We must try.
            for indx, c_classes in enumerate(self.nclass):
                dense_softmax = (
                    self.dense_softmax[indx] if hasattr(self, "dense_softmax") else None
                )
                c_strong, c_weak = self._get_logits_one_head(
                    x, pad_mask, self.dense[indx], dense_softmax, classes_mask
                )
                out_strong.append(c_strong)
                out_weak.append(c_weak)

            # concatenate over class dimension
            return torch.cat(out_strong, 1), torch.cat(out_weak, 1)
        else:
            dense_softmax = (
                self.dense_softmax if hasattr(self, "dense_softmax") else None
            )
            return self._get_logits_one_head(
                x, pad_mask, self.dense, dense_softmax, classes_mask
            )

    def apply_specaugment(self, x):

        if self.training:

            timemask = torchaudio.transforms.TimeMasking(
                self.specaugm_t_l, True, self.specaugm_t_p
            )
            freqmask = torchaudio.transforms.TimeMasking(
                self.specaugm_f_l, True, self.specaugm_f_p
            )  # use time masking also here
            x = timemask(freqmask(x.transpose(1, -1)).transpose(1, -1))

        return x

    def forward(self, x, pad_mask=None, embeddings=None, classes_mask=None):

        x = self.apply_specaugment(x)
        x = x.transpose(1, 2).unsqueeze(1)

        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        if self.use_embeddings:
            if self.aggregation_type == "global":
                x = self.cat_tf(
                    torch.cat(
                        (
                            x,
                            self.shrink_emb(embeddings)
                            .unsqueeze(1)
                            .repeat(1, x.shape[1], 1),
                        ),
                        -1,
                    )
                )
            elif self.aggregation_type == "frame":
                # there can be some mismatch between seq length of cnn of crnn and the pretrained embeddings, we use an rnn
                # as an encoder and we use the last state
                last, _ = self.frame_embs_encoder(embeddings.transpose(1, 2))
                embeddings = last[:, -1]
                reshape_emb = (
                    self.shrink_emb(embeddings).unsqueeze(1).repeat(1, x.shape[1], 1)
                )

            elif self.aggregation_type == "interpolate":
                output_shape = (embeddings.shape[1], x.shape[1])
                reshape_emb = (
                    torch.nn.functional.interpolate(
                        embeddings.unsqueeze(1), size=output_shape, mode="nearest-exact"
                    )
                    .squeeze(1)
                    .transpose(1, 2)
                )

            elif self.aggregation_type == "pool1d":
                reshape_emb = torch.nn.functional.adaptive_avg_pool1d(
                    embeddings, x.shape[1]
                ).transpose(1, 2)
            else:
                raise NotImplementedError

        if self.use_embeddings:
            if self.dropstep_recurrent and self.training:
                dropstep = torchaudio.transforms.TimeMasking(
                    self.dropstep_recurrent_len, True, self.dropstep_recurrent
                )
                x = dropstep(x.transpose(1, -1)).transpose(1, -1)
                reshape_emb = dropstep(reshape_emb.transpose(1, -1)).transpose(1, -1)
            x = self.cat_tf(self.dropout(torch.cat((x, reshape_emb), -1)))
        else:
            if self.dropstep_recurrent and self.training:
                dropstep = torchaudio.transforms.TimeMasking(
                    self.dropstep_recurrent_len, True, self.dropstep_recurrent
                )
                x = dropstep(x.transpose(1, 2)).transpose(1, 2)
                x = self.dropout(x)

        x = self.rnn(x)
        x = self.dropout(x)

        return self._get_logits(x, pad_mask, classes_mask)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
