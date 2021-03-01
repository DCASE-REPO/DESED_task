import os
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import pandas as pd
from .utils import batched_decode_preds, log_sedeval_metrics
from desed.data_augm import add_noise
from desed.features import Fbanks


class DESED(pl.LightningModule):
    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        sed_teacher,
        optimizer,
        train_data,
        valid_data,
        train_sampler=None,
        scheduler=None,
        scaler=None,
    ):
        super(DESED, self).__init__()
        self.hparams = hparams

        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = sed_teacher
        self.opt = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        if scaler is None:
            self.scaler = lambda x: x
        else:
            self.scaler = scaler

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        self.feats = Fbanks(**self.hparams["feats"], log=False)

        self.val_buffer_student = pd.DataFrame()
        self.val_buffer_teacher = pd.DataFrame()

    def update_ema(self, alpha, global_step, model, ema_model):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(1 - alpha, params.data)

    def take_log(self, mels):

        return Fbanks.take_log(mels)

    def training_step(self, batch, batch_indx):

        mixture, labels, padded_indxs = batch
        indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        mixture = add_noise(self.feats(mixture))

        mixture = self.take_log(mixture)
        batch_num = mixture.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(mixture).bool()
        weak_mask = torch.zeros(batch_num).to(mixture).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()

        # sed student forward
        strong_preds_student, weak_preds_student = self.sed_student(
            self.scaler(mixture)
        )

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds_student[strong_mask], labels[strong_mask]
        )
        # supervised loss on weakly labelled
        loss_weak = self.supervised_loss(weak_preds_student[weak_mask], labels_weak)
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak

        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.sed_teacher(
                self.scaler(mixture)
            )

        # we apply consistency between the predictions
        weight = (
            self.hparams["training"]["const_max"]
            * self.scheduler["scheduler"]._get_scaling_factor()
        )

        strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student, strong_preds_teacher.detach()
        )
        weak_self_sup_loss = self.selfsup_loss(
            weak_preds_student, weak_preds_teacher.detach()
        )
        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight

        tot_loss = tot_loss_supervised + tot_self_loss

        tqdm_dict = {
            "train_tot_sup": tot_loss_supervised,
            "train_tot_self": tot_self_loss,
            "lr": self.opt.param_groups[-1]["lr"],
            "step": self.scheduler["scheduler"].step_num,
        }

        tensorboard_logs = {
            "train/loss_strong": loss_strong,
            "train/loss_weak": loss_weak,
            "train/step": self.scheduler["scheduler"].step_num,
            "train/tot_self_loss": tot_self_loss,
            "train/weight": weight,
            "train/tot_supervised": strong_self_sup_loss,
            "train/weak_self_sup_loss": weak_self_sup_loss,
            "train/strong_self_sup_loss": strong_self_sup_loss,
            "train/lr": self.opt.param_groups[-1]["lr"],
        }

        output = OrderedDict(
            {"loss": tot_loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}
        )
        return output

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )

    def validation_step(self, batch, batch_indx):

        mixture, labels, padded_indxs, filenames = batch
        labels_weak = (torch.sum(labels, -1) >= 1).float()
        logmels = self.scaler(self.take_log(self.feats(mixture)))

        # prediction for student
        strong_preds_student, weak_preds_student = self.sed_student(logmels)
        loss_strong_student = self.supervised_loss(strong_preds_student, labels)
        loss_weak_student = self.supervised_loss(weak_preds_student, labels_weak)
        decoded_student = batched_decode_preds(
            strong_preds_student, filenames, self.encoder
        )
        self.val_buffer_student = self.val_buffer_student.append(decoded_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.sed_teacher(logmels)
        loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)
        loss_weak_teacher = self.supervised_loss(weak_preds_teacher, labels_weak)
        decoded_teacher = batched_decode_preds(
            strong_preds_teacher, filenames, self.encoder
        )

        self.val_buffer_teacher = self.val_buffer_teacher.append(decoded_teacher)

        output = OrderedDict(
            {
                "val_loss_student": loss_strong_student,
                "val_loss_weak_student": loss_weak_student,
                "val_loss_teacher": loss_strong_teacher,
                "val_loss_weak_teacher": loss_weak_teacher,
            }
        )
        return output

    def validation_epoch_end(self, outputs):

        avg_loss_strong_student = torch.stack(
            [x["val_loss_student"] for x in outputs]
        ).mean()
        avg_loss_weak_student = torch.stack(
            [x["val_loss_weak_student"] for x in outputs]
        ).mean()

        avg_loss_strong_teacher = torch.stack(
            [x["val_loss_teacher"] for x in outputs]
        ).mean()
        avg_loss_weak_teacher = torch.stack(
            [x["val_loss_weak_teacher"] for x in outputs]
        ).mean()

        ground_truth = pd.read_csv(self.hparams["data"]["val_tsv"], sep="\t")
        save_dir = os.path.join(self.logger.log_dir, "metrics")
        os.makedirs(save_dir, exist_ok=True)

        f1_student = log_sedeval_metrics(
            self.val_buffer_student, ground_truth, save_dir, self.current_epoch,
        )
        f1_teacher = log_sedeval_metrics(
            self.val_buffer_teacher, ground_truth, save_dir, self.current_epoch,
        )

        obj_function = -max(
            f1_student, f1_teacher
        )  # we want to maximize f1 event based score.

        tqdm_dict = {
            "val_loss_student": avg_loss_strong_student,
            "obj_function": obj_function,
        }

        tensorboard_logs = {
            "val/strong_loss_student": avg_loss_strong_student,
            "val/weak_loss_student": avg_loss_weak_student,
            "val/strong_loss_teacher": avg_loss_strong_teacher,
            "val/weak_loss_teacher": avg_loss_weak_teacher,
            "val/event_macro_F1_student": f1_student,
            "val/event_macro_F1_teacher": f1_teacher,
        }

        output = OrderedDict(
            {
                "obj_function": obj_function,
                "progress_bar": tqdm_dict,
                "log": tensorboard_logs,
            }
        )

        self.val_buffer_student = pd.DataFrame()  # free the buffers
        self.val_buffer_teacher = pd.DataFrame()

        return output

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.hparams["training"]["num_workers"],
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.hparams["training"]["num_workers"],
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader
