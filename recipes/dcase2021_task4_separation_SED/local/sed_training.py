import os
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import pandas as pd
from .utils import batched_decode_preds, log_sedeval_metrics
from desed.data_augm import add_noise, mixup, frame_shift
from desed.features import Fbanks
from pathlib import Path
from desed.utils.torch_utils import nanmean, nantensor


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

        self.val_buffer_student_synth = pd.DataFrame()
        self.val_buffer_teacher_synth = pd.DataFrame()

        self.val_buffer_student_weak = pd.DataFrame()
        self.val_buffer_teacher_weak = pd.DataFrame()

        self.val_buffer_student_eval = pd.DataFrame()
        self.val_buffer_teacher_eval = pd.DataFrame()

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
        mixture = self.feats(mixture)

        mixture = add_noise(mixture, self.hparams["training"]["noise_snr"])
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
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[strong_mask], labels[strong_mask]
            )

            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak
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
            "train/student/loss_strong": loss_strong,
            "train/student/loss_weak": loss_weak,
            "train/teacher/loss_strong": loss_strong_teacher,
            "train/teacher/loss_weak": loss_weak_teacher,
            "train/step": self.scheduler["scheduler"].step_num,
            "train/student/tot_self_loss": tot_self_loss,
            "train/weight": weight,
            "train/student/tot_supervised": strong_self_sup_loss,
            "train/student/weak_self_sup_loss": weak_self_sup_loss,
            "train/student/strong_self_sup_loss": strong_self_sup_loss,
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

        # prediction for student
        logmels = self.scaler(self.take_log(self.feats(mixture)))
        strong_preds_student, weak_preds_student = self.sed_student(logmels)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.sed_teacher(logmels)

        # we derive a mask based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_val_folder"]))
                    for x in filenames
                ]
            )
            .to(mixture)
            .bool()
        )
        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["synth_val_folder"]))
                    for x in filenames
                ]
            )
            .to(mixture)
            .bool()
        )
        mask_eval = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["pub_eval_folder"]))
                    for x in filenames
                ]
            )
            .to(mixture)
            .bool()
        )

        output = OrderedDict(
            {
                "loss_weak_student_on_weak": nantensor(()).to(mixture),
                "loss_weak_teacher_on_weak": nantensor(()).to(mixture),
                "loss_strong_student_on_synth": nantensor(()).to(mixture),
                "loss_strong_teacher_on_synth": nantensor(()).to(mixture),
                "loss_strong_student_on_eval": nantensor(()).to(mixture),
                "loss_strong_teacher_on_eval": nantensor(()).to(mixture),
            }
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()
            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )

            output.update(
                {
                    "loss_weak_student_on_weak": loss_weak_student,
                    "loss_weak_teacher_on_weak": loss_weak_teacher,
                }
            )
            filenames_weak = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["weak_val_folder"])
            ]

            decoded_student_weak = batched_decode_preds(
                strong_preds_student[mask_weak], filenames_weak, self.encoder
            )
            self.val_buffer_student_weak = self.val_buffer_student_weak.append(
                decoded_student_weak
            )
            decoded_teacher_weak = batched_decode_preds(
                strong_preds_teacher[mask_weak], filenames_weak, self.encoder
            )
            self.val_buffer_teacher_weak = self.val_buffer_teacher_weak.append(
                decoded_teacher_weak
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            output.update(
                {
                    "loss_strong_student_on_synth": loss_strong_student,
                    "loss_strong_teacher_on_synth": loss_strong_teacher,
                }
            )
            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["synth_val_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_synth], filenames_synth, self.encoder
            )
            self.val_buffer_student_synth = self.val_buffer_student_synth.append(
                decoded_student_strong
            )

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth], filenames_synth, self.encoder
            )
            self.val_buffer_teacher_synth = self.val_buffer_teacher_synth.append(
                decoded_teacher_strong
            )

        if torch.any(mask_eval):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_eval], labels[mask_eval]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_eval], labels[mask_eval]
            )

            output.update(
                {
                    "loss_strong_student_on_eval": loss_strong_student,
                    "loss_strong_teacher_on_eval": loss_strong_teacher,
                }
            )
            filenames_eval = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["pub_eval_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_eval], filenames_eval, self.encoder
            )
            self.val_buffer_student_eval = self.val_buffer_student_eval.append(
                decoded_student_strong
            )

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_eval], filenames_eval, self.encoder
            )
            self.val_buffer_teacher_eval = self.val_buffer_teacher_eval.append(
                decoded_teacher_strong
            )

        return output

    def validation_epoch_end(self, outputs):

        loss_weak_student_on_weak = nanmean(
            torch.stack([x["loss_weak_student_on_weak"] for x in outputs])
        )
        loss_weak_teacher_on_weak = nanmean(
            torch.stack([x["loss_weak_teacher_on_weak"] for x in outputs])
        )

        loss_strong_student_on_synth = nanmean(
            torch.stack([x["loss_strong_student_on_synth"] for x in outputs])
        )
        loss_strong_teacher_on_synth = nanmean(
            torch.stack([x["loss_strong_teacher_on_synth"] for x in outputs])
        )

        loss_strong_student_on_eval = nanmean(
            torch.stack([x["loss_strong_student_on_eval"] for x in outputs])
        )
        loss_strong_teacher_on_eval = nanmean(
            torch.stack([x["loss_strong_teacher_on_eval"] for x in outputs])
        )

        # TODO uncomment this
        """
        # weak dataset
        ground_truth = pd.read_csv(self.hparams["data"]["weak_val_tsv"],
                                   sep="\t")
        save_dir = os.path.join(self.logger.log_dir, "metrics_weak_val")
        os.makedirs(save_dir, exist_ok=True)

        _, _, weak_student_seg_macro, weak_student_seg_micro = log_sedeval_metrics(
            self.val_buffer_student_weak, ground_truth, save_dir,
            self.current_epoch,
        )
        _, _, weak_teacher_seg_macro, weak_teacher_seg_micro = log_sedeval_metrics(
            self.val_buffer_teacher_weak, ground_truth, save_dir,
            self.current_epoch,
        )
        """

        # synth dataset
        ground_truth = pd.read_csv(self.hparams["data"]["synth_val_tsv"], sep="\t")
        save_dir = os.path.join(self.logger.log_dir, "metrics_synth_val")
        os.makedirs(save_dir, exist_ok=True)

        (
            synth_student_event_macro,
            synth_student_event_micro,
            synth_student_seg_macro,
            synth_student_seg_micro,
        ) = log_sedeval_metrics(
            self.val_buffer_student_synth, ground_truth, save_dir, self.current_epoch,
        )
        (
            synth_teacher_event_macro,
            synth_teacher_event_micro,
            synth_teacher_seg_macro,
            synth_teacher_seg_micro,
        ) = log_sedeval_metrics(
            self.val_buffer_teacher_eval, ground_truth, save_dir, self.current_epoch,
        )

        # pub eval dataset
        ground_truth = pd.read_csv(self.hparams["data"]["pub_eval_tsv"], sep="\t")
        save_dir = os.path.join(self.logger.log_dir, "metrics_pub_eval")
        os.makedirs(save_dir, exist_ok=True)

        (
            eval_student_event_macro,
            eval_student_event_micro,
            eval_student_seg_macro,
            eval_student_seg_micro,
        ) = log_sedeval_metrics(
            self.val_buffer_student_eval, ground_truth, save_dir, self.current_epoch,
        )
        (
            eval_teacher_event_macro,
            eval_teacher_event_micro,
            eval_teacher_seg_macro,
            eval_teacher_seg_micro,
        ) = log_sedeval_metrics(
            self.val_buffer_teacher_eval, ground_truth, save_dir, self.current_epoch,
        )

        obj_function = torch.tensor(
            -max(eval_student_event_macro, eval_teacher_event_macro)
        )  # we want to maximize f1 event based score.

        tqdm_dict = {
            "val_loss_student_eval": loss_strong_student_on_eval,
            "obj_metric": obj_function,
        }

        tensorboard_logs = {
            "val/weak/student/loss_weak": loss_weak_student_on_weak,
            "val/weak/teacher/loss_weak": loss_weak_teacher_on_weak,
            "val/synth/student/loss_strong": loss_strong_student_on_synth,
            "val/synth/teacher/loss_strong": loss_strong_teacher_on_synth,
            "val/eval/student/loss_strong": loss_strong_student_on_eval,
            "val/eval/teacher/loss_strong": loss_strong_teacher_on_eval,
            "val/eval/student/event_macro_F1": eval_student_event_macro,
            "val/eval/teacher/event_macro_F1": eval_teacher_event_macro,
            "val/synth/student/event_macro_F1": synth_student_event_macro,
            "val/synth/teacher/event_macro_F1": synth_teacher_event_macro,
        }

        output = OrderedDict(
            {
                "obj_metric": obj_function,
                "progress_bar": tqdm_dict,
                "log": tensorboard_logs,
            }
        )

        # free the buffers
        self.val_buffer_student_synth = pd.DataFrame()
        self.val_buffer_teacher_synth = pd.DataFrame()

        self.val_buffer_student_weak = pd.DataFrame()
        self.val_buffer_teacher_weak = pd.DataFrame()

        self.val_buffer_student_eval = pd.DataFrame()
        self.val_buffer_teacher_eval = pd.DataFrame()

        self.log("hp_metric", obj_function)  # log tensorboard hyperpar metric

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
