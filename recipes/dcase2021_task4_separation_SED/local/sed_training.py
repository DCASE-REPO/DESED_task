import os
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from desed_task.data_augm import add_noise
from desed_task.features import Fbanks


from .utils import batched_decode_preds, log_sedeval_metrics


class SEDTask4_2021(pl.LightningModule):
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
        super(SEDTask4_2021, self).__init__()
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

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = pl.metrics.classification.F1(
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        self.get_weak_teacher_f1_seg_macro = pl.metrics.classification.F1(
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_student_synth = pd.DataFrame()
        self.val_buffer_teacher_synth = pd.DataFrame()

        self.val_buffer_student_eval = pd.DataFrame()
        self.val_buffer_teacher_eval = pd.DataFrame()

    def update_ema(self, alpha, global_step, model, ema_model):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

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

        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/step", self.scheduler["scheduler"].step_num, prog_bar=True)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/weight", weight)
        self.log("train/student/tot_supervised", strong_self_sup_loss, prog_bar=True)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)

        return tot_loss

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
                    == str(Path(self.hparams["data"]["weak_folder"]))
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

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()
            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )

            self.log("loss_weak_student_on_weak", loss_weak_student)
            self.log("loss_weak_teacher_on_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[mask_weak], labels_weak
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            self.log("loss_strong_student_on_synth", loss_strong_student)
            self.log("loss_strong_teacher_on_synth", loss_strong_teacher)

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["synth_val_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
            )
            self.val_buffer_student_synth = self.val_buffer_student_synth.append(
                decoded_student_strong
            )

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
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

            self.log("loss_strong_student_on_eval", loss_strong_student)
            self.log("loss_strong_teacher_on_eval", loss_strong_teacher)

            filenames_eval = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["pub_eval_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_eval],
                filenames_eval,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
            )
            self.val_buffer_student_eval = self.val_buffer_student_eval.append(
                decoded_student_strong
            )

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_eval],
                filenames_eval,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
            )
            self.val_buffer_teacher_eval = self.val_buffer_teacher_eval.append(
                decoded_teacher_strong
            )

        return

    def validation_epoch_end(self, outputs):

        loss_weak_student_on_weak = self.trainer.callback_metrics.get(
            "loss_weak_student_on_weak"
        )
        loss_weak_teacher_on_weak = self.trainer.callback_metrics.get(
            "loss_weak_teacher_on_weak"
        )
        loss_strong_student_on_synth = self.trainer.callback_metrics.get(
            "loss_strong_student_on_synth"
        )
        loss_strong_teacher_on_synth = self.trainer.callback_metrics.get(
            "loss_strong_teacher_on_synth"
        )
        loss_strong_student_on_eval = self.trainer.callback_metrics.get(
            "loss_strong_student_on_eval"
        )
        loss_strong_teacher_on_eval = self.trainer.callback_metrics.get(
            "loss_strong_teacher_on_eval"
        )

        weak_student_seg_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_seg_macro = self.get_weak_teacher_f1_seg_macro.compute()

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
            -max(
                weak_student_seg_macro.item() + synth_student_event_macro,
                weak_teacher_seg_macro.item() + synth_teacher_event_macro,
            )
        )

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
            "val/weak/student/segment_macro_F1": weak_student_seg_macro,
            "val/weak/teacher/segment_macro_F1": weak_teacher_seg_macro,
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

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

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
