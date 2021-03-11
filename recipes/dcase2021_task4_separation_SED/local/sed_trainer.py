import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from desed_task.data_augm import add_noise
from desed_task.utils.scaler import TorchScaler
import numpy as np

from .utils import (
    batched_decode_preds,
    compute_pdsd_macro_f1,
    compute_psds_from_operating_points,
    log_sedeval_metrics,
)


class SEDTask4_2021(pl.LightningModule):
    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        sed_teacher,
        opt,
        train_data,
        valid_data,
        test_data,
        train_sampler,
        scheduler,
    ):
        super(SEDTask4_2021, self).__init__()
        self.hparams = hparams

        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = sed_teacher
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        # self.feats = Fbanks(**self.hparams["feats"], log=False)

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

        self.scaler = self._init_scaler()

        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}

        self.test_eventF1_buffer_student = pd.DataFrame()
        self.test_eventF1_buffer_teacher = pd.DataFrame()

    def update_ema(self, alpha, global_step, model, ema_model):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):

        if self.hparams["scaler"]["statistic"] == "instance":
            self.scaler = TorchScaler(
                "instance", "minmax", self.hparams["scaler"]["dims"]
            )
            return self.scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader, transform_func=lambda x: self.take_log(self.mel_spec(x[0]))
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):

        amp_to_db = AmplitudeToDB(stype='amplitude')
        return amp_to_db(mels)

    def training_step(self, batch, batch_indx):

        audio, labels, padded_indxs = batch
        indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        features = self.mel_spec(audio)

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()

        # sed student forward
        strong_preds_student, weak_preds_student = self.sed_student(
            self.scaler(
                self.take_log(
                    add_noise(features, self.hparams["training"]["noise_snr"])
                )
            )
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
            ema_features = self.scaler(
                self.take_log(
                    add_noise(features, self.hparams["training"]["noise_snr"])
                )
            )
            strong_preds_teacher, weak_preds_teacher = self.sed_teacher(
                self.scaler(ema_features)
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

        audio, labels, padded_indxs, filenames = batch

        # prediction for student
        logmels = self.scaler(self.take_log(self.mel_spec(audio)))
        strong_preds_student, weak_preds_student = self.sed_student(logmels)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.sed_teacher(logmels)

        # we derive masks for each dataset based on folders of filenames
        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
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
            .to(audio)
            .bool()
        )

        mask_devtest = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["test_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
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

            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

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

            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)

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
                thresholds=list(self.val_buffer_student_synth.keys()),
            )

            for th in self.val_buffer_student_synth.keys():
                self.val_buffer_student_synth[th] = self.val_buffer_student_synth[
                    th
                ].append(decoded_student_strong[th])

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_teacher_synth.keys()),
            )
            for th in self.val_buffer_teacher_synth.keys():
                self.val_buffer_teacher_synth[th] = self.val_buffer_teacher_synth[
                    th
                ].append(decoded_teacher_strong[th])

        if torch.any(mask_devtest):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_devtest], labels[mask_devtest]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_devtest], labels[mask_devtest]
            )

            self.log("val/test/student/loss_strong", loss_strong_student)
            self.log("val/test/teacher/loss_strong", loss_strong_teacher)

            filenames_test = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["test_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_devtest],
                filenames_test,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_student_test.keys()),
            )

            for th in self.val_buffer_student_test.keys():
                self.val_buffer_student_test[th] = self.val_buffer_student_test[
                    th
                ].append(decoded_student_strong[th])

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_devtest],
                filenames_test,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_teacher_test.keys()),
            )
            for th in self.val_buffer_teacher_test.keys():
                self.val_buffer_teacher_test[th] = self.val_buffer_teacher_test[
                    th
                ].append(decoded_teacher_strong[th])

        return

    def validation_epoch_end(self, outputs):

        weak_student_seg_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_seg_macro = self.get_weak_teacher_f1_seg_macro.compute()

        # synth dataset
        psds_f1_macro_student = compute_pdsd_macro_f1(
            self.val_buffer_student_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["test_dur"],
        )

        synth_student_event_macro = log_sedeval_metrics(
            self.val_buffer_student_synth[0.5], self.hparams["data"]["synth_val_tsv"],
        )[0]

        psds_f1_macro_teacher = compute_pdsd_macro_f1(
            self.val_buffer_teacher_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["test_dur"],
        )

        synth_teacher_event_macro = log_sedeval_metrics(
            self.val_buffer_teacher_synth[0.5], self.hparams["data"]["synth_val_tsv"],
        )[0]

        # dev-test dataset
        psds_f1_macro_student_test = compute_pdsd_macro_f1(
            self.val_buffer_student_test,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
        )

        test_student_event_macro = log_sedeval_metrics(
            self.val_buffer_student_test[0.5], self.hparams["data"]["test_tsv"],
        )[0]

        psds_f1_macro_teacher_test = compute_pdsd_macro_f1(
            self.val_buffer_teacher_test,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
        )

        test_teacher_event_macro = log_sedeval_metrics(
            self.val_buffer_teacher_test[0.5], self.hparams["data"]["test_tsv"],
        )[0]

        obj_metric = torch.tensor(
            # -max(
            #     weak_student_seg_macro.item() + psds_f1_macro_student,
            #     weak_teacher_seg_macro.item() + psds_f1_macro_teacher,
            # )
            -(weak_student_seg_macro.item() + synth_student_event_macro)
        )

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/segment_macro_F1", weak_student_seg_macro)
        self.log("val/weak/teacher/segment_macro_F1", weak_teacher_seg_macro)
        self.log("val/synth/student/psds_f1_macro", psds_f1_macro_student)
        self.log("val/synth/teacher/psds_f1_macro", psds_f1_macro_teacher)
        self.log("val/synth/student/event_f1_macro", synth_student_event_macro)
        self.log("val/synth/teacher/event_f1_macro", synth_teacher_event_macro)
        self.log("val/test/student/psds_f1_macro", psds_f1_macro_student_test)
        self.log("val/test/student/event_f1_macro", test_student_event_macro)
        self.log("val/test/teacher/psds_f1_macro", psds_f1_macro_teacher_test)
        self.log("val/test/teacher/event_f1_macro", test_teacher_event_macro)

        # free the buffers
        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):

        audio, labels, padded_indxs, filenames = batch

        # prediction for student
        logmels = self.scaler(self.take_log(self.mel_spec(audio)))
        strong_preds_student, weak_preds_student = self.sed_student(logmels)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.sed_teacher(logmels)

        loss_strong_student = self.supervised_loss(strong_preds_student, labels)
        loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

        self.log("test/student/loss_strong", loss_strong_student)
        self.log("test/teacher/loss_strong", loss_strong_teacher)

        # compute psds
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()),
        )

        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = self.test_psds_buffer_student[
                th
            ].append(decoded_student_strong[th])

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()),
        )

        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = self.test_psds_buffer_teacher[
                th
            ].append(decoded_teacher_strong[th])

        # compute f1 score
        decoded_student_strong = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )

        self.test_eventF1_buffer_student = self.test_eventF1_buffer_student.append(
            decoded_student_strong[0.5]
        )

        decoded_teacher_strong = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=[0.5],
        )

        self.test_eventF1_buffer_teacher = self.test_eventF1_buffer_teacher.append(
            decoded_teacher_strong[0.5]
        )

    def on_test_epoch_end(self):

        # pub eval dataset
        save_dir = os.path.join(self.logger.log_dir, "metrics_test")

        (
            psds_score,
            psds_ct_score,
            psds_macro_score,
        ) = compute_psds_from_operating_points(
            self.test_psds_buffer_student,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            os.path.join(save_dir, "student"),
        )

        (
            psds_score_teacher,
            psds_ct_score_teacher,
            psds_macro_score_teacher,
        ) = compute_psds_from_operating_points(
            self.test_psds_buffer_teacher,
            self.hparams["data"]["test_tsv"],
            self.hparams["data"]["test_dur"],
            os.path.join(save_dir, "teacher"),
        )

        event_macro_student = log_sedeval_metrics(
            self.test_eventF1_buffer_student,
            self.hparams["data"]["test_tsv"],
            os.path.join(save_dir, "student"),
        )[0]

        event_macro_teacher = log_sedeval_metrics(
            self.test_eventF1_buffer_teacher,
            self.hparams["data"]["test_tsv"],
            os.path.join(save_dir, "teacher"),
        )[0]

        best_test_result = torch.tensor(-max(psds_score, psds_score_teacher))

        self.log("hp_metric", best_test_result)  # log tensorboard hyperpar metric
        self.log("test/student/psds_score", psds_score)
        self.log("test/student/psds_ct_score", psds_ct_score)
        self.log("test/student/psds_macro_score", psds_macro_score)
        self.log("test/teacher/psds_score", psds_score_teacher)
        self.log("test/teacher/psds_ct_score", psds_ct_score_teacher)
        self.log("test/teacher/psds_macro_score", psds_macro_score_teacher)
        self.log("test/student/event_f1_macro", event_macro_student)
        self.log("test/teacher/event_f1_macro", event_macro_teacher)

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

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.hparams["training"]["num_workers"],
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader
