import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sed_scores_eval
import torch
import torchmetrics
from codecarbon import OfflineEmissionsTracker
from sed_scores_eval.base_modules.scores import (create_score_dataframe,
                                                 validate_score_dataframe)
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from desed_task.data_augm import mixup
from desed_task.utils.postprocess import ClassWiseMedianFilter
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1, compute_psds_from_operating_points,
    compute_psds_from_scores)
from desed_task.utils.scaler import TorchScaler

from .classes_dict import (classes_labels_desed, classes_labels_maestro_real,
                           classes_labels_maestro_real_eval)
from .utils import batched_decode_preds, log_sedeval_metrics


class SEDTask4(pl.LightningModule):
    """Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed_student: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: BaseScheduler subclass object, the scheduler to be used.
                   This is used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        pretrained_model,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,
        sed_teacher=None,
    ):
        super(SEDTask4, self).__init__()
        self.hparams.update(hparams)

        self.encoder = encoder
        self.sed_student = sed_student
        self.median_filter = ClassWiseMedianFilter(self.hparams["net"]["median_filter"])


        if self.hparams["pretrained"]["e2e"]:
            self.pretrained_model = pretrained_model
        # else we use pre-computed embeddings from hdf5

        if sed_teacher is None:
            self.sed_teacher = deepcopy(sed_student)
        else:
            self.sed_teacher = sed_teacher
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

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

        for param in self.sed_teacher.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = (
            torchmetrics.classification.f_beta.MultilabelF1Score(
                len(self.encoder.labels),
                average="macro",
            )
        )
        self.get_weak_teacher_f1_seg_macro = (
            torchmetrics.classification.f_beta.MultilabelF1Score(
                len(self.encoder.labels), average="macro"
            )
        )

        self.scaler = self._init_scaler()
        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_sed_scores_eval_student = {}
        self.val_buffer_sed_scores_eval_teacher = {}

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_buffer_psds_eval_student = {
            k: pd.DataFrame() for k in test_thresholds
        }
        self.test_buffer_psds_eval_teacher = {
            k: pd.DataFrame() for k in test_thresholds
        }
        self.test_buffer_sed_scores_eval_student = {}
        self.test_buffer_sed_scores_eval_teacher = {}
        self.test_buffer_sed_scores_eval_unprocessed_student = {}
        self.test_buffer_sed_scores_eval_unprocessed_teacher = {}
        self.test_buffer_detections_thres05_student = pd.DataFrame()
        self.test_buffer_detections_thres05_teacher = pd.DataFrame()

    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
            except Exception as e:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def on_train_start(self) -> None:
        os.makedirs(os.path.join(self.exp_dir, "codecarbon"), exist_ok=True)
        self.tracker_train = OfflineEmissionsTracker(
            "DCASE Task 4 SED TRAINING",
            output_dir=os.path.join(self.exp_dir, "codecarbon"),
            output_file="emissions_baseline_training.csv",
            log_level="warning",
            country_iso_code="FRA",
            gpu_ids=[torch.cuda.current_device()],
        )
        self.tracker_train.start()

        # Remove for debugging. Those warnings can be ignored during training otherwise.
        # to_ignore = []
        to_ignore = [
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
            ".*invalid value encountered in divide*",
            ".*mean of empty slice*",
            ".*self.log*",
        ]
        for message in to_ignore:
            warnings.filterwarnings("ignore", message)

    def update_ema(self, alpha, global_step, model, ema_model):
        """Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        """Scaler inizialization

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
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
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
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
        """Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        # clamp to reproduce old code
        return amp_to_db(mels).clamp(min=-50, max=80)

    def detect(self, mel_feats, model, embeddings=None, **kwargs):
        if embeddings is None:
            return model(self.scaler(self.take_log(mel_feats)), **kwargs)
        else:
            return model(
                self.scaler(self.take_log(mel_feats)), embeddings=embeddings, **kwargs
            )

    def apply_mixup(self, features, embeddings, labels, start_indx, stop_indx):
        # made a dedicated method as we need to apply mixup only
        # within each dataset that has the same classes
        mixup_type = self.hparams["training"].get("mixup")
        batch_num = features.shape[0]
        current_mask = torch.zeros(batch_num).to(features).bool()
        current_mask[start_indx:stop_indx] = 1
        features[current_mask], labels[current_mask] = mixup(
            features[current_mask], labels[current_mask], mixup_label_type=mixup_type
        )

        if embeddings is not None:
            # apply mixup also on embeddings
            embeddings[current_mask], labels[current_mask] = mixup(
                embeddings[current_mask],
                labels[current_mask],
                mixup_label_type=mixup_type,
            )

        return features, embeddings, labels

    def _unpack_batch(self, batch):

        if not self.hparams["pretrained"]["e2e"]:
            return batch
        else:
            # untested
            raise NotImplementedError
            # we train e2e
            if len(batch) > 3:
                audio, labels, padded_indxs, ast_feats = batch
                pretrained_input = ast_feats
            else:
                audio, labels, padded_indxs = batch
                pretrained_input = audio

    def training_step(self, batch, batch_indx):
        """Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        audio, labels, padded_indxs, embeddings, valid_class_mask = self._unpack_batch(
            batch
        )

        features = self.mel_spec(audio)

        indx_maestro, indx_synth, indx_strong, indx_weak, indx_unlabelled = np.cumsum(
            self.hparams["training"]["batch_size"]
        )

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        mask_unlabeled = torch.zeros(batch_num).to(features).bool()
        strong_mask[:indx_strong] = 1
        weak_mask[indx_strong:indx_weak] = 1
        mask_unlabeled[indx_maestro:] = 1

        # deriving weak labels
        mixup_type = self.hparams["training"].get("mixup")
        if (
            mixup_type is not None
            and self.hparams["training"]["mixup_prob"] > random.random()
        ):
            # NOTE: mix only within same dataset !
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, indx_strong, indx_weak
            )
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, indx_maestro, indx_strong
            )
            features, embeddings, labels = self.apply_mixup(
                features, embeddings, labels, 0, indx_maestro
            )

        # mask labels for invalid datasets classes after mixup.
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        labels = labels.masked_fill(
            ~valid_class_mask[:, :, None].expand_as(labels), 0.0
        )
        labels_weak = labels_weak.masked_fill(~valid_class_mask[weak_mask], 0.0)

        # sed student forward
        strong_preds_student, weak_preds_student = self.detect(
            features,
            self.sed_student,
            embeddings=embeddings,
            classes_mask=valid_class_mask,
        )

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds_student[strong_mask],
            labels[strong_mask],
        )
        # supervised loss on weakly labelled

        loss_weak = self.supervised_loss(
            weak_preds_student[weak_mask],
            labels_weak,
        )
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak

        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.detect(
                features,
                self.sed_teacher,
                embeddings=embeddings,
                classes_mask=valid_class_mask,
            )

        weight = (
            self.hparams["training"]["const_max"]
            * self.scheduler["scheduler"]._get_scaling_factor()
        ) if self.current_epoch < self.hparams["training"]["epoch_decay"] else self.hparams["training"]["const_max"]
        # should we apply the valid mask for classes also here ?

        strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student[mask_unlabeled],
            strong_preds_teacher.detach()[mask_unlabeled],
        )
        weak_self_sup_loss = self.selfsup_loss(
            weak_preds_student[mask_unlabeled],
            weak_preds_teacher.detach()[mask_unlabeled],
        )
        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight

        tot_loss = tot_loss_supervised + tot_self_loss

        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
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
        """Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames, embeddings, valid_class_mask = (
            self._unpack_batch(batch)
        )

        if self.hparams["pretrained"]["e2e"]:
            # extract embeddings here
            if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
                # check that is freezed
                self.pretrained_model.eval()
            embeddings = self.pretrained_model(embeddings)[
                self.hparams["net"]["embedding_type"]
            ]

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(
            mels, self.sed_student, embeddings=embeddings, classes_mask=valid_class_mask
        )
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(
            mels, self.sed_teacher, embeddings=embeddings, classes_mask=valid_class_mask
        )

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
        mask_strong = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    in [
                        str(Path(self.hparams["data"]["synth_val_folder"])),
                        str(Path(self.hparams["data"]["real_maestro_train_folder"])),
                    ]
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
                weak_preds_student[mask_weak], labels_weak.long()
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak.long()
            )

        if torch.any(mask_strong):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_strong], labels[mask_strong]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_strong], labels[mask_strong]
            )

            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)

            filenames_strong = [
                x
                for x in filenames
                if str(Path(x).parent)
                in [
                    str(Path(self.hparams["data"]["synth_val_folder"])),
                    str(Path(self.hparams["data"]["real_maestro_train_folder"])),
                ]
            ]

            (
                scores_unprocessed_student_strong,
                scores_postprocessed_student_strong,
                decoded_student_strong,
            ) = batched_decode_preds(
                strong_preds_student[mask_strong],
                filenames_strong,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_sed_scores_eval_student.update(
                scores_postprocessed_student_strong
            )

            (
                scores_unprocessed_teacher_strong,
                scores_postprocessed_teacher_strong,
                decoded_teacher_strong,
            ) = batched_decode_preds(
                strong_preds_teacher[mask_strong],
                filenames_strong,
                self.encoder,
                median_filter=self.median_filter,
                thresholds=[],
            )

            self.val_buffer_sed_scores_eval_teacher.update(
                scores_postprocessed_teacher_strong
            )

        return

    def validation_epoch_end(self, outputs):
        """Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """
        # desed weak dataset
        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()
        # desed synth dataset
        desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
            self.hparams["data"]["synth_val_tsv"]
        )
        desed_audio_durations = sed_scores_eval.io.read_audio_durations(
            self.hparams["data"]["synth_val_dur"]
        )

        # drop audios without events
        desed_ground_truth = {
            audio_id: gt for audio_id, gt in desed_ground_truth.items() if len(gt) > 0
        }
        desed_audio_durations = {
            audio_id: desed_audio_durations[audio_id]
            for audio_id in desed_ground_truth.keys()
        }
        keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }
        psds1_sed_scores_eval_student = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        intersection_f1_macro_thres05_student_sed_scores_eval = (
            sed_scores_eval.intersection_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )[0]["macro_average"]
        )
        collar_f1_macro_thres05_student_sed_scores_eval = (
            sed_scores_eval.collar_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                onset_collar=0.2,
                offset_collar=0.2,
                offset_collar_rate=0.2,
            )[0]["macro_average"]
        )
        desed_scores = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in desed_ground_truth.keys()
        }
        psds1_sed_scores_eval_teacher = compute_psds_from_scores(
            desed_scores,
            desed_ground_truth,
            desed_audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
        intersection_f1_macro_thres05_teacher_sed_scores_eval = (
            sed_scores_eval.intersection_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                dtc_threshold=0.5,
                gtc_threshold=0.5,
            )[0]["macro_average"]
        )
        collar_f1_macro_thres05_teacher_sed_scores_eval = (
            sed_scores_eval.collar_based.fscore(
                desed_scores,
                desed_ground_truth,
                threshold=0.5,
                onset_collar=0.2,
                offset_collar=0.2,
                offset_collar_rate=0.2,
            )[0]["macro_average"]
        )

        # maestro
        maestro_ground_truth = pd.read_csv(
            self.hparams["data"]["real_maestro_train_tsv"], sep="\t"
        )
        maestro_ground_truth = maestro_ground_truth[
            maestro_ground_truth.confidence > 0.5
        ]
        maestro_ground_truth = maestro_ground_truth[
            maestro_ground_truth.event_label.isin(classes_labels_maestro_real_eval)
        ]
        maestro_ground_truth = {
            clip_id: events
            for clip_id, events in sed_scores_eval.io.read_ground_truth_events(
                maestro_ground_truth
            ).items()
            if clip_id in self.val_buffer_sed_scores_eval_student
        }
        maestro_ground_truth = _merge_overlapping_events(maestro_ground_truth)
        maestro_audio_durations = {
            clip_id: sorted(events, key=lambda x: x[1])[-1][1]
            for clip_id, events in maestro_ground_truth.items()
        }
        event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
        keys = ["onset", "offset"] + event_classes_maestro_eval
        maestro_scores_student = {
            clip_id: self.val_buffer_sed_scores_eval_student[clip_id][keys]
            for clip_id in maestro_ground_truth.keys()
        }
        segment_f1_macro_optthres_student = sed_scores_eval.segment_based.best_fscore(
            maestro_scores_student,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["macro_average"]
        segment_mauc_student = sed_scores_eval.segment_based.auroc(
            maestro_scores_student,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["mean"]
        segment_mpauc_student = sed_scores_eval.segment_based.auroc(
            maestro_scores_student,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
            max_fpr=0.1,
        )[0]["mean"]
        maestro_scores_teacher = {
            clip_id: self.val_buffer_sed_scores_eval_teacher[clip_id][keys]
            for clip_id in maestro_ground_truth.keys()
        }
        segment_f1_macro_optthres_teacher = sed_scores_eval.segment_based.best_fscore(
            maestro_scores_teacher,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["macro_average"]
        segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
            maestro_scores_teacher,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
        )[0]["mean"]
        segment_mpauc_teacher = sed_scores_eval.segment_based.auroc(
            maestro_scores_teacher,
            maestro_ground_truth,
            maestro_audio_durations,
            segment_length=1.0,
            max_fpr=0.1,
        )[0]["mean"]

        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = psds1_sed_scores_eval_student
        elif obj_metric_synth_type == "collar":
            synth_metric = collar_f1_macro_thres05_student_sed_scores_eval
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_thres05_student_sed_scores_eval
        elif obj_metric_synth_type == "psds":
            synth_metric = psds1_sed_scores_eval_student
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        obj_metric_maestro_type = self.hparams["training"].get(
            "obj_metric_maestro_type"
        )
        if obj_metric_maestro_type is None:
            maestro_metric = segment_mpauc_student
        elif obj_metric_maestro_type == "fmo":
            maestro_metric = segment_f1_macro_optthres_student
        elif obj_metric_maestro_type == "mauc":
            maestro_metric = segment_mauc_student
        elif obj_metric_maestro_type == "mpauc":
            maestro_metric = segment_f1_macro_optthres_student
        else:
            raise NotImplementedError(
                f"obj_metric_maestro_type: {obj_metric_maestro_type} not implemented."
            )

        obj_metric = torch.tensor(
            weak_student_f1_macro.item() + synth_metric + maestro_metric
        )

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log(
            "val/student/weak_f1_macro_thres05/torchmetrics", weak_student_f1_macro
        )
        self.log(
            "val/teacher/weak_f1_macro_thres05/torchmetrics", weak_teacher_f1_macro
        )
        self.log(
            "val/student/intersection_f1_macro_thres05/sed_scores_eval",
            intersection_f1_macro_thres05_student_sed_scores_eval,
        )
        self.log(
            "val/teacher/intersection_f1_macro_thres05/sed_scores_eval",
            intersection_f1_macro_thres05_teacher_sed_scores_eval,
        )
        self.log(
            "val/student/collar_f1_macro_thres05/sed_scores_eval",
            collar_f1_macro_thres05_student_sed_scores_eval,
        )
        self.log(
            "val/teacher/collar_f1_macro_thres05/sed_scores_eval",
            collar_f1_macro_thres05_teacher_sed_scores_eval,
        )
        self.log("val/student/psds1/sed_scores_eval", psds1_sed_scores_eval_student)
        self.log("val/teacher/psds1/sed_scores_eval", psds1_sed_scores_eval_teacher)
        self.log(
            "val/student/segment_f1_macro_thresopt/sed_scores_eval",
            segment_f1_macro_optthres_student,
        )
        self.log("val/student/segment_mauc/sed_scores_eval", segment_mauc_student)
        self.log("val/student/segment_mpauc/sed_scores_eval", segment_mpauc_student)
        self.log(
            "val/teacher/segment_f1_macro_thresopt/sed_scores_eval",
            segment_f1_macro_optthres_teacher,
        )
        self.log("val/teacher/segment_mauc/sed_scores_eval", segment_mauc_teacher)
        self.log("val/teacher/segment_mpauc/sed_scores_eval", segment_mpauc_teacher)

        # free the buffers
        self.val_buffer_sed_scores_eval_student = {}
        self.val_buffer_sed_scores_eval_teacher = {}

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames, embeddings, valid_class_mask = (
            self._unpack_batch(batch)
        )

        if self.hparams["pretrained"]["e2e"]:
            # extract embeddings here
            if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
                # check that is freezed
                self.pretrained_model.eval()
            embeddings = self.pretrained_model(embeddings)[
                self.hparams["net"]["embedding_type"]
            ]

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(
            mels, self.sed_student, embeddings
        )
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(
            mels, self.sed_teacher, embeddings
        )

        if not self.evaluation:
            loss_strong_student = self.supervised_loss(strong_preds_student, labels)
            loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

            self.log("test/student/loss_strong", loss_strong_student)
            self.log("test/teacher/loss_strong", loss_strong_teacher)

        # compute psds
        (
            scores_unprocessed_student_strong,
            scores_postprocessed_student_strong,
            decoded_student_strong,
        ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_student.keys()) + [0.5],
        )

        self.test_buffer_sed_scores_eval_unprocessed_student.update(
            scores_unprocessed_student_strong
        )
        self.test_buffer_sed_scores_eval_student.update(
            scores_postprocessed_student_strong
        )
        for th in self.test_buffer_psds_eval_student.keys():
            self.test_buffer_psds_eval_student[th] = pd.concat(
                [self.test_buffer_psds_eval_student[th], decoded_student_strong[th]],
                ignore_index=True,
            )

        (
            scores_unprocessed_teacher_strong,
            scores_postprocessed_teacher_strong,
            decoded_teacher_strong,
        ) = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.median_filter,
            thresholds=list(self.test_buffer_psds_eval_teacher.keys()) + [0.5],
        )

        self.test_buffer_sed_scores_eval_unprocessed_teacher.update(
            scores_unprocessed_teacher_strong
        )
        self.test_buffer_sed_scores_eval_teacher.update(
            scores_postprocessed_teacher_strong
        )
        for th in self.test_buffer_psds_eval_teacher.keys():
            self.test_buffer_psds_eval_teacher[th] = pd.concat(
                [self.test_buffer_psds_eval_teacher[th], decoded_teacher_strong[th]],
                ignore_index=True,
            )

        # compute f1 score
        self.test_buffer_detections_thres05_student = pd.concat(
            [self.test_buffer_detections_thres05_student, decoded_student_strong[0.5]]
        )
        self.test_buffer_detections_thres05_teacher = pd.concat(
            [self.test_buffer_detections_thres05_teacher, decoded_teacher_strong[0.5]]
        )

    def on_test_epoch_end(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")
        print("save_dir", save_dir)
        results = {}
        if self.evaluation:
            # only save prediction scores
            save_dir_student_unprocessed = os.path.join(
                save_dir, "student_scores", "unprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_unprocessed_student,
                save_dir_student_unprocessed,
            )
            print(f"\nRaw scores for student saved in: {save_dir_student_unprocessed}")

            save_dir_student_postprocessed = os.path.join(
                save_dir, "student_scores", "postprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_student,
                save_dir_student_postprocessed,
            )
            print(
                f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}"
            )

            save_dir_teacher_unprocessed = os.path.join(
                save_dir, "teacher_scores", "unprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_unprocessed_teacher,
                save_dir_teacher_unprocessed,
            )
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_unprocessed}")

            save_dir_teacher_postprocessed = os.path.join(
                save_dir, "teacher_scores", "postprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_buffer_sed_scores_eval_teacher,
                save_dir_teacher_postprocessed,
            )
            print(
                f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}"
            )

            self.tracker_eval.stop()
        else:
            # calculate the metrics
            # psds_eval
            psds1_student_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds2_student_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            psds1_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds2_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_buffer_psds_eval_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            # synth dataset
            intersection_f1_macro_thres05_student_psds_eval = (
                compute_per_intersection_macro_f1(
                    {"0.5": self.test_buffer_detections_thres05_student},
                    self.hparams["data"]["test_tsv"],
                    self.hparams["data"]["test_dur"],
                )
            )
            intersection_f1_macro_thres05_teacher_psds_eval = (
                compute_per_intersection_macro_f1(
                    {"0.5": self.test_buffer_detections_thres05_teacher},
                    self.hparams["data"]["test_tsv"],
                    self.hparams["data"]["test_dur"],
                )
            )
            # sed_eval
            collar_f1_macro_thres05_student = log_sedeval_metrics(
                self.test_buffer_detections_thres05_student,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]
            collar_f1_macro_thres05_teacher = log_sedeval_metrics(
                self.test_buffer_detections_thres05_teacher,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # sed_scores_eval
            desed_ground_truth = sed_scores_eval.io.read_ground_truth_events(
                self.hparams["data"]["test_tsv"]
            )
            desed_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["test_dur"]
            )

            # drop audios without events
            desed_ground_truth = {
                audio_id: gt
                for audio_id, gt in desed_ground_truth.items()
                if len(gt) > 0
            }
            desed_audio_durations = {
                audio_id: desed_audio_durations[audio_id]
                for audio_id in desed_ground_truth.keys()
            }
            keys = ["onset", "offset"] + sorted(classes_labels_desed.keys())
            desed_scores = {
                clip_id: self.test_buffer_sed_scores_eval_student[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }
            psds1_student_sed_scores_eval = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds2_student_sed_scores_eval = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            intersection_f1_macro_thres05_student_sed_scores_eval = (
                sed_scores_eval.intersection_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    dtc_threshold=0.5,
                    gtc_threshold=0.5,
                )[0]["macro_average"]
            )
            collar_f1_macro_thres05_student_sed_scores_eval = (
                sed_scores_eval.collar_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    onset_collar=0.2,
                    offset_collar=0.2,
                    offset_collar_rate=0.2,
                )[0]["macro_average"]
            )

            desed_scores = {
                clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id][keys]
                for clip_id in desed_ground_truth.keys()
            }
            psds1_teacher_sed_scores_eval = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds2_teacher_sed_scores_eval = compute_psds_from_scores(
                desed_scores,
                desed_ground_truth,
                desed_audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            intersection_f1_macro_thres05_teacher_sed_scores_eval = (
                sed_scores_eval.intersection_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    dtc_threshold=0.5,
                    gtc_threshold=0.5,
                )[0]["macro_average"]
            )
            collar_f1_macro_thres05_teacher_sed_scores_eval = (
                sed_scores_eval.collar_based.fscore(
                    desed_scores,
                    desed_ground_truth,
                    threshold=0.5,
                    onset_collar=0.2,
                    offset_collar=0.2,
                    offset_collar_rate=0.2,
                )[0]["macro_average"]
            )

            maestro_audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["real_maestro_val_dur"]
            )
            maestro_ground_truth_clips = pd.read_csv(
                self.hparams["data"]["real_maestro_val_tsv"], sep="\t"
            )
            maestro_clip_ids = [filename[:-4] for filename in maestro_ground_truth_clips["filename"]]
            maestro_ground_truth_clips = maestro_ground_truth_clips[
                maestro_ground_truth_clips.confidence > 0.5
            ]
            maestro_ground_truth_clips = maestro_ground_truth_clips[
                maestro_ground_truth_clips.event_label.isin(
                    classes_labels_maestro_real_eval
                )
            ]
            maestro_ground_truth_clips = sed_scores_eval.io.read_ground_truth_events(
                maestro_ground_truth_clips
            )

            maestro_ground_truth = _merge_maestro_ground_truth(
                maestro_ground_truth_clips
            )
            maestro_audio_durations = {
                file_id: maestro_audio_durations[file_id]
                for file_id in maestro_ground_truth.keys()
            }

            maestro_scores_student = {
                clip_id: self.test_buffer_sed_scores_eval_student[clip_id]
                for clip_id in maestro_clip_ids
            }
            maestro_scores_teacher = {
                clip_id: self.test_buffer_sed_scores_eval_teacher[clip_id]
                for clip_id in maestro_clip_ids
            }
            segment_length = 1.0
            event_classes_maestro = sorted(classes_labels_maestro_real)
            segment_scores_student = _get_segment_scores_and_overlap_add(
                frame_scores=maestro_scores_student,
                audio_durations=maestro_audio_durations,
                event_classes=event_classes_maestro,
                segment_length=segment_length,
            )
            sed_scores_eval.io.write_sed_scores(
                segment_scores_student,
                os.path.join(save_dir, "student", "maestro", "postprocessed"),
            )
            segment_scores_teacher = _get_segment_scores_and_overlap_add(
                frame_scores=maestro_scores_teacher,
                audio_durations=maestro_audio_durations,
                event_classes=event_classes_maestro,
                segment_length=segment_length,
            )
            sed_scores_eval.io.write_sed_scores(
                segment_scores_teacher,
                os.path.join(save_dir, "teacher", "maestro", "postprocessed"),
            )

            event_classes_maestro_eval = sorted(classes_labels_maestro_real_eval)
            keys = ["onset", "offset"] + event_classes_maestro_eval
            segment_scores_student = {
                clip_id: scores_df[keys]
                for clip_id, scores_df in segment_scores_student.items()
            }
            segment_scores_teacher = {
                clip_id: scores_df[keys]
                for clip_id, scores_df in segment_scores_teacher.items()
            }

            segment_f1_macro_optthres_student = (
                sed_scores_eval.segment_based.best_fscore(
                    segment_scores_student,
                    maestro_ground_truth,
                    maestro_audio_durations,
                    segment_length=segment_length,
                )[0]["macro_average"]
            )
            segment_mauc_student = sed_scores_eval.segment_based.auroc(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["mean"]
            segment_mpauc_student = sed_scores_eval.segment_based.auroc(
                segment_scores_student,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
                max_fpr=0.1,
            )[0]["mean"]
            segment_f1_macro_optthres_teacher = (
                sed_scores_eval.segment_based.best_fscore(
                    segment_scores_teacher,
                    maestro_ground_truth,
                    maestro_audio_durations,
                    segment_length=segment_length,
                )[0]["macro_average"]
            )
            segment_mauc_teacher = sed_scores_eval.segment_based.auroc(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
            )[0]["mean"]
            segment_mpauc_teacher = sed_scores_eval.segment_based.auroc(
                segment_scores_teacher,
                maestro_ground_truth,
                maestro_audio_durations,
                segment_length=segment_length,
                max_fpr=0.1,
            )[0]["mean"]

            results.update({
                "test/student/psds1/psds_eval": psds1_student_psds_eval,
                "test/student/psds2/psds_eval": psds2_student_psds_eval,
                "test/teacher/psds1/psds_eval": psds1_teacher_psds_eval,
                "test/teacher/psds2/psds_eval": psds2_teacher_psds_eval,
                "test/student/intersection_f1_macro_thres05/psds_eval": intersection_f1_macro_thres05_student_psds_eval,
                "test/teacher/intersection_f1_macro_thres05/psds_eval": intersection_f1_macro_thres05_teacher_psds_eval,
                "test/student/collar_f1_macro_thres05/sed_eval": collar_f1_macro_thres05_student,
                "test/teacher/collar_f1_macro_thres05/sed_eval": collar_f1_macro_thres05_teacher,
                "test/student/psds1/sed_scores_eval": psds1_student_sed_scores_eval,
                "test/student/psds2/sed_scores_eval": psds2_student_sed_scores_eval,
                "test/teacher/psds1/sed_scores_eval": psds1_teacher_sed_scores_eval,
                "test/teacher/psds2/sed_scores_eval": psds2_teacher_sed_scores_eval,
                "test/student/intersection_f1_macro_thres05/sed_scores_eval": intersection_f1_macro_thres05_student_sed_scores_eval,
                "test/teacher/intersection_f1_macro_thres05/sed_scores_eval": intersection_f1_macro_thres05_teacher_sed_scores_eval,
                "test/student/collar_f1_macro_thres05/sed_scores_eval": collar_f1_macro_thres05_student_sed_scores_eval,
                "test/teacher/collar_f1_macro_thres05/sed_scores_eval": collar_f1_macro_thres05_teacher_sed_scores_eval,
                "test/student/segment_f1_macro_thresopt/sed_scores_eval": segment_f1_macro_optthres_student,
                "test/student/segment_mauc/sed_scores_eval": segment_mauc_student,
                "test/student/segment_mpauc/sed_scores_eval": segment_mpauc_student,
                "test/teacher/segment_f1_macro_thresopt/sed_scores_eval": segment_f1_macro_optthres_teacher,
                "test/teacher/segment_mauc/sed_scores_eval": segment_mauc_teacher,
                "test/teacher/segment_mpauc/sed_scores_eval": segment_mpauc_teacher,
            })
            self.tracker_devtest.stop()

        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=True)

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader

    def on_train_end(self) -> None:
        # dump consumption
        self.tracker_train.stop()
        training_kwh = self.tracker_train._total_energy.kWh
        self.logger.log_metrics(
            {"/train/tot_energy_kWh": torch.tensor(float(training_kwh))}
        )

    def on_test_start(self) -> None:
        if self.evaluation:
            os.makedirs(os.path.join(self.exp_dir, "codecarbon"), exist_ok=True)
            self.tracker_eval = OfflineEmissionsTracker(
                "DCASE Task 4 SED EVALUATION",
                output_dir=os.path.join(self.exp_dir, "codecarbon"),
                output_file="emissions_basleline_eval.csv",
                log_level="warning",
                country_iso_code="FRA",
                gpu_ids=[torch.cuda.current_device()],
            )
            self.tracker_eval.start()
        else:
            os.makedirs(os.path.join(self.exp_dir, "codecarbon"), exist_ok=True)
            self.tracker_devtest = OfflineEmissionsTracker(
                "DCASE Task 4 SED DEVTEST",
                output_dir=os.path.join(self.exp_dir, "codecarbon"),
                output_file="emissions_baseline_test.csv",
                log_level="warning",
                country_iso_code="FRA",
                gpu_ids=[torch.cuda.current_device()],
            )

            self.tracker_devtest.start()


def _merge_maestro_ground_truth(clip_ground_truth):
    ground_truth = defaultdict(list)
    for clip_id in clip_ground_truth:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit("-", maxsplit=2)
        clip_onset_time = int(clip_onset_time) // 100
        ground_truth[file_id].extend(
            [
                (
                    clip_onset_time + event_onset_time,
                    clip_onset_time + event_offset_time,
                    event_class,
                )
                for event_onset_time, event_offset_time, event_class in clip_ground_truth[
                    clip_id
                ]
            ]
        )
    return _merge_overlapping_events(ground_truth)


def _merge_overlapping_events(ground_truth_events):
    for clip_id, events in ground_truth_events.items():
        per_class_events = defaultdict(list)
        for event in events:
            per_class_events[event[2]].append(event)
        ground_truth_events[clip_id] = []
        for event_class, events in per_class_events.items():
            events = sorted(events)
            merged_events = []
            current_offset = -1e6
            for event in events:
                if event[0] > current_offset:
                    merged_events.append(list(event))
                else:
                    merged_events[-1][1] = max(current_offset, event[1])
                current_offset = merged_events[-1][1]
            ground_truth_events[clip_id].extend(merged_events)
    return ground_truth_events


def _get_segment_scores_and_overlap_add(
    frame_scores, audio_durations, event_classes, segment_length=1.0
):
    """
    >>> event_classes = ['a', 'b', 'c']
    >>> audio_durations = {'f1': 201.6, 'f2':133.1, 'f3':326}
    >>> frame_scores = {\
        f'{file_id}-{int(100*onset)}-{int(100*(onset+10.))}': create_score_dataframe(np.random.rand(156,3), np.arange(157.)*0.064, event_classes)\
        for file_id in audio_durations for onset in range(int((audio_durations[file_id]-9.)))\
    }
    >>> frame_scores.keys()
    >>> seg_scores = _get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.)
    >>> [(key, validate_score_dataframe(value)[0][-3:]) for key, value in seg_scores.items()]
    """
    segment_scores_file = {}
    summand_count = {}
    keys = ["onset", "offset"] + event_classes
    for clip_id in frame_scores:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit("-", maxsplit=2)
        clip_onset_time = float(clip_onset_time) / 100
        clip_offset_time = float(clip_offset_time) / 100
        if file_id not in segment_scores_file:
            segment_scores_file[file_id] = np.zeros(
                (ceil(audio_durations[file_id] / segment_length), len(event_classes))
            )
            summand_count[file_id] = np.zeros_like(segment_scores_file[file_id])
        segment_scores_clip = _get_segment_scores(
            frame_scores[clip_id][keys],
            clip_length=(clip_offset_time - clip_onset_time),
            segment_length=1.0,
        )[event_classes].to_numpy()
        seg_idx = int(clip_onset_time // segment_length)
        segment_scores_file[file_id][
            seg_idx : seg_idx + len(segment_scores_clip)
        ] += segment_scores_clip
        summand_count[file_id][seg_idx : seg_idx + len(segment_scores_clip)] += 1
    return {
        file_id: create_score_dataframe(
            segment_scores_file[file_id] / np.maximum(summand_count[file_id], 1),
            np.minimum(
                np.arange(
                    0.0, audio_durations[file_id] + segment_length, segment_length
                ),
                audio_durations[file_id],
            ),
            event_classes,
        )
        for file_id in segment_scores_file
    }


def _get_segment_scores(scores_df, clip_length, segment_length=1.0):
    """
    >>> scores_arr = np.random.rand(156,3)
    >>> timestamps = np.arange(157)*0.064
    >>> event_classes = ['a', 'b', 'c']
    >>> scores_df = create_score_dataframe(scores_arr, timestamps, event_classes)
    >>> seg_scores_df = _get_segment_scores(scores_df, clip_length=10., segment_length=1.)
    """
    frame_timestamps, event_classes = validate_score_dataframe(scores_df)
    scores_arr = scores_df[event_classes].to_numpy()
    segment_scores = []
    segment_timestamps = []
    seg_onset_idx = 0
    seg_offset_idx = 0
    for seg_onset in np.arange(0.0, clip_length, segment_length):
        seg_offset = seg_onset + segment_length
        while frame_timestamps[seg_onset_idx + 1] <= seg_onset:
            seg_onset_idx += 1
        while (
            seg_offset_idx < len(scores_arr)
            and frame_timestamps[seg_offset_idx] < seg_offset
        ):
            seg_offset_idx += 1
        seg_weights = np.minimum(
            frame_timestamps[seg_onset_idx + 1 : seg_offset_idx + 1], seg_offset
        ) - np.maximum(frame_timestamps[seg_onset_idx:seg_offset_idx], seg_onset)
        segment_scores.append(
            (seg_weights[:, None] * scores_arr[seg_onset_idx:seg_offset_idx]).sum(0)
            / seg_weights.sum()
        )
        segment_timestamps.append(seg_onset)
    segment_timestamps.append(clip_length)
    return create_score_dataframe(
        np.array(segment_scores), np.array(segment_timestamps), event_classes
    )
