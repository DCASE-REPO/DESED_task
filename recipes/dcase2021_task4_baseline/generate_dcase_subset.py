# Script to generate the different version of the DESED dataset used in the paper for the DCASE Workshop.

import argparse
import glob
import json
import os
import shutil

import desed
import jams
import scaper
import yaml


def create_folder(folder, exist_ok=True, delete_if_exists=False):
    """Create folder (and parent folders) if not exists.

    Args:
        folder (str): path of folder to create.
        exist_ok (bool): if set to True (default), the FileExistsError is not raised.
        delete_if_exists: bool, True if you want to delete the folder if already exists.
    Returns:
        None
    """
    if not folder == "":
        if delete_if_exists:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                os.mkdir(folder)

        os.makedirs(folder, exist_ok=exist_ok)


def jam_initialization(annotations):
    """The function initializes the jam file, setting the annotations given as input

    Args:
        annotations (dict): annotation of jam file

    Returns:
        JAMS: jam file initialized
    """

    # Jams instances
    jam = jams.JAMS()

    # file_metadata
    jam.file_metadata.duration = annotations["duration"]

    return jam


def get_sandbox(annotations, background_path, foreground_path):
    """The function gets the sandbox from the annotation and generate the new sandbox for the jam file to create
    updating the proper data, as the path to the background and foreground soundbank folders.

    Args:
        annotations (dict): annotations of the jam file
        background_path (str): path to background soundbank folder
        foreground_path (str): path to foreground soundbank folder

    Returns:
        dict: sandbox of the jam file
    """
    # getting the sandbox
    sandbox = annotations["sandbox"]
    duration = annotations["duration"]

    # setting the background and foreground folder path
    sandbox["scaper"]["fg_path"] = foreground_path
    sandbox["scaper"]["bg_path"] = background_path

    sandbox_tmp = {"scaper": {}}

    sandbox_meta = {
        "duration": duration,
        "original_duration": duration,
        "fg_path": sandbox["scaper"]["fg_path"],
        "bg_path": sandbox["scaper"]["bg_path"],
        "protected_labels": [],
    }
    sandbox_tmp["scaper"].update(sandbox_meta)

    sandbox_tmp = {"scaper": sandbox["scaper"].copy()}
    del sandbox_tmp["scaper"]["fg_spec"]
    del sandbox_tmp["scaper"]["bg_spec"]

    return sandbox_tmp


def jam_annotation(annotations, background_path, foreground_path):
    """The function sets the annotations for the jam file

    Args:
        annotations (dict): annotations
        background_path (str): path to the background soundbank folder
        foreground_path (str): path to foreground soundbank folder

    Returns:
        dict: annotations of the jam file
    """

    # get the necessary annotations and adding them to the annotations to be added to the jam file
    namespace = annotations["namespace"]
    time = annotations["time"]
    duration = annotations["duration"]

    sandbox = get_sandbox(annotations, background_path, foreground_path)

    ann = jams.Annotation(
        namespace=namespace, sandbox=sandbox, time=time, duration=duration
    )

    return ann


def get_jam_annotations(annotations, background_path, foreground_path):
    """The function initialize a jam file, setting the annotations given as input and return the JAMS file and the annotations

    Args:
        Args:
        annotations (dict): annotations
        background_path (str): path to the background soundbank folder
        foreground_path (str): path to foreground soundbank folder

    Returns:
        JAMS: jam file initialized
        Annotations: jam annotations
    """

    jam = jam_initialization(annotations)
    ann = jam_annotation(annotations, background_path, foreground_path)

    return jam, ann


def generate_audio(jam, anns, background_path, foreground_path, out_file_path):
    """The function generates an audio clip from a jam file given as input

    Args
        jam (JAMS): jam file
        anns (Annotations): annotations of the jam file
        background_path (str): path to background soundbank folder
        foreground_path (str): path to foreeground soundbank folder
        out_file_path (str): path to output file
    Returns:
        None
    """

    jam.annotations.append(anns)
    jam.save(out_file_path)

    # generate the audio clip
    scaper.generate_from_jams(
        out_file_path,
        audio_outfile=out_file_path.replace(".jams", ".wav"),
        fg_path=foreground_path,
        bg_path=background_path,
        jams_outfile=out_file_path,
        save_isolated_events=False,
        txt_path=out_file_path.replace(".jams", ".txt"),
    )


def change_snr(annotations, ann_snr, db_to_decrease, target_labels):
    """The function changes (decreases) the SNR of all the non-target events by db_to_decrease given as input

    Args:
        annotations (Annotations): annotations of the jam file
        ann_snr (Annotations): annotations of the jam file with the new SNR
        db_to_decrease (int): db to decrease for no-target events
        target_labels (list): list of target labels to consider

    Returns:
        None
    """

    number_of_events = len(annotations["data"])

    for event in range(number_of_events):
        tmp_ann = annotations["data"][event]
        if tmp_ann["value"]["role"] == "background":
            ann_snr.append(
                time=tmp_ann["time"],
                duration=tmp_ann["duration"],
                value=tmp_ann["value"],
                confidence=tmp_ann["confidence"],
            )
        elif tmp_ann["value"]["role"] == "foreground":
            if tmp_ann["value"]["label"] in target_labels:
                ann_snr.append(
                    time=tmp_ann["time"],
                    duration=tmp_ann["duration"],
                    value=tmp_ann["value"],
                    confidence=tmp_ann["confidence"],
                )
            else:
                tmp_ann["value"]["snr"] = tmp_ann["value"]["snr"] - db_to_decrease
                ann_snr.append(
                    time=tmp_ann["time"],
                    duration=tmp_ann["duration"],
                    value=tmp_ann["value"],
                    confidence=tmp_ann["confidence"],
                )


def collect_target_events(annotations, ann_target, target_labels):
    """The function collects only the target events present in the annotations and return the number of event collected

    Args:
        annotations (dict): annotations from jam file
        ann_target (dict): only target events annotations
        target_labels (list): list of target events


    Returns:
        int: number of target events present in the jam file
    """

    number_of_events = len(annotations["data"])

    for event in range(number_of_events):
        tmp_ann = annotations["data"][event]
        if tmp_ann["value"]["role"] == "background":
            ann_target.append(
                time=tmp_ann["time"],
                duration=tmp_ann["duration"],
                value=tmp_ann["value"],
                confidence=tmp_ann["confidence"],
            )
        elif tmp_ann["value"]["role"] == "foreground":
            if tmp_ann["value"]["label"] in target_labels:
                ann_target.append(
                    time=tmp_ann["time"],
                    duration=tmp_ann["duration"],
                    value=tmp_ann["value"],
                    confidence=tmp_ann["confidence"],
                )

    return len(ann_target["data"])


def collect_nontarget_events(annotations, ann_ntarget, target_labels):
    """The function collects only the non-target events present in the annotations and return the number
    of non-target events collected

    Args:
        annotations (dict): annotations from jam file
        ann_target (dict): only no-target events annotations
        target_labels (list): list of target events

    Returns:
        int: number of no-target events present in the jam file
    """

    number_of_events = len(annotations["data"])

    for event in range(number_of_events):
        tmp_ann = annotations["data"][event]
        if tmp_ann["value"]["role"] == "background":
            ann_ntarget.append(
                time=tmp_ann["time"],
                duration=tmp_ann["duration"],
                value=tmp_ann["value"],
                confidence=tmp_ann["confidence"],
            )
        elif tmp_ann["value"]["role"] == "foreground":
            if tmp_ann["value"]["label"] not in target_labels:
                ann_ntarget.append(
                    time=tmp_ann["time"],
                    duration=tmp_ann["duration"],
                    value=tmp_ann["value"],
                    confidence=tmp_ann["confidence"],
                )

    return len(ann_ntarget["data"])


class Subset:
    """Subset class modeling a subsets"""

    def __init__(
        self,
        synth_files_folder,
        output_folder_ext,
        background_folder,
        foreground_folder,
        target_labels,
    ):
        """Initialization of subset class

        Args:
            synth_files_folder (str): synthetic data folder path
            output_folder_ext (str): extension to add to the output folder path
            background_folder (str): background soundbank folder path
            foreground_folder (str): foreground soundbank folder path
            target_labels (list): list of target events

        Return:
            None
        """

        self.synth_files_folder = synth_files_folder
        self.jams_files = glob.glob(os.path.join(self.synth_files_folder, "*.jams"))
        self.output_folder = self.synth_files_folder + output_folder_ext
        self.background_folder = background_folder
        self.foreground_folder = foreground_folder
        self.target_labels = target_labels

        create_folder(self.output_folder, exist_ok=True)

    def generate_target_nontarget_files(self, only_target):
        """
        The function generate a version of the subset containing only target or non-target events depending on the only_target flag

        Args:
            only_target (boolean): True if the subset to generate contains only target events, False if the subset to generate contains only non-target events

        Returns:
            None
        """

        self.only_target = only_target

        for file in self.jams_files:
            n_file = (file.split("/"))[-1].split(".")[0]

            with open(file, "r") as f:
                contents = json.loads(f.read())

            annotations = contents["annotations"][0]

            out_file_path = os.path.join(self.output_folder, n_file + ".jams")

            if only_target:
                self.generate_target_nontarget_file_from_jam(annotations, out_file_path)
            else:
                self.generate_target_nontarget_file_from_jam(annotations, out_file_path)

    def generate_target_nontarget_file_from_jam(self, annotations, out_file_path):
        """The function generates audio clip containing only target or non-target events depending on the jam file specification

        Args:
            annotations (Annotations): annotations of the jam file
            out_file_path (str): output file path
        """

        jam, ann = get_jam_annotations(
            annotations, self.background_folder, self.foreground_folder
        )

        if self.only_target:
            collect_target_events(annotations, ann, target_labels)
            generate_audio(
                jam, ann, self.background_folder, self.foreground_folder, out_file_path
            )
        else:
            if collect_nontarget_events(annotations, ann, target_labels) > 1:
                generate_audio(
                    jam,
                    ann,
                    self.background_folder,
                    self.foreground_folder,
                    out_file_path,
                )

    def change_snr_from_jam_file(self, annotations, db_to_decrease, out_file_path):
        """The function initializes the jam file and the annotation and change SNR for the non-targets events

        Args:
            annotations (Annotations): annotations of the file
            db_to_decrease (int): db to decrease from the file
            out_file_path (str): path to the output file

        Return:
            None
        """

        jam, ann = get_jam_annotations(
            annotations, self.background_folder, self.foreground_folder
        )

        change_snr(annotations, ann, db_to_decrease, self.target_labels)

        # generate audio
        generate_audio(
            jam, ann, self.background_folder, self.foreground_folder, out_file_path
        )

    def decrease_snr(
        self,
        db_to_decrease,
    ):
        """The functions process all the files to decrease the SNR of non-target events contained in the audio clips

        Args:
            db_to_decrease (int): db to decrease from the file

        Return:
            None
        """

        for file in self.jams_files:
            n_file = (file.split("/"))[-1].split(".")[0]

            with open(file, "r") as f:
                contents = json.loads(f.read())

            annotations = contents["annotations"][0]

            out_file_path = os.path.join(self.output_folder, n_file + ".jams")

            self.change_snr_from_jam_file(annotations, db_to_decrease, out_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generating synthetic audio files")
    parser.add_argument(
        "--conf_file",
        default="./confs/sed_dataset.yaml",
        help="The configuration file with all the experiment parameters.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="""Generation of the target versions of ths train, validation and evaluation dataset, 
        non-target events version of the evaluation set and SNR versions of the train, 
        validation subset of the DESED dataset.""",
    )

    parser.add_argument(
        "--tg",
        action="store_true",
        default=False,
        help="""Generation of the target versions of ths train, validation and/or evaluation dataset. 
        If only one dataset need to be generated, this should be set on the configuration file. 
        """,
    )

    parser.add_argument(
        "--ntg",
        action="store_true",
        default=False,
        help="""Generation of the non-target versions of the train, validation and/or evaluation dataset. 
        If only one dataset need to be generated, this should be set on the configuration file. 
        """,
    )

    parser.add_argument(
        "--snr",
        action="store_true",
        default=False,
        help="""Generation of the different SNR versions of the dataset. 
        If only one dataset need to be generated, this should be set on the configuration file. 
        dB to decreased can be set on the configuration file. 
        """,
    )

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    # get the path to the differnt set of data
    path_to_folders = configs["data"]

    # get the parameters
    snr = configs["params"]["snr"]
    target_set = configs["params"]["target_set"]
    nontarget_set = configs["params"]["nontarget_set"]
    snr_set = configs["params"]["snr_set"]

    target_labels = configs["params"]["target_labels"]

    ##############################
    ##### generation of data #####
    ##############################

    if args.all:
        args.tg = True
        args.ntg = True
        args.snr = True

    if args.tg:
        for split in target_set:
            folder_ext = "_target"

            subset = Subset(
                configs["data"][f"synth_{split}"],
                folder_ext,
                configs["data"][f"background_{split}"],
                configs["data"][f"foreground_{split}"],
                target_labels,
            )

            print(f"Generating subset {split}, only target files.")

            # generate the only-target subset
            subset.generate_target_nontarget_files(only_target=True)
            print(f"Target {split} subset generated.\n")

    if args.ntg:
        for split in nontarget_set:
            folder_ext = "_nontarget"

            subset = Subset(
                configs["data"][f"synth_{split}"],
                folder_ext,
                configs["data"][f"background_{split}"],
                configs["data"][f"foreground_{split}"],
                target_labels,
            )

            print(f"Generating subset {split}, only non-target files.")

            # generate the non-target subset
            subset.generate_target_nontarget_files(only_target=False)
            print(f"Non target {split} subset generated.\n")

    if args.snr:
        for db_to_decrease in snr:
            for split in snr_set:
                folder_ext = "_" + str(db_to_decrease) + "SNR"

                subset = Subset(
                    configs["data"][f"synth_{split}"],
                    folder_ext,
                    configs["data"][f"background_{split}"],
                    configs["data"][f"foreground_{split}"],
                    target_labels,
                )

                print(f"Generating subset {split}, SNR {db_to_decrease}.")

                # generate the different SNR versions of the subset
                subset.decrease_snr(db_to_decrease)
                print(f"Subset generated for SNR {db_to_decrease} for {split}.\n")
