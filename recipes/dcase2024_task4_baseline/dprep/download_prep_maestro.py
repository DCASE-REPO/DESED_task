import glob
import os.path
import zipfile
from pathlib import Path

import desed
import intervaltree
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def count_chunks(inlen, chunk_size, chunk_stride):
    return int((inlen - chunk_size + chunk_stride) / chunk_stride)


def get_chunks_indx(in_len, chunk_size, chunk_stride, discard_last=False):
    i = -1
    for i in range(count_chunks(in_len, chunk_size, chunk_stride)):
        yield i * chunk_stride, i * chunk_stride + chunk_size
    if not discard_last and i * chunk_stride + chunk_size < in_len:
        if in_len - (i + 1) * chunk_stride > 0:
            yield (i + 1) * chunk_stride, in_len


def read_maestro_annotation(annotation_f):
    annotation = []
    with open(annotation_f, "r") as f:
        lines = f.readlines()

    for l in lines:
        if Path(annotation_f).suffix != ".csv":
            start, stop, event, confidence = l.rstrip("\n").split("\t")
            annotation.append(
                {
                    "onset": float(start),
                    "offset": float(stop),
                    "event_label": event,
                    "confidence": float(confidence),
                }
            )
        else:
            start, stop, event = l.rstrip("\n").split("\t")
            if start == "onset":
                continue
            annotation.append(
                {
                    "onset": float(start),
                    "offset": float(stop),
                    "event_label": event,
                    "confidence": 1.0,
                }
            )
    return annotation


def ann2intervaltree(annotation):
    tree = intervaltree.IntervalTree()
    for elem in annotation:
        tree.add(intervaltree.Interval(elem["onset"], elem["offset"], elem))

    return tree


def get_current_annotation(annotation, start, end):
    # use intervaltree here !
    overlapping = []
    for ann in annotation:
        if ann.overlaps(start, end):
            c_segment = ann.data
            # make it relative
            onset = max(0.0, c_segment["onset"] - start)
            offset = min(end - start, c_segment["offset"] - start)
            c_segment = {
                "onset": onset,
                "offset": offset,
                "event_label": c_segment["event_label"],
                "confidence": c_segment["confidence"],
            }
            overlapping.append(c_segment)

    overlapping = sorted(overlapping, key=lambda x: x["onset"])
    return overlapping


def split_maestro_single_file(output_audio_folder, audiofile, annotation, window_len=10, hop_len=1):
    audio, fs = sf.read(audiofile)
    annotation = read_maestro_annotation(annotation)
    annotation = ann2intervaltree(annotation)
    new_annotation = []
    for st, end in get_chunks_indx(len(audio), int(window_len * fs), int(hop_len * fs)):
        c_seg = audio[st:end]
        c_annotation = get_current_annotation(annotation, st / fs, end / fs)

        # save
        start = st / fs * 100
        end = end / fs * 100
        filename = Path(audiofile).stem + f"-{int(start):06d}-{int(end):06d}"
        sf.write(os.path.join(output_audio_folder, filename + ".wav"), c_seg, fs)
        for line in c_annotation:
            new_annotation.append(
                {
                    "filename": filename + Path(audiofile).suffix,
                    "onset": line["onset"],
                    "offset": line["offset"],
                    "event_label": line["event_label"],
                    "confidence": line["confidence"],
                }
            )  # tsv like desed

    return new_annotation


def split_maestro_real(download_folder, out_audio_folder, out_meta_folder):
    audiofiles = glob.glob(
        os.path.join(download_folder, "development_audio", "**/*.wav"), recursive=True
    )
    annotation_files = glob.glob(
        os.path.join(download_folder, "development_annotation", "**/*.txt"),
        recursive=True,
    )

    assert len(audiofiles) == len(annotation_files)
    assert len(audiofiles) > 0, (
        "You probably have the wrong folder as input to this script."
        f"Check {download_folder}, does it contain MAESTRO dev data ? "
        f"It must have development_annotation and development_audio as sub-folders."
    )
    for split in ["train", "validation"]:
        split_info = os.path.join(os.path.dirname(__file__), f"{split}_split.csv")
        if split == "validation":
            split_info = pd.read_csv(split_info)["val"]
            hop_len = 5
        else:
            split_info = pd.read_csv(split_info)[f"{split}"]
            hop_len = 1
        split_info = set([Path(x).stem for x in split_info])
        # filter audiofiles here now and annotation
        c_audiofiles = [x for x in audiofiles if Path(x).stem in split_info]
        c_annotation_files = [x for x in annotation_files if Path(x).stem in split_info]

        # get corresponding annotation files.
        c_audiofiles = sorted(c_audiofiles, key=lambda x: Path(x).stem)
        c_audiofiles = {Path(x).stem: x for x in c_audiofiles}
        # get all metadata
        c_annotation_files = {Path(x).stem: x for x in c_annotation_files}

        Path(os.path.join(out_audio_folder, f"maestro_real_{split}")).mkdir(
            parents=True, exist_ok=True
        )
        Path(out_meta_folder).mkdir(parents=True, exist_ok=True)

        # split here
        all_annotations = []
        for k in tqdm(c_audiofiles.keys()):
            c_path = c_audiofiles[k]
            c_metadata_f = c_annotation_files[k]
            c_annotations = split_maestro_single_file(
                os.path.join(out_audio_folder, f"maestro_real_{split}"),
                c_path,
                c_metadata_f,
                window_len=10,
                hop_len=hop_len,
            )
            all_annotations.extend(c_annotations)

        all_annotations = pd.DataFrame.from_dict(all_annotations)
        all_annotations = all_annotations.sort_values(by="filename", ascending=True)
        all_annotations.to_csv(
            os.path.join(out_meta_folder, f"maestro_real_{split}.tsv"),
            sep="\t",
            index=False,
        )
    (Path(download_folder) / "development_metadata.csv").rename(
        Path(out_meta_folder) / "maestro_real_durations.tsv"
    )


def split_maestro_synth(download_folder, out_audio_folder, out_meta_folder):
    audiofiles = glob.glob(
        os.path.join(download_folder, "audio", "**/*.wav"), recursive=True
    )
    annotation_files = glob.glob(
        os.path.join(download_folder, "estimated_strong_labels", "**/*.csv"),
        recursive=True,
    )

    assert len(audiofiles) == len(annotation_files)
    assert len(audiofiles) > 0, (
        "You probably have the wrong folder as input to this script."
        f"Check {download_folder}, does it contain MAESTRO dev data ? "
        f"It must have development_annotation and development_audio as sub-folders."
    )

    c_audiofiles = audiofiles
    c_annotation_files = annotation_files
    split = "train"
    # get corresponding annotation files.
    c_audiofiles = sorted(c_audiofiles, key=lambda x: Path(x).stem)
    c_audiofiles = {Path(x).stem: x for x in c_audiofiles}
    # get all metadata
    c_annotation_files = {Path(x).stem: x for x in c_annotation_files}

    Path(os.path.join(out_audio_folder, f"maestro_synth_{split}")).mkdir(
        parents=True, exist_ok=True
    )
    Path(out_meta_folder).mkdir(parents=True, exist_ok=True)

    # split here
    all_annotations = []
    for k in tqdm(c_audiofiles.keys()):
        c_path = c_audiofiles[k]
        c_metadata_f = c_annotation_files["mturk_" + k]
        c_annotations = split_maestro_single_file(
            os.path.join(out_audio_folder, f"maestro_synth_{split}"),
            c_path,
            c_metadata_f,
        )
        all_annotations.extend(c_annotations)

    all_annotations = pd.DataFrame.from_dict(all_annotations)
    all_annotations = all_annotations.sort_values(by="filename", ascending=True)
    all_annotations.to_csv(
        os.path.join(out_meta_folder, f"maestro_synth_{split}.tsv"),
        sep="\t",
        index=False,
    )


def download_and_prepare_maestro(dcase_dataset_folder):
    url_synth_audio = "https://zenodo.org/records/5126478/files/audio.zip?download=1"
    url_synth_meta = "https://zenodo.org/records/5126478/files/meta.zip?download=1"

    synth_label_metadata_path = os.path.join(dcase_dataset_folder, "maestro_synth")

    def help_extract(main_dir, url_name, name):
        Path(main_dir).mkdir(parents=True, exist_ok=True)
        desed.utils.download_file_from_url(url_name, os.path.join(main_dir, name))
        with zipfile.ZipFile(os.path.join(main_dir, name), "r") as zip_ref:
            zip_ref.extractall(main_dir)

    help_extract(synth_label_metadata_path, url_synth_meta, "meta.zip")
    synth_audio_path = os.path.join(dcase_dataset_folder, "maestro_synth")
    help_extract(synth_audio_path, url_synth_audio, "audio.zip")

    url_dev_audio = (
        "https://zenodo.org/records/7244360/files/development_audio.zip?download=1"
    )
    dev_audio_path = os.path.join(dcase_dataset_folder, "maestro_dev")
    help_extract(dev_audio_path, url_dev_audio, "development_audio.zip")
    url_dev_meta = (
        "https://zenodo.org/records/7244360/files/development_annotation.zip?download=1"
    )
    dev_meta_path = os.path.join(dcase_dataset_folder, "maestro_dev")
    help_extract(dev_meta_path, url_dev_meta, "development_audio.zip")
    url_dev_audio_durations = "https://raw.githubusercontent.com/marmoi/dcase2023_task4b_baseline/main/metadata/development_metadata.csv"
    desed.utils.download_file_from_url(
        url_dev_audio_durations, os.path.join(dev_meta_path, "development_metadata.csv")
    )


def get_maestro(dcase_dataset_folder):
    download_and_prepare_maestro(os.path.join(dcase_dataset_folder, "MAESTRO_original"))
    print(
        "Preparing MAESTRO real development and training sets."
        "Splitting it into 10s chunks."
    )

    split_maestro_real(
        os.path.join(dcase_dataset_folder, "MAESTRO_original", "maestro_dev"),
        os.path.join(dcase_dataset_folder, "audio"),
        os.path.join(dcase_dataset_folder, "metadata"),
    )

    print("Preparing MAESTRO synth training set." "Splitting it into 10s chunks.")

    split_maestro_synth(
        os.path.join(dcase_dataset_folder, "MAESTRO_original", "maestro_synth"),
        os.path.join(dcase_dataset_folder, "audio"),
        os.path.join(dcase_dataset_folder, "metadata"),
    )


if __name__ == "__main__":
    split_maestro_real(
        "/media/samco/Data1/MAESTRO/maestro_dev/",
        "/media/samco/Data1/MAESTRO_split/audio",
        "/media/samco/Data1/MAESTRO_split/metadata",
    )
