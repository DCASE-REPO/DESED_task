"""
many clusters have slower data IO, so we cache absolute paths in a json file
for unlabelled audio clips.
"""

import glob
import os
import json


def parse_files2json(folder, json_file, regex="/**/*.wav"):
    files = glob.glob(os.path.join(folder, regex), recursive=True)
    with open(json_file, "w") as f:
        json.dump(files, f, indent=4)
