from setuptools import setup

setup(
    name="desed_task",
    version="0.1.1",
    description="Sound Event Detection and Separation in Domestic Environments.",
    author="DCASE2023 Task 4 Organizers",
    license="MIT",
    packages=["desed_task"],
    python_requires=">=3.8",
    install_requires=[
        "dcase_util>=0.2.16",
        "psds_eval>=0.4.0",
        "sed_eval>=0.2.1",
        "sed_scores_eval>=0.0.0",
    ],
)
