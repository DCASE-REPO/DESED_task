from setuptools import setup

setup(
    name="desed_task",
    version="0.1.0",
    description="Sound Event Detection and Separation in Domestic Environments.",
    author="DCASE2021 Task 4 Organizers",
    author_email="cornellsamuele@gmail.com",
    license="MIT",
    packages=["desed_task"],
    python_requires=">=3.8",
    install_requires=[
        "asteroid==0.4.1",
        "dcase_util==0.2.16",
        "psds_eval==0.3.0",
        "sed_eval==0.2.1",
    ],
)
