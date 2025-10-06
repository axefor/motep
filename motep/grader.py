"""`motep grade` command."""

import argparse
import pathlib
import pprint

from ase import Atoms
from mpi4py import MPI

import motep.io
from motep.active import AlgorithmBase, make_algorithm
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.potentials.mtp.data import MTPData
from motep.setting import GradeSetting, load_setting_grade


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


class Grader:
    """Grader class for calculating grade."""

    def __init__(
        self,
        images_training: list[Atoms],
        mtp_data: MTPData,
        setting: GradeSetting,
        comm: MPI.Comm,
    ):
        self.setting = setting
        algorithm_class = make_algorithm(self.setting.algorithm)
        self.optimality: AlgorithmBase = algorithm_class(
            images_training,
            mtp_data,
            setting.engine,
            rng=setting.rng,
        )
        self.comm = comm

    def update(self, new_images: list[Atoms]) -> None:
        """Update the optimality."""
        self.optimality.update(new_images)

    def grade(self, images_in: list) -> None:
        """Calculate grades for images."""
        if self.comm.rank == 0:
            print(f"{'':=^72s}\n")
            print("[data_active]")
            print(self.optimality.indices)
            print(flush=True)

        self.optimality.calc_grade(images_in)
        return [_.info["MV_grade"] for _ in images_in]


def grade(filename_setting: str, comm: MPI.Comm) -> None:
    """Grade.

    This adds `MV_grade` to `atoms.info`.
    """
    rank = comm.Get_rank()
    setting = load_setting_grade(filename_setting)
    if rank == 0:
        pprint.pp(setting)
        print(flush=True)

    mtp_file = str(pathlib.Path(setting.potential_final).expanduser().resolve())

    species = setting.species or None
    images_training = read_images(
        setting.data_training,
        species=species,
        comm=comm,
        title="data_training",
    )
    if not setting.species:
        species = get_dummy_species(images_training)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade`"
        raise ValueError(msg)

    grader = Grader(images_training, mtp_data, setting, comm)
    images_in = read_images(
        setting.data_in,
        species=species,
        comm=comm,
        title="data_in",
    )
    grader.grade(images_in)

    motep.io.write(setting.data_out[0], images_in)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    grade(args.setting, comm)
