import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from latch.executions import rename_current_execution
from latch.functions.messages import message
from latch.types.directory import LatchDir, LatchOutputDir
from latch.types.file import LatchFile

from wf.utils import privileged_largedisk_g6e_8xlarge_task

sys.stdout.reconfigure(line_buffering=True)


class AcceleratorType(Enum):
    GPU = "gpu"
    CPU = "cpu"


class OutputFormat(Enum):
    PDB = "pdb"
    MMCIF = "mmcif"


class MSAPairingStrategy(Enum):
    complete = "complete"
    greedy = "greedy"


@privileged_largedisk_g6e_8xlarge_task(cache=True)
def boltz_task(
    run_name: str,
    input_mode: str,
    msa_mode: str,
    msa_directory: Optional[LatchDir],
    input_file: Optional[LatchFile],
    input_directory: Optional[LatchDir],
    output_directory: LatchOutputDir,
    cache_dir: Optional[str],
    checkpoint: Optional[str],
    affinity_checkpoint: Optional[str],
    devices: int,
    accelerator: str,
    num_workers: int,
    preprocessing_threads: int,
    recycling_steps: int,
    sampling_steps: int,
    step_scale: float,
    diffusion_samples: int,
    max_parallel_samples: Optional[int],
    sampling_steps_affinity: int,
    diffusion_samples_affinity: int,
    use_msa_server: bool,
    msa_server_url: Optional[str],
    msa_pairing_strategy: Optional[str],
    max_msa_seqs: int,
    subsample_msa: bool,
    num_subsampled_msa: int,
    write_full_pae: bool,
    write_full_pde: bool,
    output_format: str,
    override: bool,
    seed: Optional[int],
    model: str,
    method: Optional[str],
    use_potentials: bool,
    affinity_mw_correction: bool,
    no_kernels: bool,
) -> LatchOutputDir:
    """ """
    rename_current_execution(str(run_name))

    print("-" * 60)
    print("Creating local directories")
    local_output_dir = Path("/root/outputs") / run_name
    local_output_dir.mkdir(parents=True, exist_ok=True)
    if msa_directory:
        local_msa_dir = Path("/root/msa")
        msa_dir_path = Path(msa_directory.local_path)

        msa_dir_path.rename(local_msa_dir)

    print("-" * 60)
    print("Running Boltz-2")

    if input_mode == "single_file":
        input = str(input_file.local_path)
    else:
        input = str(input_directory.local_path)

    command = [
        "boltz",
        "predict",
        str(input),
        "--devices",
        str(devices),
        "--accelerator",
        str(accelerator),
        "--out_dir",
        str(local_output_dir),
        "--cache",
        str(cache_dir),
        "--recycling_steps",
        str(recycling_steps),
        "--sampling_steps",
        str(sampling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--output_format",
        str(output_format),
        "--num_workers",
        str(num_workers),
        "--step_scale",
        str(step_scale),
        "--model",
        str(model),
        "--preprocessing-threads",
        str(preprocessing_threads),
        "--sampling_steps_affinity",
        str(sampling_steps_affinity),
        "--diffusion_samples_affinity",
        str(diffusion_samples_affinity),
        "--max_msa_seqs",
        str(max_msa_seqs),
        "--num_subsampled_msa",
        str(num_subsampled_msa),
    ]

    if subsample_msa:
        command.extend(["--subsample_msa"])

    if override:
        command.extend(["--override"])

    if use_potentials:
        command.extend(["--use_potentials"])

    if affinity_mw_correction:
        command.extend(["--affinity_mw_correction"])

    if no_kernels:
        command.extend(["--no_kernels"])

    if affinity_checkpoint:
        command.extend(["--affinity_checkpoint", str(affinity_checkpoint)])

    if max_parallel_samples:
        command.extend(["--max_parallel_samples", str(max_parallel_samples)])

    if seed:
        command.extend(["--seed", str(seed)])

    if method:
        command.extend(["--method", str(method)])

    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])

    if write_full_pae:
        command.append("--write_full_pae")

    if write_full_pde:
        command.append("--write_full_pde")

    if use_msa_server:
        command.append("--use_msa_server")
        if msa_server_url:
            command.extend(["--msa_server_url", msa_server_url])
        if msa_pairing_strategy:
            command.extend(["--msa_pairing_strategy", msa_pairing_strategy])

    print(f"Running command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
    except Exception as e:
        print("FAILED")
        message("error", {"title": "Boltz-2 failed", "body": f"{e}"})
        sys.exit(1)

    print("-" * 60)
    print("Returning results")
    return LatchOutputDir(str("/root/outputs"), output_directory.remote_path)
