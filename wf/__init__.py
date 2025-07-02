from typing import Optional

from latch.resources.launch_plan import LaunchPlan
from latch.resources.workflow import workflow
from latch.types.directory import LatchDir, LatchOutputDir
from latch.types.file import LatchFile
from latch.types.metadata import (
    Fork,
    ForkBranch,
    LatchAuthor,
    LatchMetadata,
    LatchParameter,
    LatchRule,
    Params,
    Section,
    Spoiler,
    Text,
)
from latch.types.metadata import (
    Multiselect,
    MultiselectOption,
)

from wf.task import AcceleratorType, MSAPairingStrategy, OutputFormat, boltz_task

# UI layout for Boltz workflow

flow = [
    # ─────────────────────────────────────────────
    # 1. Prediction input
    # ─────────────────────────────────────────────
    Section(
        "Prediction Input",
        Text("Select your input mode for Boltz prediction."),
        Fork(
            "input_mode",
            "Choose input type",
            single_file=ForkBranch(
                "Single File",
                Text("Use a single FASTA or YAML file for prediction."),
                Params("input_file"),
            ),
            directory=ForkBranch(
                "Directory",
                Text("Use a directory containing multiple FASTA/YAML files."),
                Params("input_directory"),
            ),
        ),
    ),
    # ─────────────────────────────────────────────
    # 2. MSA input
    # ─────────────────────────────────────────────
    Section(
        "MSA Input",
        Fork(
            "msa_mode",
            "Choose MSA method",
            directory=ForkBranch(
                "Directory",
                Text(
                    "Provide a directory of MSA files.  The workflow mounts "
                    "this at `/root/msa/`."
                ),
                Params("msa_directory"),
            ),
            server=ForkBranch(
                "Server",
                Text("Use the MMSeqs2 server for on-the-fly MSA generation."),
                Params(
                    "use_msa_server",
                    "msa_server_url",
                    "msa_pairing_strategy",
                ),
            ),
        ),
    ),
    # ─────────────────────────────────────────────
    # 3. Output location & run name
    # ─────────────────────────────────────────────
    Section(
        "Output",
        Params("run_name"),
        Text("Directory where all results will be written"),
        Params("output_directory"),
    ),
    # ─────────────────────────────────────────────
    # 4. Advanced settings (collapsible)
    # ─────────────────────────────────────────────
    Section(
        "Advanced Settings",
        # 4-a  Inference / sampling knobs
        Spoiler(
            "Run Options",
            Params(
                "recycling_steps",
                "sampling_steps",
                "diffusion_samples",
                "max_parallel_samples",
                "step_scale",
                "sampling_steps_affinity",
                "diffusion_samples_affinity",
            ),
        ),
        # 4-b  Extra MSA controls
        Spoiler(
            "MSA Options",
            Text("Fine-tune MSA handling and subsampling"),
            Params(
                "max_msa_seqs",
                "subsample_msa",
                "num_subsampled_msa",
            ),
        ),
        # 4-c  Compute / hardware controls
        Spoiler(
            "Compute Options",
            Text("Configure hardware, workers, and threads"),
            Params(
                "devices",
                "num_workers",
                "accelerator",
                "preprocessing_threads",
            ),
        ),
        # 4-d  Output-file toggles
        Spoiler(
            "Output Options",
            Text("Choose formats and optional artefacts to save"),
            Params(
                "output_format",
                "write_full_pae",
                "write_full_pde",
                "override",
                "seed",
            ),
        ),
        # 4-e  Paths, checkpoints, and expert flags
        Spoiler(
            "System Configuration",
            Text("Advanced paths, checkpoints, and model tweaks"),
            Params(
                "cache_dir",
                "checkpoint",
                "affinity_checkpoint",
                "model",
                "method",
                "use_potentials",
                "affinity_mw_correction",
                "no_kernels",
            ),
        ),
    ),
]

metadata = LatchMetadata(
    display_name="Boltz-2",
    author=LatchAuthor(
        name="Jeremy Wohlwend et. al.",
    ),
    repository="https://github.com/latchbio-workflows/wf-latchbio-boltz",
    license="MIT",
    tags=["Protein Engineering"],
    parameters={
        # ─────────────────────────────────────────────
        # General / bookkeeping
        # ─────────────────────────────────────────────
        "run_name": LatchParameter(
            display_name="Run Name",
            description="Name of this Boltz prediction run",
            batch_table_column=True,
            rules=[
                LatchRule(
                    regex=r"^[a-zA-Z0-9_-]+$",
                    message="Run name may contain letters, numbers, dashes, and underscores only.",
                )
            ],
        ),
        # ─────────────────────────────────────────────
        # I/O paths
        # ─────────────────────────────────────────────
        "input_mode": LatchParameter(),  # your code decides whether the user chooses file vs directory
        "input_file": LatchParameter(
            display_name="Input File",
            description="Single FASTA or YAML file for prediction",
            batch_table_column=True,
        ),
        "input_directory": LatchParameter(
            display_name="Input Directory",
            description="Directory containing FASTA / YAML files",
            batch_table_column=True,
        ),
        "output_directory": LatchParameter(
            display_name="Output Directory",
            description="Folder where prediction results will be written (--out_dir)",
            batch_table_column=True,
        ),
        "msa_directory": LatchParameter(
            display_name="MSA Directory",
            description="Optional directory containing pre-computed MSA files",
        ),
        "cache_dir": LatchParameter(
            display_name="Cache Directory",
            description="Where to download model weights and data (--cache)",
        ),
        "checkpoint": LatchParameter(
            display_name="Checkpoint Path",
            description="Optional model checkpoint to load (--checkpoint)",
        ),
        "affinity_checkpoint": LatchParameter(
            display_name="Affinity Checkpoint",
            description="Optional checkpoint for affinity prediction (--affinity_checkpoint)",
        ),
        # ─────────────────────────────────────────────
        # Hardware
        # ─────────────────────────────────────────────
        "devices": LatchParameter(
            display_name="Number of Devices",
            description="Count of GPUs/TPUs/CPUs to use (--devices)",
        ),
        "accelerator": LatchParameter(
            display_name="Accelerator",
            description="Hardware accelerator to target (--accelerator)",
            appearance_type=Multiselect(["cpu", "gpu"]),
        ),
        "num_workers": LatchParameter(
            display_name="Dataloader Workers",
            description="Number of workers for data loading (--num_workers)",
        ),
        "preprocessing_threads": LatchParameter(
            display_name="Pre-processing Threads",
            description="Threads allocated to MSA/feature generation (--preprocessing-threads)",
        ),
        # ─────────────────────────────────────────────
        # Core inference parameters
        # ─────────────────────────────────────────────
        "recycling_steps": LatchParameter(
            display_name="Recycling Steps",
            description="Number of recycles during structure refinement (--recycling_steps)",
        ),
        "sampling_steps": LatchParameter(
            display_name="Sampling Steps",
            description="Number of diffusion steps for structure generation (--sampling_steps)",
        ),
        "diffusion_samples": LatchParameter(
            display_name="Diffusion Samples",
            description="Independent diffusion samples per target (--diffusion_samples)",
        ),
        "max_parallel_samples": LatchParameter(
            display_name="Max Parallel Samples",
            description="Cap on concurrently generated samples (--max_parallel_samples)",
        ),
        "step_scale": LatchParameter(
            display_name="Step Scale",
            description="Controls exploration temperature; lower → more diverse (--step_scale)",
        ),
        # Affinity-specific
        "sampling_steps_affinity": LatchParameter(
            display_name="Affinity Sampling Steps",
            description="Sampling steps for affinity head (--sampling_steps_affinity)",
        ),
        "diffusion_samples_affinity": LatchParameter(
            display_name="Affinity Diffusion Samples",
            description="Diffusion samples for affinity head (--diffusion_samples_affinity)",
        ),
        # ─────────────────────────────────────────────
        # MSA & server settings
        # ─────────────────────────────────────────────
        "use_msa_server": LatchParameter(
            display_name="Use MSA Server",
            description="Call MMSeqs2 web server for MSA generation (--use_msa_server)",
        ),
        "msa_server_url": LatchParameter(
            display_name="MSA Server URL",
            description="URL for the MMSeqs2 server (--msa_server_url)",
        ),
        "msa_pairing_strategy": LatchParameter(
            display_name="MSA Pairing Strategy",
            description="MSA pairing: 'greedy' or 'complete' (--msa_pairing_strategy)",
            appearance_type=Multiselect(["complete", "greedy"]),
        ),
        "msa_mode": LatchParameter(),  # include if your workflow exposes a choice of msa_mode
        "msa_directory": LatchParameter(
            display_name="MSA Directory",
            description="Optional directory containing MSA files",
        ),
        "max_msa_seqs": LatchParameter(
            display_name="Max MSA Seqs",
            description="Upper limit of sequences kept in the MSA (--max_msa_seqs)",
        ),
        "subsample_msa": LatchParameter(
            display_name="Subsample MSA",
            description="Enable stochastic MSA subsampling (--subsample_msa)",
        ),
        "num_subsampled_msa": LatchParameter(
            display_name="Num Subsampled MSA",
            description="Sequences to keep when subsampling (--num_subsampled_msa)",
        ),
        # ─────────────────────────────────────────────
        # Output / bookkeeping toggles
        # ─────────────────────────────────────────────
        "write_full_pae": LatchParameter(
            display_name="Write Full PAE",
            description="Save Predicted Alignment Errors as .npz (--write_full_pae)",
        ),
        "write_full_pde": LatchParameter(
            display_name="Write Full PDE",
            description="Save Predicted Distance Errors as .npz (--write_full_pde)",
        ),
        "output_format": LatchParameter(
            display_name="Output Format",
            description="File format for structures (--output_format)",
            appearance_type=Multiselect(["pdb", "mmcif"]),
        ),
        "override": LatchParameter(
            display_name="Override Existing",
            description="Overwrite existing predictions if present (--override)",
        ),
        "seed": LatchParameter(
            display_name="Random Seed",
            description="Seed for deterministic sampling (--seed)",
        ),
        # ─────────────────────────────────────────────
        # Model selection & extras
        # ─────────────────────────────────────────────
        "model": LatchParameter(
            display_name="Model",
            description="Choose Boltz-1 or Boltz-2 (--model)",
            appearance_type=Multiselect(["boltz1", "boltz2"]),
        ),
        "method": LatchParameter(
            display_name="Method",
            description="Optional method tag for custom protocol (--method)",
        ),
        "use_potentials": LatchParameter(
            display_name="Use Potentials",
            description="Enable potential steering (--use_potentials)",
        ),
        "affinity_mw_correction": LatchParameter(
            display_name="Affinity MW Correction",
            description="Apply molecular-weight correction to affinity head (--affinity_mw_correction)",
        ),
        "no_kernels": LatchParameter(
            display_name="Disable Kernels",
            description="Skip kernel evaluation during inference (--no_kernels)",
        ),
    },
    flow=flow,
)


@workflow(metadata)
def boltz2_workflow(
    # bookkeeping / run metadata
    run_name: str,
    input_mode: str,
    msa_mode: str,
    msa_directory: Optional[LatchDir] = None,
    input_file: Optional[LatchFile] = None,
    input_directory: Optional[LatchDir] = None,
    output_directory: LatchOutputDir = LatchOutputDir("latch:///Boltz-2"),
    # caching / checkpoints
    cache_dir: Optional[str] = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    # hardware
    devices: int = 1,
    accelerator: str = "gpu",
    num_workers: int = 2,
    preprocessing_threads: int = 1,
    # core structure-generation hyper-params
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    step_scale: float = 1.638,
    diffusion_samples: int = 1,
    max_parallel_samples: Optional[int] = None,
    # affinity head hyper-params
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    # msa / server settings
    use_msa_server: bool = False,
    msa_server_url: Optional[str] = None,
    msa_pairing_strategy: Optional[str] = None,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    # output toggles
    write_full_pae: bool = True,
    write_full_pde: bool = False,
    output_format: str = "mmcif",
    override: bool = False,
    seed: Optional[int] = None,
    # extra knobs
    model: str = "boltz2",
    method: Optional[str] = None,
    use_potentials: bool = False,
    affinity_mw_correction: bool = False,
    no_kernels: bool = False,
) -> LatchOutputDir:
    """Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction
    <p align="center">
        <img src="https://www.biopharmatrend.com/files/uploads/2025/06/07/image_2025-06-07_13-58-57.png" alt="boltz1" width="800px"/>
    </p>

    <html>
    <p align="center">
    <img src="https://user-images.githubusercontent.com/31255434/182289305-4cc620e3-86ae-480f-9b61-6ca83283caa5.jpg" alt="Latch Verified" width="100">
    </p>

    <p align="center">
    <strong>
    Latch Verified
    </strong>
    </p>

    ## Boltz-2

    ----
    Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction

    Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy.
    Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and
    binding affinities, a critical component towards accurate molecular design.
    Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods,
    while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

    For more information about the model, see the technical report [https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1].

    Key Features
    ----
    - Structure prediction for proteins, RNA, DNA, and small molecules
    - Support for modified residues
    - Handling of covalent ligands and glycans
    - Pocket conditioning capabilities
    - Multiple Sequence Alignment (MSA) integration

    Input Formats
    ----
    1. FASTA Format
    ```
    >CHAIN_ID|ENTITY_TYPE|MSA_PATH
    SEQUENCE
    ```

    Where:
    - CHAIN_ID: Unique identifier for each chain
    - ENTITY_TYPE: One of [protein, dna, rna, smiles, ccd]
    - MSA_PATH: Path to MSA file (for proteins only)
    - SEQUENCE: Amino acid sequence, nucleotide bases, SMILES string, or CCD code

    2. YAML Format
    ```yaml
    sequences:
      - protein:
          id: [A, B]
          sequence: SEQUENCE
          msa: /root/msa/your_msa.a3m
      - ligand:
          id: [C]
          smiles: SMILES_STRING
    constraints:
      - bond:
          atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
          atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
      - pocket:
          binder: CHAIN_ID
          contacts: [[CHAIN_ID, RES_IDX], [CHAIN_ID, RES_IDX]]
    ```

    MSA Files
    ----
    - All MSA files will be placed in the "/root/msa/" directory from the workflow, adjust YAML and FASTA files accordingly
    - Supported formats: .a3m or CSV
    - For CSV format:
      - Required columns: 'sequence' and 'key'
      - 'key' column used for matching across protein chains

    Features Support Matrix
    ----
    | Feature            | FASTA | YAML |
    |-------------------|--------|------|
    | Polymers          | ✅     | ✅   |
    | SMILES            | ✅     | ✅   |
    | CCD code          | ✅     | ✅   |
    | Custom MSA        | ✅     | ✅   |
    | Modified Residues | ❌     | ✅   |
    | Covalent bonds    | ❌     | ✅   |
    | Pocket condition  | ❌     | ✅   |

    Output Structure
    ----
    The workflow generates the following output structure:

    ```
    out_dir/
    ├── predictions/
    │   └── [input_file]/
    │       ├── [input_file]_model_0.cif           # Structure prediction
    │       ├── confidence_[input_file]_model_0.json # Confidence scores
    │       ├── pae_[input_file]_model_0.npz       # PAE scores
    │       ├── pde_[input_file]_model_0.npz       # PDE scores
    │       └── plddt_[input_file]_model_0.npz     # pLDDT scores
    └── processed/                                  # Processed input data
    ```

    Model Parameters
    ----
    - Recycling steps: 3 (default)
    - Sampling steps: 200 (default)
    - Diffusion samples: 1 (default)
    - Step scale: 1.638 (default)

    References
    ----
    Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction
    Jeremy Wohlwend, Gabriele Corso, Saro Passaro, Mateo Reveiz, Ken Leidal, Wojtek Swiderski, Tally Portnoi, Itamar Chinn, Jacob Silterra, Tommi Jaakkola, Regina Barzilay
    bioRxiv 2024.11.19.624167; doi: https://doi.org/10.1101/2024.11.19.624167

    """

    return boltz_task(
        # ────────────────── basic I/O ──────────────────
        run_name=run_name,
        input_mode=input_mode,
        input_file=input_file,
        input_directory=input_directory,
        output_directory=output_directory,
        msa_mode=msa_mode,
        msa_directory=msa_directory,
        # ────────────── paths & checkpoints ─────────────
        cache_dir=cache_dir,
        checkpoint=checkpoint,
        affinity_checkpoint=affinity_checkpoint,
        # ─────────────────── hardware ───────────────────
        devices=devices,
        accelerator=accelerator,
        num_workers=num_workers,
        preprocessing_threads=preprocessing_threads,
        # ───────────── core inference knobs ─────────────
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        step_scale=step_scale,
        diffusion_samples=diffusion_samples,
        max_parallel_samples=max_parallel_samples,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=diffusion_samples_affinity,
        # ─────────────── MSA / server opts ──────────────
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        max_msa_seqs=max_msa_seqs,
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        # ────────────────── output opts ─────────────────
        write_full_pae=write_full_pae,
        write_full_pde=write_full_pde,
        output_format=output_format,
        override=override,
        seed=seed,
        # ─────────────── model & extras ────────────────
        model=model,
        method=method,
        use_potentials=use_potentials,
        affinity_mw_correction=affinity_mw_correction,
        no_kernels=no_kernels,
    )


# LaunchPlan(
#     boltz2_workflow,
#     "Protein_Fasta",
#     {
#         "run_name": "protein_fasta",
#         "input_mode": "single_file",
#         "input_file": LatchFile(
#             "s3://latch-public/proteinengineering/boltz1/prot.fasta"
#         ),
#         "input_directory": None,
#         "msa_directory": LatchDir("s3://latch-public/proteinengineering/boltz1/msa/"),
#         "use_msa_server": False,
#     },
# )

LaunchPlan(
    boltz2_workflow,
    "Protein_Yaml",
    {
        "run_name": "protein_yaml",
        "input_mode": "single_file",
        "input_file": LatchFile(
            "s3://latch-public/proteinengineering/boltz1/prot.yaml"
        ),
        "input_directory": None,
        "use_msa_server": True,
    },
)

LaunchPlan(
    boltz2_workflow,
    "Protein_Single_Sequence",
    {
        "run_name": "protein_single_sequence",
        "input_mode": "single_file",
        "input_file": LatchFile(
            "s3://latch-public/proteinengineering/boltz1/prot_single_sequence.yaml"
        ),
        "input_directory": None,
        "use_msa_server": True,
    },
)

# LaunchPlan(
#     boltz2_workflow,
#     "Protein_Custom_MSA",
#     {
#         "run_name": "protein_custom_msa",
#         "input_mode": "single_file",
#         "input_file": LatchFile(
#             "s3://latch-public/proteinengineering/boltz1/prot_custom_msa.yaml"
#         ),
#         "input_directory": None,
#         "msa_directory": LatchDir("s3://latch-public/proteinengineering/boltz1/msa/"),
#         "use_msa_server": False,
#     },
# )

# LaunchPlan(
#     boltz2_workflow,
#     "Ligand_Fasta",
#     {
#         "run_name": "ligand_fasta",
#         "input_mode": "single_file",
#         "input_file": LatchFile(
#             "s3://latch-public/proteinengineering/boltz1/ligand.fasta"
#         ),
#         "input_directory": None,
#         "msa_directory": LatchDir("s3://latch-public/proteinengineering/boltz1/msa/"),
#         "use_msa_server": False,
#     },
# )

LaunchPlan(
    boltz2_workflow,
    "Ligand_Yaml",
    {
        "run_name": "ligand_yaml",
        "input_mode": "single_file",
        "input_file": LatchFile(
            "s3://latch-public/proteinengineering/boltz1/ligand.yaml"
        ),
        "input_directory": None,
        "msa_directory": LatchDir("s3://latch-public/proteinengineering/boltz1/msa/"),
        "use_msa_server": False,
    },
)

LaunchPlan(
    boltz2_workflow,
    "Multimer",
    {
        "run_name": "multimer",
        "input_mode": "single_file",
        "input_file": LatchFile(
            "s3://latch-public/proteinengineering/boltz1/multimer.yaml"
        ),
        "input_directory": None,
        "use_msa_server": True,
    },
)
