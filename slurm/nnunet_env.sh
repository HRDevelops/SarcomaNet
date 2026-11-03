# sourced by the nnU-Net Slurm scripts (run from the repo root)
module load Python/3.11.5-GCCcore-13.2.0
source ../venv_nnunet311/bin/activate
export nnUNet_raw=$(realpath ../nnunet)/raw nnUNet_preprocessed=$(realpath ../nnunet)/preprocessed nnUNet_results=$(realpath ../nnunet)/results
export nnUNet_n_proc_DA=8
export nnUNet_compile=f   # torch.compile/triton writes .so files to /tmp, which is noexec on compute nodes (job 34875 failed)
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
