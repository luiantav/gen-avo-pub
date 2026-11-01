PATH_BASE="${HOME}/gen-avo/behavioral-modeling"

#/home/mpib/verra
# Give message to user
echo "strt"
echo PATH_BASE
N_CPUS=1
# maximum number of threads per process:
N_THREADS=1
# memory demand in *GB*
MEM_MB=20



JOB_NAME="LS_REC"


# Create job file
echo "#!/bin/bash" > job.slurm
# name of the job
echo "#SBATCH --job-name ${JOB_NAME}" >> job.slurm
# set the expected maximum running time for the job:
echo "#SBATCH --time 24:00:00" >> job.slurm
# determine how much RAM your operation needs:
echo "#SBATCH --mem ${MEM_MB}GB" >> job.slurm
# determine number of CPUs
echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job.slurm
# write to log folder
#echo "#SBATCH --output ${PATH_LOG}/slurm-${JOB_NAME}.%j.out" >> job.slurm
echo "#SBATCH --output /home/mpib/verra/avgenlogs/slurm_${JOB_NAME}_%j.out" >> job.slurm
echo "#SBATCH --partition short" >> job.slurm
    
# Load R module
echo "module unload R" >> job.slurm
echo "module load R/4.2" >> job.slurm
echo "python3 ${PATH_BASE}/ls_recovery_cluster_peaks.py \
    -s ${sub}" >> job.slurm


# submit job to cluster queue and remove it to avoid confusion:
sbatch job.slurm
rm -f job.slurm
sleep 6




