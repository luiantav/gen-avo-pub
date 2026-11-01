#!/bin/bash
PATH_BASE="${HOME}/gen-avo/behavioral-modeling/MDP"

# Give message to user
echo "starting model recovery"
echo "${PATH_BASE}"
N_CPUS=1
# maximum number of threads per process:
N_THREADS=1
# memory demand in *GB*
MEM_MB=1


#models=("weighted" "pessimistic" "baseline" "dyn_weighted")
models=("baseline")


for mod in "${models[@]}"; do
    JOB_NAME="mdp_modrec_${mod}"


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
echo "python3 /home/mpib/verra/gen-avo/behavioral-modeling/MDP/mdp_modelrecovery_cluster.py -s ${mod}" >> job.slurm


# submit job to cluster queue and remove it to avoid confusion:
sbatch job.slurm
rm -f job.slurm
sleep 6

done



