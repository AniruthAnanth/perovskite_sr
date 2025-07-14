#!/bin/bash
#SBATCH -J perovskite_sr       # Job name
#SBATCH -o perovskite_sr.o%j   # Name of stdout output file
#SBATCH -e perovskite_sr.e%j   # Name of stderr error file
#SBATCH -p skx                 # Queue (partition) name - using SKX nodes
#SBATCH -N 1                   # Total number of nodes (1 for single-node job)
#SBATCH -n 1                   # Total number of MPI tasks (1 for single process)
#SBATCH -t 04:00:00            # Run time (hh:mm:ss) - 4 hours should be sufficient
#SBATCH --mail-user=your_email@domain.com  # Replace with your email
#SBATCH --mail-type=all        # Send email at begin and end of job
#SBATCH -A your_project        # Replace with your allocation name

# =============================================================================
# TACC Stampede3 SLURM Job Script for Perovskite Structure Prediction
# =============================================================================
# 
# BEFORE RUNNING:
# 1. Replace 'your_email@domain.com' with your actual email address
# 2. Replace 'your_project' with your TACC allocation name
# 3. Replace the GITHUB_REPO URL below with your actual repository URL
# 4. Ensure your GitHub repository is public or you have SSH keys set up
#
# TO SUBMIT: sbatch run.sh
# TO CHECK STATUS: squeue -u $USER
# TO CANCEL: scancel <job_id>
# =============================================================================

# This script:
# 1. Clones your GitHub repository to SCRATCH directory
# 2. Runs the perovskite structure prediction using dimensional symbolic regression
# 3. Copies results back to both WORK and HOME directories for permanent storage

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Load required modules
module list
echo "Loading required modules..."
module load python3/3.9.7
module load git/2.24.1

# Verify environment
echo "Python version:"
python3 --version
echo "Python path:"
which python3
echo "Git version:"
git --version

# Navigate to scratch directory for better I/O performance
# Always run jobs from $SCRATCH on TACC systems
echo "Changing to scratch directory..."
cd $SCRATCH

# Create job-specific directory
JOB_DIR="perovskite_sr_job_${SLURM_JOB_ID}"
mkdir -p $JOB_DIR
cd $JOB_DIR

# Clone the repository from GitHub
# Replace with your actual GitHub repository URL
GITHUB_REPO="https://github.com/yourusername/perovskite_sr.git"
echo "Cloning repository from GitHub..."
git clone $GITHUB_REPO .

# If clone fails, try with specific branch or handle authentication
if [ $? -ne 0 ]; then
    echo "Git clone failed. Trying alternative approaches..."
    # Alternative: try cloning with specific branch
    # git clone -b main $GITHUB_REPO .
    # Or if using SSH:
    # git clone git@github.com:yourusername/perovskite_sr.git .
    echo "Please check your repository URL and authentication"
    exit 1
fi

# List files to verify they were cloned
echo "Files cloned from repository:"
ls -la

# Install required Python packages if needed
echo "Installing required Python packages..."
pip3 install --user pandas numpy scikit-learn

# Run the main Python script
echo "Starting perovskite structure prediction..."
echo "----------------------------------------"

# Run with time measurement and capture both stdout and stderr
time python3 main.py 2>&1 | tee perovskite_sr_output.log

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code: $?"
fi

# List output files
echo "Output files generated:"
ls -la results/ 2>/dev/null || echo "No results directory found"
ls -la *.csv *.log 2>/dev/null || echo "No CSV or log files found"

# Copy results back to permanent locations
echo "Copying results to permanent storage..."

# Create timestamped directory names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_RESULTS_DIR="$WORK/perovskite_sr_results_${SLURM_JOB_ID}_${TIMESTAMP}"
HOME_RESULTS_DIR="$HOME/perovskite_sr_results_${SLURM_JOB_ID}_${TIMESTAMP}"

# Copy to WORK directory
echo "Copying results to WORK directory: $WORK_RESULTS_DIR"
mkdir -p "$WORK_RESULTS_DIR"
cp -r results/ "$WORK_RESULTS_DIR/" 2>/dev/null || echo "No results directory to copy"
cp *.csv "$WORK_RESULTS_DIR/" 2>/dev/null || echo "No CSV files to copy"
cp *.log "$WORK_RESULTS_DIR/" 2>/dev/null || echo "No log files to copy"
cp main.py "$WORK_RESULTS_DIR/" 2>/dev/null || echo "No main.py to copy"

# Copy to HOME directory
echo "Copying results to HOME directory: $HOME_RESULTS_DIR"
mkdir -p "$HOME_RESULTS_DIR"
cp -r results/ "$HOME_RESULTS_DIR/" 2>/dev/null || echo "No results directory to copy"
cp *.csv "$HOME_RESULTS_DIR/" 2>/dev/null || echo "No CSV files to copy"
cp *.log "$HOME_RESULTS_DIR/" 2>/dev/null || echo "No log files to copy"
cp main.py "$HOME_RESULTS_DIR/" 2>/dev/null || echo "No main.py to copy"

# Create summary file with job information
SUMMARY_FILE="job_summary_${SLURM_JOB_ID}.txt"
cat > "$SUMMARY_FILE" << EOF
Perovskite Structure Prediction Job Summary
==========================================
Job ID: $SLURM_JOB_ID
Node: $(hostname)
Start Time: $(date)
Working Directory: $(pwd)
GitHub Repository: $GITHUB_REPO
Results Location (WORK): $WORK_RESULTS_DIR
Results Location (HOME): $HOME_RESULTS_DIR

Files Generated:
$(ls -la *.csv *.log 2>/dev/null || echo "No CSV or log files found")

Results Directory Contents:
$(ls -la results/ 2>/dev/null || echo "No results directory found")
EOF

# Copy summary to both locations
cp "$SUMMARY_FILE" "$WORK_RESULTS_DIR/"
cp "$SUMMARY_FILE" "$HOME_RESULTS_DIR/"

echo "Results copied to:"
echo "  WORK: $WORK_RESULTS_DIR"
echo "  HOME: $HOME_RESULTS_DIR"

echo "Job finished at: $(date)"
echo "Total job duration: $SECONDS seconds"

# Print final job statistics
echo "Job statistics:"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit status: $?"