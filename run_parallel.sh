#!/bin/bash
#SBATCH -J perovskite_sr_parallel  # Job name
#SBATCH -o perovskite_sr_parallel.o%j  # Name of stdout output file
#SBATCH -e perovskite_sr_parallel.e%j  # Name of stderr error file
#SBATCH -p skx                     # Queue (partition) name - using SKX nodes
#SBATCH -N 1                       # Total number of nodes
#SBATCH -n 48                      # Total number of tasks (full SKX node: 48 cores)
#SBATCH -t 08:00:00                # Run time (hh:mm:ss) - 8 hours for larger runs
#SBATCH --mail-user=your_email@domain.com  # Replace with your email
#SBATCH --mail-type=all            # Send email at begin and end of job
#SBATCH -A your_project            # Replace with your allocation name

# This script runs the perovskite structure prediction using dimensional symbolic regression
# on TACC Stampede3 with parallel processing capabilities. Designed for larger parameter sweeps
# or multiple independent runs.

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Number of tasks: $SLURM_NTASKS"

# Load required modules
module list
echo "Loading Python module..."
module load python3/3.9.7

# Set environment variables for parallel processing
export OMP_NUM_THREADS=1  # Prevent numpy/sklearn from using multiple threads per process
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Navigate to scratch directory
echo "Changing to scratch directory..."
cd $SCRATCH

# Create job-specific directory
JOB_DIR="perovskite_sr_parallel_job_${SLURM_JOB_ID}"
mkdir -p $JOB_DIR
cd $JOB_DIR

# Copy required files
echo "Copying input files..."
cp $HOME/perovskite_sr/*.py .
cp $HOME/perovskite_sr/*.csv .
cp $HOME/perovskite_sr/*.json .

# List files to verify they were copied
echo "Files in working directory:"
ls -la

# Install required Python packages if needed
echo "Installing required Python packages..."
pip3 install --user pandas numpy scikit-learn joblib

# Create a parameter sweep script for parallel execution
cat > run_parameter_sweep.py << 'EOF'
import os
import sys
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

def run_single_experiment(params):
    """Run a single experiment with given parameters."""
    pop_size, generations, mutation_rate, crossover_rate = params
    
    # Create temporary directory for this run
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy main script to temp directory
        temp_script = os.path.join(temp_dir, 'main_temp.py')
        
        # Read original main.py and modify hyperparameters
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Replace hyperparameters in the content
        content = content.replace('SR_POPULATION_SIZE = 10', f'SR_POPULATION_SIZE = {pop_size}')
        content = content.replace('SR_GENERATIONS = 200', f'SR_GENERATIONS = {generations}')
        content = content.replace('SR_MUTATION_RATE = 0.15', f'SR_MUTATION_RATE = {mutation_rate}')
        content = content.replace('SR_CROSSOVER_RATE = 0.7', f'SR_CROSSOVER_RATE = {crossover_rate}')
        
        # Write modified script
        with open(temp_script, 'w') as f:
            f.write(content)
        
        # Copy data files to temp directory
        for file in ['Database_S1.1_with_radii.csv', 'Database_S1.2_with_radii.csv']:
            if os.path.exists(file):
                subprocess.run(['cp', file, temp_dir])
        
        # Run the experiment
        try:
            result = subprocess.run(
                ['python3', temp_script],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per experiment
            )
            
            return {
                'pop_size': pop_size,
                'generations': generations,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'success': result.returncode == 0,
                'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                'stderr': result.stderr[-1000:] if result.stderr else ''   # Last 1000 chars
            }
        except Exception as e:
            return {
                'pop_size': pop_size,
                'generations': generations,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'success': False,
                'stdout': '',
                'stderr': str(e)
            }

def main():
    # Define parameter grid for sweep
    param_grid = [
        (10, 100, 0.1, 0.7),   # Small, fast run
        (10, 200, 0.15, 0.7),  # Default parameters
        (20, 150, 0.12, 0.8),  # Larger population
        (15, 250, 0.18, 0.6),  # More generations
        (25, 100, 0.2, 0.75),  # Higher mutation
    ]
    
    # Limit to available cores
    max_workers = min(len(param_grid), int(os.environ.get('SLURM_NTASKS', 1)))
    
    print(f"Running parameter sweep with {max_workers} parallel workers")
    print(f"Parameter combinations: {len(param_grid)}")
    
    results = []
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {executor.submit(run_single_experiment, params): params 
                           for params in param_grid}
        
        for i, future in enumerate(as_completed(future_to_params)):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed experiment {i+1}/{len(param_grid)}: {params} - "
                      f"Success: {result['success']}")
            except Exception as exc:
                print(f"Experiment {params} generated an exception: {exc}")
                results.append({
                    'pop_size': params[0],
                    'generations': params[1],
                    'mutation_rate': params[2],
                    'crossover_rate': params[3],
                    'success': False,
                    'stdout': '',
                    'stderr': str(exc)
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('parameter_sweep_results.csv', index=False)
    
    print("\nParameter sweep completed!")
    print(f"Successful runs: {results_df['success'].sum()}/{len(results_df)}")
    print("Results saved to parameter_sweep_results.csv")

if __name__ == '__main__':
    main()
EOF

# Run the parameter sweep or single experiment based on argument
if [ "$1" == "sweep" ]; then
    echo "Running parameter sweep with parallel processing..."
    time python3 run_parameter_sweep.py 2>&1 | tee parameter_sweep_output.log
else
    echo "Running single experiment..."
    time python3 main.py 2>&1 | tee perovskite_sr_output.log
fi

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

echo "Job finished at: $(date)"
echo "Total job duration: $SECONDS seconds"

# Print final job statistics
echo "Job statistics:"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Tasks used: $SLURM_NTASKS"
echo "Exit status: $?"
