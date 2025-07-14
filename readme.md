# Perovskite Structure Prediction with MPI-Parallelized Symbolic Regression

This project uses dimensional symbolic regression with MPI parallelization to predict whether a compound will form a perovskite structure based on ionic radii and oxidation states.

## Features

- **MPI Parallelization**: Distributed genetic programming across multiple processes
- **Dimensional Analysis**: Ensures physically meaningful expressions
- **Cross-validation**: Robust model evaluation
- **Shannon Radii Integration**: Uses experimental ionic radii data

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

For MPI support on Windows, you may also need to install Microsoft MPI:
- Download from: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- Or use Intel MPI or MPICH

## Running the Code

### Serial Mode (Single Process)
```bash
python main.py
```

### MPI Parallel Mode (Recommended)
```bash
# Run with 4 processes
mpiexec -n 4 python main.py

# Run the simple MPI test
mpiexec -n 4 python run_mpi.py

# On Windows, you can also use the batch file
run_mpi.bat
```

### Adjusting Population Size for MPI
The population size is automatically scaled based on the number of MPI processes:
- With 4 processes: 4000 individuals total (1000 per process)
- With 8 processes: 8000 individuals total (1000 per process)

## Files

### Core Code
- `customsr_v2.py` - MPI-parallelized symbolic regression implementation
- `main.py` - Main execution script with cross-validation
- `run_mpi.py` - Simple MPI test script
- `ldspfeliciano.py` - Shannon radii data processing

### Data Files
- `Database_S1.1.csv` - Known perovskites
- `Database_S1.2.csv` - Known non-perovskites
- `Database_S1.1_with_radii.csv` - Known perovskites with Shannon radii
- `Database_S1.2_with_radii.csv` - Known non-perovskites with Shannon radii
- `shannon-radii.json` - Shannon ionic radii database

## Database CSV Files

1. `S1.1` - Database of known perovskites
2. `S1.2` - Database of known non-perovskites
3. `S2.2.1` - Database of predicted compounds that were found on the internet and are perovskites
4. `S2.2.2` - Database of predicted compounds that were found on the internet and are non-perovskites

Dataset accessible at [https://archive.materialscloud.org/record/2018.0012/v1](https://archive.materialscloud.org/record/2018.0012/v1).

The dataset csv files were modified slightly in terms of their headlines/column headers to simplify parsing. The data was not changed.

## MPI Performance Benefits

The MPI implementation provides several performance advantages:

1. **Parallel Fitness Evaluation**: Each process evaluates its local population independently
2. **Distributed Population**: Population is split across processes for parallel evolution
3. **Global Best Sharing**: Best individuals are shared across all processes
4. **Migration**: Periodic exchange of individuals between processes maintains diversity
5. **Scalable**: Performance scales nearly linearly with the number of processes

## Example Output

```
Running with MPI: 4 processes
Population size: 4000
Generations: 100
Starting MPI evolution with 4000 individuals (1000 per process)
Generation   0: Best fitness = 0.234567
  Best expression: 0.742*(rA + rX)/(1.414*(rB + rX)) + 0.123 (Variables: 3)
...
```