
"""
Perovskite Structure Prediction using Symbolic Regression with gplearn and MPI

This script uses gplearn symbolic regression to predict whether a compound
will form a perovskite structure based on ionic radii and oxidation states.

To run with MPI parallelization:
    mpiexec -n 4 python main.py

Where 4 is the number of processes to use.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from datetime import datetime
import os
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# For MPI support
try:
    from mpi4py import MPI
    HAS_MPI = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    HAS_MPI = False
    rank = 0
    size = 1
    print("Warning: mpi4py not available. Running in serial mode.")

# ============================================================================
# HYPERPARAMETERS AND CONFIGURATION
# ============================================================================

# Data files
PEROVSKITE_DATA_FILE = 'Database_S1.1_with_radii.csv'
NON_PEROVSKITE_DATA_FILE = 'Database_S1.2_with_radii.csv'

# Cross-validation settings
N_FOLDS = 5
RANDOM_STATE = 42

# Symbolic regression hyperparameters - gplearn parameters
SR_POPULATION_SIZE = 5000  # Population size for genetic programming
SR_GENERATIONS = 20  # Number of generations
SR_TOURNAMENT_SIZE = 20  # Tournament size for selection
SR_CONST_RANGE = (-1.0, 1.0)  # Range for random constants
SR_INIT_DEPTH = (2, 6)  # Initial depth range for trees
SR_INIT_METHOD = 'half and half'  # Initialization method
SR_FUNCTION_SET = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min')
SR_PARSIMONY_COEFFICIENT = 0.001  # Parsimony coefficient to control complexity
SR_P_CROSSOVER = 0.7  # Probability of crossover
SR_P_SUBTREE_MUTATION = 0.1  # Probability of subtree mutation
SR_P_HOIST_MUTATION = 0.05  # Probability of hoist mutation
SR_P_POINT_MUTATION = 0.1  # Probability of point mutation
SR_MAX_SAMPLES = 0.9  # Fraction of samples to use for training each individual
SR_VERBOSE = 1  # Verbosity level
SR_RANDOM_STATE = 42  # Random state for reproducibility
SR_N_JOBS = 1  # Number of parallel jobs (set to 1 for MPI compatibility)

# Prediction threshold
PREDICTION_THRESHOLD = 0.5

# Output settings
OUTPUT_DIR = 'results'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = f'perovskite_sr_results_{TIMESTAMP}.csv'
SUMMARY_FILE = f'perovskite_sr_summary_{TIMESTAMP}.csv'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if rank == 0 and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
def load_and_examine_data():
    """Load and examine the perovskite datasets."""
    if rank == 0:
        print("Loading datasets...")
    known_perovskites = pd.read_csv(PEROVSKITE_DATA_FILE)
    known_non_perovskites = pd.read_csv(NON_PEROVSKITE_DATA_FILE)
    
    if rank == 0:
        print(f"Known Perovskites shape: {known_perovskites.shape}")
        print(f"Known Non-Perovskites shape: {known_non_perovskites.shape}")
        
        print(f"\nKnown Perovskites columns: {known_perovskites.columns.tolist()}")
        
        # Check radius column statistics
        radii_cols = ['A Radius', 'B Radius', "B' Radius", 'X Radius']
        print(f"\nRadii columns statistics:")
        for col in radii_cols:
            if col in known_perovskites.columns:
                non_null_count = known_perovskites[col].notna().sum()
                total_count = len(known_perovskites)
                print(f"  {col}: {non_null_count}/{total_count} non-null values")
    
    return known_perovskites, known_non_perovskites


def prepare_combined_dataset(known_perovskites, known_non_perovskites):
    """Prepare and combine perovskite and non-perovskite datasets."""
    if rank == 0:
        print("\nPreparing combined dataset...")
    
    # Add labels
    perovskite_data = known_perovskites.copy()
    perovskite_data['is_perovskite'] = 1
    
    non_perovskite_data = known_non_perovskites.copy()
    non_perovskite_data['is_perovskite'] = 0
    
    # Combine datasets
    combined_data = pd.concat([perovskite_data, non_perovskite_data], ignore_index=True)
    
    if rank == 0:
        print(f"Combined dataset shape: {combined_data.shape}")
        print(f"Perovskites: {sum(combined_data['is_perovskite'] == 1)}")
        print(f"Non-perovskites: {sum(combined_data['is_perovskite'] == 0)}")
    
    return combined_data

def clean_and_balance_data(combined_data):
    """Clean missing data and balance the dataset."""
    if rank == 0:
        print("\nCleaning and balancing data...")
    
    # Check available columns
    radii_cols = ['A Radius', 'B Radius', "B' Radius", 'X Radius']
    available_radii = [col for col in radii_cols if col in combined_data.columns]
    if rank == 0:
        print(f"Available radius columns: {available_radii}")
    
    oxidation_cols = [col for col in combined_data.columns if 'Oxidation State' in col]
    if rank == 0:
        print(f"Available oxidation state columns: {oxidation_cols}")
    
    # Remove rows with missing radius data
    if rank == 0:
        print(f"Before removing missing data: {len(combined_data)} samples")
    for col in available_radii:
        combined_data = combined_data.dropna(subset=[col])
    if rank == 0:
        print(f"After removing missing radius data: {len(combined_data)} samples")
    
    # Balance the dataset
    perovskite_samples = combined_data[combined_data['is_perovskite'] == 1]
    non_perovskite_samples = combined_data[combined_data['is_perovskite'] == 0]
    
    n_perovskite = len(perovskite_samples)
    n_non_perovskite = len(non_perovskite_samples)
    
    if n_perovskite > n_non_perovskite:
        perovskite_samples_bal = resample(
            perovskite_samples, 
            replace=False, 
            n_samples=n_non_perovskite, 
            random_state=RANDOM_STATE
        )
        non_perovskite_samples_bal = non_perovskite_samples
    elif n_non_perovskite > n_perovskite:
        non_perovskite_samples_bal = resample(
            non_perovskite_samples, 
            replace=False, 
            n_samples=n_perovskite, 
            random_state=RANDOM_STATE
        )
        perovskite_samples_bal = perovskite_samples
    else:
        perovskite_samples_bal = perovskite_samples
        non_perovskite_samples_bal = non_perovskite_samples
    
    balanced_data = pd.concat([perovskite_samples_bal, non_perovskite_samples_bal], ignore_index=True)  # type: ignore
    
    if rank == 0:
        print(f"Balanced dataset shape: {balanced_data.shape}")
        print(f"Perovskites: {sum(balanced_data['is_perovskite'] == 1)}")
        print(f"Non-perovskites: {sum(balanced_data['is_perovskite'] == 0)}")
    
    return balanced_data

def prepare_features(balanced_data):
    """Prepare feature matrix and labels."""
    if rank == 0:
        print("\nPreparing features...")
    
    features = []
    feature_names = []
    
    # Add ionic radii (physical dimension: length)
    for col in ['A Radius', 'B Radius', "B' Radius", 'X Radius']:
        if col in balanced_data.columns:
            features.append(balanced_data[col].values)
            feature_names.append(col.replace(' Radius', '_radius'))
    
    # Add oxidation states (dimensionless)
    oxidation_mapping = {
        'A Oxidation State': 'nA',
        'B Oxidation State': 'nB', 
        "B' Oxidation State": "nB'",
        'X Oxidation State': 'nX'
    }
    
    for col, name in oxidation_mapping.items():
        if col in balanced_data.columns:
            features.append(balanced_data[col].values)
            feature_names.append(name)
    
    # Create feature matrix
    X = np.column_stack(features)
    y = balanced_data['is_perovskite'].values
    
    if rank == 0:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature names: {feature_names}")
        print(f"Target distribution: {np.bincount(y)}")
    
    return X, y, feature_names

def run_cross_validation(X, y, feature_names):
    """Run stratified k-fold cross-validation."""
    if rank == 0:
        print(f"\nRunning {N_FOLDS}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    cv_train_accuracies = []
    cv_test_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        if rank == 0:
            print(f"\nProcessing Fold {fold}...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Balance only the training set
        perovskite_idx = np.where(y_train == 1)[0]
        non_perovskite_idx = np.where(y_train == 0)[0]
        n_perovskite = len(perovskite_idx)
        n_non_perovskite = len(non_perovskite_idx)
        
        if n_perovskite > n_non_perovskite:
            perovskite_idx_bal = np.random.choice(perovskite_idx, n_non_perovskite, replace=False)
            non_perovskite_idx_bal = non_perovskite_idx
        else:
            non_perovskite_idx_bal = np.random.choice(non_perovskite_idx, n_perovskite, replace=False)
            perovskite_idx_bal = perovskite_idx
            
        balanced_idx = np.concatenate([perovskite_idx_bal, non_perovskite_idx_bal])
        X_train_bal = X_train[balanced_idx]
        y_train_bal = y_train[balanced_idx]
        
        # Train gplearn symbolic regressor on balanced training set
        regressor = SymbolicRegressor(
            population_size=SR_POPULATION_SIZE,
            generations=SR_GENERATIONS,
            tournament_size=SR_TOURNAMENT_SIZE,
            const_range=SR_CONST_RANGE,
            init_depth=SR_INIT_DEPTH,
            init_method=SR_INIT_METHOD,
            function_set=SR_FUNCTION_SET,
            parsimony_coefficient=SR_PARSIMONY_COEFFICIENT,
            p_crossover=SR_P_CROSSOVER,
            p_subtree_mutation=SR_P_SUBTREE_MUTATION,
            p_hoist_mutation=SR_P_HOIST_MUTATION,
            p_point_mutation=SR_P_POINT_MUTATION,
            max_samples=SR_MAX_SAMPLES,
            verbose=SR_VERBOSE,
            random_state=SR_RANDOM_STATE + rank,  # Different seed for each MPI process
            n_jobs=SR_N_JOBS
        )
        
        regressor.fit(X_train_bal, y_train_bal)
        
        # Evaluate on both train and test sets
        y_train_pred = regressor.predict(X_train_bal)
        y_train_pred_binary = (y_train_pred > PREDICTION_THRESHOLD).astype(int)
        y_test_pred = regressor.predict(X_test)
        y_test_pred_binary = (y_test_pred > PREDICTION_THRESHOLD).astype(int)
        
        train_acc = accuracy_score(y_train_bal, y_train_pred_binary)
        test_acc = accuracy_score(y_test, y_test_pred_binary)
        
        cv_train_accuracies.append(train_acc)
        cv_test_accuracies.append(test_acc)
        
        if rank == 0:
            print(f"  Train Accuracy (balanced): {train_acc:.4f}")
            print(f"  Test Accuracy (original): {test_acc:.4f}")
            print(f"  Best program: {regressor._program}")
        
        # Generate classification report
        report = classification_report(
            y_test, y_test_pred_binary, 
            target_names=['Non-Perovskite', 'Perovskite'],
            output_dict=True
        )  # type: ignore
        
        # Store results for this fold
        fold_result = {
            'fold': fold,
            'best_individual': str(regressor._program),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision_non_perovskite': report['Non-Perovskite']['precision'],
            'recall_non_perovskite': report['Non-Perovskite']['recall'],
            'f1_non_perovskite': report['Non-Perovskite']['f1-score'],
            'precision_perovskite': report['Perovskite']['precision'],
            'recall_perovskite': report['Perovskite']['recall'],
            'f1_perovskite': report['Perovskite']['f1-score'],
            'macro_avg_precision': report['macro avg']['precision'],
            'macro_avg_recall': report['macro avg']['recall'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_precision': report['weighted avg']['precision'],
            'weighted_avg_recall': report['weighted avg']['recall'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
        results.append(fold_result)
        
        if rank == 0:
            print("-" * 50)
    
    return results, cv_train_accuracies, cv_test_accuracies

def save_results(results, cv_train_accuracies, cv_test_accuracies, feature_names):
    """Save all results to CSV files."""
    if rank == 0:
        print("\nSaving results...")
    
    ensure_output_dir()
    
    # Save detailed fold results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    results_df.to_csv(results_path, index=False)
    if rank == 0:
        print(f"Detailed results saved to: {results_path}")
    
    # Create and save summary statistics
    summary_stats = {
        'metric': [
            'mean_train_accuracy', 'std_train_accuracy',
            'mean_test_accuracy', 'std_test_accuracy',
            'mean_precision_perovskite', 'std_precision_perovskite',
            'mean_recall_perovskite', 'std_recall_perovskite',
            'mean_f1_perovskite', 'std_f1_perovskite',
            'mean_macro_avg_f1', 'std_macro_avg_f1'
        ],
        'value': [
            np.mean(cv_train_accuracies), np.std(cv_train_accuracies),
            np.mean(cv_test_accuracies), np.std(cv_test_accuracies),
            np.mean([r['precision_perovskite'] for r in results]), 
            np.std([r['precision_perovskite'] for r in results]),
            np.mean([r['recall_perovskite'] for r in results]), 
            np.std([r['recall_perovskite'] for r in results]),
            np.mean([r['f1_perovskite'] for r in results]), 
            np.std([r['f1_perovskite'] for r in results]),
            np.mean([r['macro_avg_f1'] for r in results]), 
            np.std([r['macro_avg_f1'] for r in results])
        ]
    }
    
    # Add configuration information
    config_info = {
        'metric': [
            'n_folds', 'random_state', 'population_size', 'generations',
            'tournament_size', 'parsimony_coefficient', 'p_crossover', 'p_subtree_mutation',
            'p_hoist_mutation', 'p_point_mutation', 'prediction_threshold', 'feature_names'
        ],
        'value': [
            N_FOLDS, RANDOM_STATE, SR_POPULATION_SIZE, SR_GENERATIONS,
            SR_TOURNAMENT_SIZE, SR_PARSIMONY_COEFFICIENT, SR_P_CROSSOVER, SR_P_SUBTREE_MUTATION,
            SR_P_HOIST_MUTATION, SR_P_POINT_MUTATION, PREDICTION_THRESHOLD, ', '.join(feature_names)
        ]
    }
    
    # Combine summary and config
    summary_data = {
        'metric': summary_stats['metric'] + config_info['metric'],
        'value': summary_stats['value'] + config_info['value']
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_FILE)
    summary_df.to_csv(summary_path, index=False)
    if rank == 0:
        print(f"Summary statistics saved to: {summary_path}")
    
    return results_df, summary_df

def print_final_summary(cv_train_accuracies, cv_test_accuracies):
    """Print final summary statistics."""
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Mean Train Accuracy: {np.mean(cv_train_accuracies):.4f} ± {np.std(cv_train_accuracies):.4f}")
    print(f"Mean Test Accuracy:  {np.mean(cv_test_accuracies):.4f} ± {np.std(cv_test_accuracies):.4f}")
    print(f"{'='*60}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    if rank == 0:
        print("Perovskite Structure Prediction using gplearn Symbolic Regression with MPI")
        print("="*75)
        if HAS_MPI:
            print(f"Running with MPI: {size} processes")
        else:
            print("Running in serial mode (MPI not available)")
        print(f"Population size: {SR_POPULATION_SIZE}")
        print(f"Generations: {SR_GENERATIONS}")
    
    # Load and examine data
    known_perovskites, known_non_perovskites = load_and_examine_data()
    
    # Prepare combined dataset
    combined_data = prepare_combined_dataset(known_perovskites, known_non_perovskites)
    
    # Clean and balance data
    balanced_data = clean_and_balance_data(combined_data)
    
    # Prepare features
    X, y, feature_names = prepare_features(balanced_data)
    
    # Run cross-validation
    results, cv_train_accuracies, cv_test_accuracies = run_cross_validation(
        X, y, feature_names
    )
    
    # Save results (only on root process)
    if rank == 0:
        results_df, summary_df = save_results(
            results, cv_train_accuracies, cv_test_accuracies, feature_names
        )
    
    # Print final summary
    if rank == 0:
        print_final_summary(cv_train_accuracies, cv_test_accuracies)

if __name__ == "__main__":
    main()


