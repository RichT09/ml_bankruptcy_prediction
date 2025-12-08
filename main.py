#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Pipeline Orchestrator
Bankruptcy Prediction - Complete ML Workflow
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pickle
import logging

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data_features import run_data_features_pipeline
from src.preprocessing import run_preprocessing_pipeline
from src.models import run_models_pipeline
from src.evaluation import run_evaluation_pipeline
from src.eda_interpretation import run_eda_interpretation_pipeline

# Configure root logger to use stdout (captured by TeeOutput)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------------

class TeeOutput:
    """Capture stdout/stderr to both console and file, filtering progress bars"""
    def __init__(self, filename, stream, filter_progress=True):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stream = stream
        self.filter_progress = filter_progress
        
    def write(self, data):
        # Always write to console
        self.stream.write(data)
        
        # Filter out tqdm/progress bar lines from log file
        if self.filter_progress and data:
            # Skip lines with progress bar patterns
            if any(pattern in data for pattern in [
                'it/s]', 'PermutationExplainer', '\r', '%|#', '%|'
            ]):
                return
        
        self.file.write(data)
        self.file.flush()
        
    def flush(self):
        self.stream.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


def setup_logging(log_dir: Path, filter_progress: bool = True):
    """Setup logging to file and console"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"pipeline_run_{timestamp}.log"
    
    # Redirect stdout and stderr to file + console (filter progress bars)
    sys.stdout = TeeOutput(log_file, sys.__stdout__, filter_progress)
    sys.stderr = TeeOutput(log_file, sys.__stderr__, filter_progress)
    
    return log_file


def cleanup_logging():
    """Restore original stdout/stderr"""
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    if hasattr(sys.stderr, 'close'):
        sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------

def run_complete_pipeline(
    skip_eda: bool = False,
    skip_shap: bool = False,
    verbose: bool = True,
    log_output: bool = True
):
    """
    Run complete bankruptcy prediction pipeline
    
    Pipeline steps:
    1. Data loading & feature selection
    2. Preprocessing (cleaning, split, scaling, SMOTE)
    3. Model training (base + tuned)
    4. Evaluation & comparison
    5. EDA & interpretation (SHAP)
    
    Args:
        skip_eda: Skip exploratory data analysis
        skip_shap: Skip SHAP analysis
        verbose: Print detailed progress
        log_output: Save terminal output to logs folder
    
    Returns:
        0 if success, 1 if error
    """
    
    # Setup logging
    log_file = None
    if log_output:
        logs_dir = config.metrics_dir  # outputs/logs/
        log_file = setup_logging(logs_dir)
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("BANKRUPTCY PREDICTION - COMPLETE ML PIPELINE")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Split year: {config.data.split_year}")
    print(f"SMOTE: {config.models.use_smote}")
    if log_file:
        print(f"Log file: {log_file}")
    print("="*80)
    
    try:
        # ====================================================================
        # STEP 1: DATA LOADING & FEATURE SELECTION
        # ====================================================================
        print("\n" + "-"*80)
        print("STEP 1/5: DATA LOADING & FEATURE SELECTION")
        print("-"*80)
        
        dataset = run_data_features_pipeline(verbose=verbose)
        
        # ====================================================================
        # STEP 2: PREPROCESSING
        # ====================================================================
        print("\n" + "-"*80)
        print("STEP 2/5: PREPROCESSING")
        print("-"*80)
        
        preprocessed = run_preprocessing_pipeline(dataset, verbose=verbose)
        
        # Save preprocessed data for later modules
        preprocess_path = config.models_dir / "preprocessed_data.pkl"
        with open(preprocess_path, 'wb') as f:
            pickle.dump(preprocessed, f)
        
        # ====================================================================
        # STEP 3: MODEL TRAINING (BASE + TUNED)
        # ====================================================================
        print("\n" + "-"*80)
        print("STEP 3/5: MODEL TRAINING (BASE + TUNED)")
        print("-"*80)
        
        models = run_models_pipeline(preprocessed, verbose=verbose)
        
        # ====================================================================
        # STEP 4: EVALUATION & COMPARISON (BASE + TUNED)
        # ====================================================================
        print("\n" + "-"*80)
        print("STEP 4/5: EVALUATION & COMPARISON (BASE + TUNED)")
        print("-"*80)
        
        results = run_evaluation_pipeline(
            models['models_base'],
            models.get('models_tuned', {}),
            preprocessed,
            verbose=verbose,
            selected_features=models.get('selected_features')
        )
        
        # ====================================================================
        # STEP 5: EDA & INTERPRETATION
        # ====================================================================
        if not skip_eda:
            print("\n" + "-"*80)
            print("STEP 5/5: EDA & INTERPRETATION")
            print("-"*80)
            
            # Load clean dataset
            df_clean = preprocessed['df_clean']
            
            # Override SHAP if requested
            if skip_shap:
                original_shap_setting = config.evaluation.enable_shap
                config.evaluation.enable_shap = False
            
            # Load scaler for LogReg SHAP
            scaler = preprocessed.get('scaler')
            
            # Use BASE and TUNED models for interpretation
            run_eda_interpretation_pipeline(
                df_clean,
                models['models_base'],
                verbose=verbose,
                selected_features=models.get('selected_features'),
                scaler=scaler,
                models_tuned=models['models_tuned']
            )
            
            # Restore SHAP setting
            if skip_shap:
                config.evaluation.enable_shap = original_shap_setting
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nExecution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        print(f"\nResults saved in:")
        print(f"  Data:   {config.data_dir}")
        print(f"  Models: {config.models_dir}")
        print(f"  Logs:   {config.metrics_dir}")
        print(f"  Plots:  {config.plots_dir}")
        
        # Best models
        df_results = results['df_results']
        
        print("\nBest Models by AUC:")
        df_results_reset = df_results.reset_index().rename(columns={'index': 'model'})
        
        base_models = df_results_reset[df_results_reset['model'].str.contains('_base', na=False)]
        tuned_models = df_results_reset[df_results_reset['model'].str.contains('_tuned', na=False)]
        
        if len(base_models) > 0:
            best_base = base_models.loc[base_models['auc'].idxmax()]
            print(f"  Best BASE:  {best_base['model']:30s} AUC = {best_base['auc']:.4f}")
        
        if len(tuned_models) > 0:
            best_tuned = tuned_models.loc[tuned_models['auc'].idxmax()]
            print(f"  Best TUNED: {best_tuned['model']:30s} AUC = {best_tuned['auc']:.4f}")
            
            if len(base_models) > 0:
                improvement = (best_tuned['auc'] - best_base['auc']) * 100
                print(f"  Improvement: {improvement:+.2f}% points")
        
        if log_file:
            print(f"\nFull log saved to: {log_file}")
        
        print("\n" + "="*80)
        print("END OF PIPELINE")
        print("="*80)
        
        # Cleanup logging
        if log_output:
            cleanup_logging()
        
        return 0
    
    except Exception as e:
        print("\n" + "="*80)
        print("PIPELINE FAILED")
        print("="*80)
        print(f"\nError: {e}")
        
        import traceback
        traceback.print_exc()
        
        # Cleanup logging
        if log_output:
            cleanup_logging()
        
        return 1


# ----------------------------------------------------------------------------
# COMMAND LINE INTERFACE
# ----------------------------------------------------------------------------

def main():
    """Main entry point with CLI options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bankruptcy Prediction ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --skip-shap        # Skip SHAP analysis (faster)
  python main.py --skip-eda         # Skip EDA & interpretation
  python main.py --no-log           # Don't save terminal output
        """
    )
    
    parser.add_argument('--skip-eda', action='store_true', help='Skip exploratory data analysis')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis (faster)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--split-year', type=int, help=f'Override split year (default: {config.data.split_year})')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    parser.add_argument('--no-log', action='store_true', help='Do not save terminal output to log file')
    
    args = parser.parse_args()
    
    if args.split_year:
        config.data.split_year = args.split_year
        print(f"Split year override: {args.split_year}")
    
    if args.no_smote:
        config.models.use_smote = False
        print("SMOTE disabled")
    
    exit_code = run_complete_pipeline(
        skip_eda=args.skip_eda,
        skip_shap=args.skip_shap,
        verbose=not args.quiet,
        log_output=not args.no_log
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
