#!/usr/bin/env python3

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# ✅ Add project root (/teamspace/studios/this_studio) to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Adds rag_system/
# ✅ Optionally also add the current directory (experiments/)
sys.path.insert(0, str(Path(__file__).parent))  # Adds rag_system/experiments/

from tracking.experiment_runner import ExperimentRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Run RAG system experiments")
    parser.add_argument(
        '--experiment', 
        required=True,
        choices=['retrieval_optimization', 'generation_optimization', 'hybrid_optimization', 'all'],
        help="Type of experiment to run"
    )
    parser.add_argument(
        '--config',
        default=None,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Perform a dry run without executing experiments"
    )
    
    args = parser.parse_args()
    
    # Set config path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / 'config' / 'experiment_config.yaml'
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    logger.info(f"Using configuration file: {config_path}")
    
    try:
        runner = ExperimentRunner(str(config_path))
        logger.info("ExperimentRunner initialized successfully")
        
        if args.dry_run:
            logger.info("Dry run mode - experiments will not be executed")
            return 0
        
        if args.experiment == 'retrieval_optimization':
            logger.info("Starting retrieval optimization experiments")
            await runner.run_retrieval_optimization()
        elif args.experiment == 'generation_optimization':
            logger.info("Starting generation optimization experiments")
            await runner.run_generation_optimization()
        elif args.experiment == 'hybrid_optimization':
            logger.info("Starting hybrid optimization experiments")
            await runner.run_hybrid_optimization()
        elif args.experiment == 'all':
            logger.info("Starting all experiments")
            await runner.run_retrieval_optimization()
            await runner.run_generation_optimization()
            await runner.run_hybrid_optimization()
        
        logger.info("All experiments completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error running experiments: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
