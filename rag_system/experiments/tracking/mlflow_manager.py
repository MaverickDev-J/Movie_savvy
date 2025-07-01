# import mlflow
# import yaml
# import json
# from pathlib import Path
# import logging
# from typing import Dict, Any, Optional
# import os

# logger = logging.getLogger(__name__)

# class MLflowManager:
#     """Manages MLflow operations for experiment tracking."""
    
#     def __init__(self, config_path: str):
#         """Initialize MLflow manager with configuration.
        
#         Args:
#             config_path: Path to experiment configuration file
#         """
#         self.config_path = Path(config_path)
#         self.base_dir = self.config_path.resolve().parent.parent.parent
        
#         # Load configuration
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
        
#         # Set up MLflow with an absolute tracking URI
#         tracking_uri = "file:///teamspace/studios/this_studio/rag_system/mlruns"
#         mlflow.set_tracking_uri(tracking_uri)
        
#         # Create or get experiment
#         experiment_name = self.config["mlflow"]["experiment_name"]
#         try:
#             experiment = mlflow.get_experiment_by_name(experiment_name)
#             if experiment is None:
#                 experiment_id = mlflow.create_experiment(
#                     experiment_name,
#                     artifact_location=self.config["mlflow"]["artifact_location"]
#                 )
#                 logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
#             else:
#                 logger.info(f"Using existing experiment: {experiment_name}")
#         except Exception as e:
#             logger.error(f"Error setting up experiment: {e}")
#             raise
        
#         mlflow.set_experiment(experiment_name)
#         self.experiment_name = experiment_name
    
#     def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
#         """Start a new MLflow run. Ends previous run if still active.
        
#         Args:
#             run_name: Optional name for the run
#             tags: Optional tags to add to the run
            
#         Returns:
#             MLflow run context manager
#         """
#         if mlflow.active_run():
#             mlflow.end_run()  # Automatically end previous active run

#         tags = tags or {}
#         str_tags = {k: str(v) for k, v in tags.items()}
#         return mlflow.start_run(run_name=run_name, tags=str_tags)

    
#     def log_params(self, params: Dict[str, Any]):
#         """Log parameters to current run.
        
#         Args:
#             params: Dictionary of parameters to log
#         """
#         try:
#             # Convert all values to strings for MLflow compatibility
#             str_params = {k: str(v) for k, v in params.items()}
#             mlflow.log_params(str_params)
#             logger.debug(f"Logged parameters: {str_params}")
#         except Exception as e:
#             logger.error(f"Error logging parameters: {e}")
    
#     def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
#         """Log metrics to current run.
        
#         Args:
#             metrics: Dictionary of metrics to log
#             step: Optional step number for the metrics
#         """
#         try:
#             # Filter out non-numeric values and convert to float
#             numeric_metrics = {}
#             for k, v in metrics.items():
#                 try:
#                     if isinstance(v, (int, float)) and not isinstance(v, bool):
#                         numeric_metrics[k] = float(v)
#                     elif isinstance(v, dict):
#                         # Handle nested dictionaries (e.g., source_distribution)
#                         for sub_k, sub_v in v.items():
#                             if isinstance(sub_v, (int, float)) and not isinstance(sub_v, bool):
#                                 numeric_metrics[f"{k}_{sub_k}"] = float(sub_v)
#                 except (ValueError, TypeError):
#                     logger.warning(f"Skipping non-numeric metric: {k} = {v}")
#                     continue
            
#             if numeric_metrics:
#                 mlflow.log_metrics(numeric_metrics, step=step)
#                 logger.debug(f"Logged metrics: {list(numeric_metrics.keys())}")
#         except Exception as e:
#             logger.error(f"Error logging metrics: {e}")
    
#     def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
#         """Log an artifact file.
        
#         Args:
#             file_path: Path to the file to log
#             artifact_path: Optional path within the artifact repository
#         """
#         try:
#             if Path(file_path).exists():
#                 mlflow.log_artifact(file_path, artifact_path)
#                 logger.debug(f"Logged artifact: {file_path}")
#             else:
#                 logger.warning(f"Artifact file not found: {file_path}")
#         except Exception as e:
#             logger.error(f"Error logging artifact: {e}")
    
#     def log_dict(self, data: Dict[str, Any], artifact_path: str):
#         """Log a dictionary as a JSON artifact.
        
#         Args:
#             data: Dictionary to log
#             artifact_path: Path for the artifact (e.g., "data.json")
#         """
#         try:
#             mlflow.log_dict(data, artifact_path)
#             logger.debug(f"Logged dictionary as artifact: {artifact_path}")
#         except Exception as e:
#             logger.error(f"Error logging dictionary: {e}")
    
#     def log_text(self, text: str, artifact_path: str):
#         """Log text as an artifact.
        
#         Args:
#             text: Text to log
#             artifact_path: Path for the artifact (e.g., "response.txt")
#         """
#         try:
#             mlflow.log_text(text, artifact_path)
#             logger.debug(f"Logged text as artifact: {artifact_path}")
#         except Exception as e:
#             logger.error(f"Error logging text: {e}")
    
#     def get_experiment_info(self):
#         """Get information about the current experiment.
        
#         Returns:
#             Dictionary with experiment information
#         """
#         try:
#             experiment = mlflow.get_experiment_by_name(self.experiment_name)
#             return {
#                 "experiment_id": experiment.experiment_id,
#                 "name": experiment.name,
#                 "artifact_location": experiment.artifact_location,
#                 "lifecycle_stage": experiment.lifecycle_stage
#             }
#         except Exception as e:
#             logger.error(f"Error getting experiment info: {e}")
#             return {}
    
#     def end_run(self, status: str = "FINISHED"):
#         """End the current run.
        
#         Args:
#             status: Run status (FINISHED, FAILED, KILLED)
#         """
#         try:
#             mlflow.end_run(status=status)
#             logger.debug(f"Ended run with status: {status}")
#         except Exception as e:
#             logger.error(f"Error ending run: {e}")
    
#     def create_child_run(self, parent_run_id: str, run_name: Optional[str] = None):
#         """Create a child run under a parent run.
        
#         Args:
#             parent_run_id: ID of the parent run
#             run_name: Optional name for the child run
            
#         Returns:
#             MLflow run context manager
#         """
#         try:
#             return mlflow.start_run(run_name=run_name, nested=True)
#         except Exception as e:
#             logger.error(f"Error creating child run: {e}")
#             raise


import mlflow
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow operations for experiment tracking with proper run management."""
    
    def __init__(self, config_path: str):
        """Initialize MLflow manager with configuration.
        
        Args:
            config_path: Path to experiment configuration file
        """
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.resolve().parent.parent.parent
        self.active_runs = []  # Track active runs for proper cleanup
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set up MLflow with an absolute tracking URI
        tracking_uri = "file:///teamspace/studios/this_studio/rag_system/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment_name = self.config["mlflow"]["experiment_name"]
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=self.config["mlflow"]["artifact_location"]
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None, nested: bool = False):
        """Context manager for MLflow runs with proper cleanup.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run
            nested: Whether this is a nested run
            
        Yields:
            MLflow run object
        """
        run_obj = None
        try:
            # Convert tags to strings
            tags = tags or {}
            str_tags = {k: str(v) for k, v in tags.items()}
            
            # Start the run
            run_obj = mlflow.start_run(run_name=run_name, tags=str_tags, nested=nested)
            self.active_runs.append(run_obj.info.run_id)
            logger.debug(f"Started run: {run_name or 'unnamed'} (ID: {run_obj.info.run_id})")
            
            yield run_obj
            
        except Exception as e:
            logger.error(f"Error in run {run_name}: {e}")
            # Log the error as a metric
            try:
                mlflow.log_params({'error': str(e)[:250]})  # Limit error message length
                mlflow.log_metrics({'error_occurred': 1})
            except:
                pass
            raise
        finally:
            # Always end the run
            if run_obj and run_obj.info.run_id in self.active_runs:
                try:
                    mlflow.end_run()
                    self.active_runs.remove(run_obj.info.run_id)
                    logger.debug(f"Ended run: {run_name or 'unnamed'}")
                except Exception as e:
                    logger.error(f"Error ending run {run_name}: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Convert all values to strings for MLflow compatibility
            str_params = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    # Flatten nested dictionaries
                    for sub_k, sub_v in v.items():
                        str_params[f"{k}_{sub_k}"] = str(sub_v)
                else:
                    str_params[k] = str(v)
            
            mlflow.log_params(str_params)
            logger.debug(f"Logged parameters: {list(str_params.keys())}")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        try:
            # Filter and convert metrics
            numeric_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        # Handle NaN and infinity values
                        if str(v).lower() in ['nan', 'inf', '-inf']:
                            logger.warning(f"Skipping invalid metric value: {k} = {v}")
                            continue
                        numeric_metrics[k] = float(v)
                    elif isinstance(v, dict):
                        # Handle nested dictionaries (e.g., source_distribution)
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float)) and not isinstance(sub_v, bool):
                                if str(sub_v).lower() not in ['nan', 'inf', '-inf']:
                                    numeric_metrics[f"{k}_{sub_k}"] = float(sub_v)
                except (ValueError, TypeError, OverflowError):
                    logger.warning(f"Skipping non-numeric metric: {k} = {v}")
                    continue
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                logger.debug(f"Logged metrics: {list(numeric_metrics.keys())}")
            else:
                logger.warning("No valid numeric metrics to log")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_artifact_safe(self, file_path: str, artifact_path: Optional[str] = None):
        """Safely log an artifact file.
        
        Args:
            file_path: Path to the file to log
            artifact_path: Optional path within the artifact repository
        """
        try:
            if Path(file_path).exists():
                mlflow.log_artifact(file_path, artifact_path)
                logger.debug(f"Logged artifact: {file_path}")
            else:
                logger.warning(f"Artifact file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def log_dict_safe(self, data: Dict[str, Any], artifact_path: str):
        """Safely log a dictionary as a JSON artifact.
        
        Args:
            data: Dictionary to log
            artifact_path: Path for the artifact (e.g., "data.json")
        """
        try:
            # Clean the data to ensure it's JSON serializable
            clean_data = self._clean_dict_for_json(data)
            mlflow.log_dict(clean_data, artifact_path)
            logger.debug(f"Logged dictionary as artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging dictionary: {e}")
    
    def _clean_dict_for_json(self, data: Any) -> Any:
        """Clean data to make it JSON serializable."""
        if isinstance(data, dict):
            return {k: self._clean_dict_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_dict_for_json(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            # Handle NaN and infinity
            if isinstance(data, float) and str(data).lower() in ['nan', 'inf', '-inf']:
                return str(data)
            return data
        else:
            return str(data)
    
    def cleanup_all_runs(self):
        """Clean up any remaining active runs."""
        for run_id in self.active_runs.copy():
            try:
                with mlflow.start_run(run_id=run_id):
                    mlflow.end_run()
                self.active_runs.remove(run_id)
                logger.debug(f"Cleaned up orphaned run: {run_id}")
            except Exception as e:
                logger.error(f"Error cleaning up run {run_id}: {e}")
        
        # Force end any remaining active run
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.error(f"Error force-ending active run: {e}")
    
    def get_experiment_info(self):
        """Get information about the current experiment.
        
        Returns:
            Dictionary with experiment information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            return {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage
            }
        except Exception as e:
            logger.error(f"Error getting experiment info: {e}")
            return {}