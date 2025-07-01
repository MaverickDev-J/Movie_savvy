# import json
import logging
import os
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sqlite3
import numpy as np
import json

logger = logging.getLogger(__name__)

class DataLogger:
    """Handles logging of experiment data for the rag_system.
    
    Supports both file-based (JSON) and database (SQLite) logging.
    Integrates with MLflow for artifact logging.
    """

    def __init__(self, 
                 log_dir: str = "logs",
                 log_format: str = "json",
                 db_name: str = "experiment_data.db",
                 mlflow_manager: Optional[Any] = None):
        """Initialize the DataLogger.
        
        Args:
            log_dir: Directory to store log files
            log_format: Logging format ('json' or 'sqlite')
            db_name: Name of the SQLite database file (if using SQLite)
            mlflow_manager: Optional MLflow manager for artifact logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Check write permissions
        if not os.access(self.log_dir, os.W_OK):
            raise PermissionError(f"Cannot write to {self.log_dir}")
            
        self.log_format = log_format.lower()
        self.mlflow_manager = mlflow_manager
        self.lock = threading.Lock()
        
        if self.log_format == "sqlite":
            self.db_path = self.log_dir / db_name
            self._setup_database()
        
        logger.info(f"DataLogger initialized with format: {self.log_format}")

    def _setup_database(self):
        """Set up the SQLite database for logging."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiment_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        query TEXT,
                        response TEXT,
                        context TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
            logger.debug(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error setting up SQLite database: {e}")
            raise

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-serializable types."""
        try:
            if isinstance(obj, np.floating):  # Handle numpy float types (e.g., float32)
                return float(obj)
            if isinstance(obj, np.ndarray):  # Handle numpy arrays
                return obj.tolist()
            if isinstance(obj, dict):  # Recursively process dictionaries
                return {k: self._convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):  # Recursively process lists
                return [self._convert_to_serializable(item) for item in obj]
            if isinstance(obj, (np.integer, np.bool_)):  # Handle numpy integers and booleans
                return obj.item()
            return obj
        except Exception as e:
            logger.warning(f"Cannot serialize {type(obj)}, converting to string: {e}")
            return str(obj)

    def log_data(self, 
                 query: str, 
                 response: str, 
                 context: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Log experiment data.
        
        Args:
            query: The user's query
            response: The generated response
            context: The context used for generation
            metadata: Optional additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context": context,
            "metadata": metadata or {}
        }
        
        # Convert log_entry to JSON-serializable format
        try:
            serializable_log_entry = self._convert_to_serializable(log_entry)
        except Exception as e:
            logger.error(f"Error converting log entry to serializable format: {e}")
            return

        with self.lock:
            if self.log_format == "json":
                self._log_to_json(serializable_log_entry)
            elif self.log_format == "sqlite":
                self._log_to_sqlite(serializable_log_entry)
            else:
                logger.warning(f"Unsupported log format: {self.log_format}")
            
            # Log to MLflow if manager is provided (moved inside lock for thread safety)
            if self.mlflow_manager:
                try:
                    artifact_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_log.json"
                    artifact_file_path = self.log_dir / artifact_filename
                    with open(artifact_file_path, "w") as f:
                        json.dump(serializable_log_entry, f, indent=2)
                    self.mlflow_manager.log_artifact_safe(str(artifact_file_path))
                    # Log additional MLflow metrics and params
                    self.mlflow_manager.log_params({
                    f'query_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}': query[:50],  # Added microseconds for uniqueness
                    f'response_length_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}': len(response)
                    })
                    metrics = serializable_log_entry.get('metadata', {}).get('metrics', {})
                    self.mlflow_manager.log_metrics(metrics)
                except Exception as e:
                    logger.error(f"Error logging to MLflow: {e}")

    def _log_to_json(self, log_entry: Dict[str, Any]):
        """Log data to a JSON file."""
        log_file = self.log_dir / f"{log_entry['timestamp'].replace(':', '-')}_log.json"
        try:
            with open(log_file, "w") as f:
                json.dump(log_entry, f, indent=2)
            logger.debug(f"Logged data to {log_file}")
        except Exception as e:
            logger.error(f"Error writing to JSON log: {e}")

    def _log_to_sqlite(self, log_entry: Dict[str, Any]):
        """Log data to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO experiment_logs (timestamp, query, response, context, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    log_entry["timestamp"],
                    log_entry["query"],
                    log_entry["response"],
                    log_entry["context"],
                    json.dumps(log_entry["metadata"])
                ))
                conn.commit()
            logger.debug(f"Logged data to SQLite: {self.db_path}")
        except Exception as e:
            logger.error(f"Error writing to SQLite log: {e}")

# Example usage:
# logger = DataLogger(log_dir="experiment_logs", log_format="json", mlflow_manager=mlflow_manager)
# logger.log_data("What is the capital of France?", "The capital of France is Paris.", "France is a country in Europe.", {"experiment_id": "123"})