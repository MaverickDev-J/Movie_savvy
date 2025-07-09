import asyncio
import sys
from itertools import product
from pathlib import Path
import logging
from typing import Dict, Any, List
import time
import psutil
import traceback
import signal

# ✅ Fix BASE_DIR to point to the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Should resolve to /teamspace/studios/this_studio
sys.path.insert(0, str(BASE_DIR))

from tracking.mlflow_manager import MLflowManager
from utils.data_logger import DataLogger
from utils.query_analyzer import QueryAnalyzer
from metrics.retrieval_metrics import RetrievalMetrics
from metrics.generation_metrics import GenerationMetrics
from metrics.system_metrics import SystemMetrics

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Manages the execution of experiments for the rag_system with proper cleanup."""
    
    def __init__(self, config_path: str):
        """Initialize the ExperimentRunner.
        
        Args:
            config_path: Path to the experiment configuration file
        """
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            import yaml
            self.config = yaml.safe_load(f)
        
        self.mlflow_manager = MLflowManager(str(self.config_path))
        self.data_logger = DataLogger(
            log_dir=str(Path('experiments') / 'logs'),
            log_format='json',
            mlflow_manager=self.mlflow_manager
        )
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.system_metrics = SystemMetrics()
        
        # Initialize RAG components
        self._initialize_rag_components()
        
        # Setup signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.mlflow_manager.cleanup_all_runs()
        sys.exit(0)
    
    def _initialize_rag_components(self):
        """Initialize RAG pipeline components."""
        try:
            # Import your RAG components
            from rag_system.scripts.run_rag import get_generator, retrieve_vector, get_web_results_async, analyze_query_intent
            self.get_generator = get_generator
            self.retrieve_vector = retrieve_vector
            self.get_web_results_async = get_web_results_async
            self.analyze_query_intent = analyze_query_intent
            
            # Initialize generator
            self.generator = self.get_generator()
            logger.info("RAG components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    async def run_retrieval_optimization(self):
        """Run retrieval parameter optimization experiments."""
        logger.info("Starting retrieval optimization experiments")
        await self._run_experiment('retrieval_optimization')
    
    async def run_generation_optimization(self):
        """Run generation parameter optimization experiments."""
        logger.info("Starting generation optimization experiments")
        await self._run_experiment('generation_optimization')
    
    async def run_hybrid_optimization(self):
        """Run hybrid pipeline optimization experiments."""
        logger.info("Starting hybrid optimization experiments")
        await self._run_experiment('hybrid_optimization')
    
    async def _run_experiment(self, experiment_type: str):
        """Run a specific type of experiment with proper MLflow management.
        
        Args:
            experiment_type: Type of experiment to run
        """
        try:
            experiment_config = self.config['experiments'][experiment_type]
            parameters = experiment_config['parameters']
            test_queries = self.config['test_queries']
            
            # Generate parameter combinations
            param_keys = list(parameters.keys())
            param_values = [parameters[k] for k in param_keys]
            param_combinations = list(product(*param_values))
            
            logger.info(f"Running {len(param_combinations)} parameter combinations for {experiment_type}")
            
            # Main experiment run
            with self.mlflow_manager.start_run(
                run_name=f"{experiment_type}_experiment",
                tags={
                    'experiment_type': experiment_type,
                    'total_combinations': len(param_combinations),
                    'total_queries': sum(len(queries) for queries in test_queries.values())
                }
            ) as main_run:
                
                # Log experiment configuration
                self.mlflow_manager.log_params({
                    'experiment_type': experiment_type,
                    'parameter_combinations': len(param_combinations),
                    'query_types': list(test_queries.keys())
                })
                
                all_combination_metrics = []
                
                for i, param_set in enumerate(param_combinations):
                    param_dict = dict(zip(param_keys, param_set))
                    
                    logger.info(f"Running parameter combination {i+1}/{len(param_combinations)}: {param_dict}")
                    
                    try:
                        # Run all queries for this parameter set
                        combination_metrics = await self._run_parameter_combination(
                            param_dict, test_queries, experiment_type, i+1
                        )
                        
                        if combination_metrics:
                            all_combination_metrics.append({
                                'params': param_dict,
                                'metrics': combination_metrics
                            })
                        
                    except Exception as e:
                        logger.error(f"Error in parameter combination {i+1}: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                # Log overall experiment metrics
                if all_combination_metrics:
                    overall_metrics = self._calculate_overall_metrics(all_combination_metrics)
                    self.mlflow_manager.log_metrics(overall_metrics)
                    
                    # Log summary artifact
                    summary = {
                        'experiment_type': experiment_type,
                        'total_combinations': len(param_combinations),
                        'successful_combinations': len(all_combination_metrics),
                        'best_combination': self._find_best_combination(all_combination_metrics),
                        'parameter_analysis': self._analyze_parameter_impact(all_combination_metrics)
                    }
                    self.mlflow_manager.log_dict_safe(summary, 'experiment_summary.json')
                
                logger.info(f"Completed {experiment_type} experiment with {len(all_combination_metrics)} successful combinations")
        
        except Exception as e:
            logger.error(f"Critical error in experiment {experiment_type}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _run_parameter_combination(self, param_dict: Dict[str, Any], test_queries: Dict[str, List[str]], 
                                       experiment_type: str, combination_num: int) -> Dict[str, float]:
        """Run all queries for a specific parameter combination.
        
        Args:
            param_dict: Parameters for this combination
            test_queries: Test queries organized by type
            experiment_type: Type of experiment
            combination_num: Combination number for naming
            
        Returns:
            Aggregated metrics for this combination
        """
        with self.mlflow_manager.start_run(
            run_name=f"combo_{combination_num}_{experiment_type}",
            tags={
                'combination_number': combination_num,
                'experiment_type': experiment_type,
                **param_dict
            },
            nested=True
        ) as combo_run:
            
            # Log parameters for this combination
            self.mlflow_manager.log_params(param_dict)
            
            all_query_metrics = []
            
            for query_type, queries in test_queries.items():
                for query_idx, query in enumerate(queries):
                    try:
                        metrics = await self._run_single_query(
                            query=query,
                            query_type=query_type,
                            parameters=param_dict,
                            experiment_type=experiment_type,
                            query_idx=query_idx
                        )
                        
                        if metrics and 'error_occurred' not in metrics:
                            all_query_metrics.append(metrics)
                        
                    except Exception as e:
                        logger.error(f"Error processing query '{query[:50]}...': {e}")
                        continue
            
            # Calculate and log aggregated metrics for this combination
            if all_query_metrics:
                aggregated_metrics = self._aggregate_metrics(all_query_metrics)
                self.mlflow_manager.log_metrics(aggregated_metrics)
                
                # Log query details
                query_summary = {
                    'total_queries': len(all_query_metrics),
                    'successful_queries': len(all_query_metrics),
                    'parameters': param_dict,
                    'avg_total_time': aggregated_metrics.get('avg_total_time', 0),
                    'avg_response_length': aggregated_metrics.get('avg_response_length', 0)
                }
                self.mlflow_manager.log_dict_safe(query_summary, f'combination_{combination_num}_summary.json')
                
                return aggregated_metrics
            else:
                logger.warning(f"No successful queries for combination {combination_num}")
                return {}
    
    # async def _run_single_query(self, query: str, query_type: str, parameters: Dict[str, Any], 
    #                           experiment_type: str, query_idx: int) -> Dict[str, float]:
    #     """Run a single query with given parameters and return metrics.
        
    #     Args:
    #         query: The query to process
    #         query_type: Type of query (factual, opinion, etc.)
    #         parameters: Parameters to use for this run
    #         experiment_type: Type of experiment being run
    #         query_idx: Index of the query
            
    #     Returns:
    #         Dictionary of calculated metrics
    #     """
    #     start_time = time.time()
        
    #     try:
    #         # Analyze query
    #         query_analysis = self.analyze_query_intent(query)
            
    #         # Set parameters based on experiment type
    #         rag_params = self._build_rag_parameters(parameters, experiment_type)
            
    #         # Measure retrieval
    #         retrieval_start = time.time()
    #         vector_chunks = await self.retrieve_vector(query, rag_params['top_k'])
    #         retrieval_time = time.time() - retrieval_start
            
    #         # Measure web search if needed
    #         web_start = time.time()
    #         web_context = []
    #         if query_analysis.get('needs_web_search', False):
    #             web_context = await self.get_web_results_async(query, query_analysis)
    #         web_time = time.time() - web_start
            
    #         # Prepare contexts
    #         vector_context = [chunk['text'] for chunk in vector_chunks] if vector_chunks else []
            
    #         # Measure generation
    #         generation_start = time.time()
    #         response = self.generator.generate(
    #             query=query,
    #             local_context=vector_context,
    #             web_context=web_context,
    #             query_analysis=query_analysis,
    #             **rag_params
    #         )
    #         generation_time = time.time() - generation_start
            
    #         total_time = time.time() - start_time
            
    #         # Calculate metrics
    #         metrics = {}
            
    #         # Retrieval metrics
    #         if vector_chunks:
    #             retrieval_metrics = self.retrieval_metrics.calculate_metrics(
    #                 query=query,
    #                 retrieved_chunks=vector_chunks,
    #                 retrieval_time=retrieval_time
    #             )
    #             metrics.update(retrieval_metrics)
            
    #         # Generation metrics
    #         generation_metrics = self.generation_metrics.calculate_metrics(
    #             query=query,
    #             response=response,
    #             context=vector_context + web_context,
    #             generation_time=generation_time
    #         )
    #         metrics.update(generation_metrics)
            
    #         # System metrics
    #         system_metrics = self.system_metrics.calculate_metrics(
    #             total_time=total_time,
    #             retrieval_time=retrieval_time,
    #             generation_time=generation_time,
    #             web_time=web_time
    #         )
    #         metrics.update(system_metrics)
            
    #         # Log the interaction
    #         self.data_logger.log_data(
    #             query=query,
    #             response=response,
    #             context=str(vector_context + web_context),
    #             metadata={
    #                 'query_type': query_type,
    #                 'parameters': parameters,
    #                 'experiment_type': experiment_type,
    #                 'metrics': metrics,
    #                 'query_idx': query_idx
    #             }
    #         )
            
    #         return metrics
            
    #     except Exception as e:
    #         logger.error(f"Error in _run_single_query: {e}")
    #         logger.error(traceback.format_exc())
    #         return {'error_occurred': 1.0, 'total_time': time.time() - start_time}



    async def _run_single_query(self, query: str, query_type: str, parameters: Dict[str, Any], 
                            experiment_type: str, query_idx: int) -> Dict[str, float]:
        start_time = time.time()
        
        try:
            # Analyze query
            query_analysis = self.analyze_query_intent(query)
            
            # Set parameters based on experiment type
            rag_params = self._build_rag_parameters(parameters, experiment_type)
            
            # Measure retrieval
            retrieval_start = time.time()
            vector_chunks = await self.retrieve_vector(query, rag_params['top_k'])
            retrieval_time = time.time() - retrieval_start
            
            # Measure web search if needed
            web_start = time.time()
            web_context = []
            if query_analysis.get('needs_web_search', False):
                web_context = await self.get_web_results_async(query, query_analysis)
            web_time = time.time() - web_start
            
            # Prepare contexts
            vector_context = [chunk['text'] for chunk in vector_chunks] if vector_chunks else []
            
            # Measure generation
            generation_start = time.time()
            generate_params = {
                'query': query,
                'local_context': vector_context,
                'web_context': web_context,
                'query_analysis': query_analysis,
                'max_new_tokens': rag_params.get('max_new_tokens'),
                'temperature': rag_params.get('temperature'),
                'top_k': rag_params.get('top_k_sampling')
            }
            response = self.generator.generate(**generate_params)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Calculate metrics (rest of the method remains unchanged)
            metrics = {}
            
            if vector_chunks:
                retrieval_metrics = self.retrieval_metrics.calculate_metrics(
                    query=query,
                    retrieved_chunks=vector_chunks,
                    retrieval_time=retrieval_time
                )
                metrics.update(retrieval_metrics)
            
            generation_metrics = self.generation_metrics.calculate_metrics(
                query=query,
                response=response,
                context=vector_context + web_context,
                generation_time=generation_time
            )
            metrics.update(generation_metrics)
            
            system_metrics = self.system_metrics.calculate_metrics(
                total_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                web_time=web_time
            )
            metrics.update(system_metrics)
            
            self.data_logger.log_data(
                query=query,
                response=response,
                context=str(vector_context + web_context),
                metadata={
                    'query_type': query_type,
                    'parameters': parameters,
                    'experiment_type': experiment_type,
                    'metrics': metrics,
                    'query_idx': query_idx
                }
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in _run_single_query: {e}")
            logger.error(traceback.format_exc())
            return {'error_occurred': 1.0, 'total_time': time.time() - start_time}





    
    def _build_rag_parameters(self, parameters: Dict[str, Any], experiment_type: str) -> Dict[str, Any]:
        """Build RAG parameters based on experiment type and parameters."""
        default_params = self.config['default_parameters']
        
        if experiment_type == 'retrieval_optimization':
            return {
                'top_k': parameters.get('top_k', default_params['retrieval']['top_k']),
                'similarity_threshold': parameters.get('similarity_threshold', default_params['retrieval']['similarity_threshold']),
                'max_new_tokens': default_params['generation']['max_new_tokens'],
                'temperature': default_params['generation']['temperature'],
                'top_k_sampling': default_params['generation']['top_k_sampling'],
                'repetition_penalty': default_params['generation']['repetition_penalty']
            }
        elif experiment_type == 'generation_optimization':
            return {
                'top_k': default_params['retrieval']['top_k'],
                'similarity_threshold': default_params['retrieval']['similarity_threshold'],
                'max_new_tokens': parameters.get('max_new_tokens', default_params['generation']['max_new_tokens']),
                'temperature': parameters.get('temperature', default_params['generation']['temperature']),
                'top_k_sampling': parameters.get('top_k_sampling', default_params['generation']['top_k_sampling']),
                'repetition_penalty': parameters.get('repetition_penalty', default_params['generation']['repetition_penalty'])
            }
        else:  # hybrid_optimization
            return {
                'top_k': default_params['retrieval']['top_k'],
                'similarity_threshold': default_params['retrieval']['similarity_threshold'],
                'max_new_tokens': default_params['generation']['max_new_tokens'],
                'temperature': default_params['generation']['temperature'],
                'top_k_sampling': default_params['generation']['top_k_sampling'],
                'repetition_penalty': default_params['generation']['repetition_penalty'],
                'web_search_threshold': parameters.get('web_search_threshold', 0.5),
                'context_mixing_ratio': parameters.get('context_mixing_ratio', 0.5)
            }
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple runs."""
        if not metrics_list:
            return {}
        
        # Filter out error runs
        valid_metrics = [m for m in metrics_list if 'error_occurred' not in m]
        
        if not valid_metrics:
            return {'error_rate': 1.0}
        
        aggregated = {}
        
        # Calculate statistics for all numeric metrics
        all_keys = set()
        for m in valid_metrics:
            all_keys.update(m.keys())
        
        for key in all_keys:
            values = [m.get(key, 0) for m in valid_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                aggregated[f"avg_{key}"] = sum(values) / len(values)
                aggregated[f"min_{key}"] = min(values)
                aggregated[f"max_{key}"] = max(values)
                aggregated[f"std_{key}"] = (sum((x - aggregated[f"avg_{key}"])**2 for x in values) / len(values))**0.5
        
        # Add summary statistics
        aggregated['error_rate'] = (len(metrics_list) - len(valid_metrics)) / len(metrics_list)
        aggregated['total_runs'] = len(metrics_list)
        aggregated['successful_runs'] = len(valid_metrics)
        
        return aggregated
    
    def _calculate_overall_metrics(self, all_combination_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate overall experiment metrics."""
        if not all_combination_metrics:
            return {}
        
        # Extract all metrics
        all_metrics = [combo['metrics'] for combo in all_combination_metrics]
        
        # Find best performing combination
        best_combo = max(all_combination_metrics, 
                        key=lambda x: x['metrics'].get('avg_response_relevance', 0))
        
        overall = {}
        
        # Calculate statistics across combinations
        for key in ['avg_total_time', 'avg_response_length', 'avg_response_relevance']:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                overall[f"overall_{key}"] = sum(values) / len(values)
                overall[f"best_{key}"] = best_combo['metrics'].get(key, 0)
        
        overall['total_combinations_tested'] = len(all_combination_metrics)
        
        return overall
    
    def _find_best_combination(self, all_combination_metrics: List[Dict]) -> Dict:
        """Find the best performing parameter combination."""
        if not all_combination_metrics:
            return {}
        
        # Score combinations based on multiple metrics (customize as needed)
        def score_combination(combo):
            metrics = combo['metrics']
            score = 0
            # Higher is better
            score += metrics.get('avg_response_relevance', 0) * 0.4
            score += metrics.get('avg_response_coherence', 0) * 0.3
            # Lower is better (invert)
            score -= metrics.get('avg_total_time', 0) * 0.1
            score += metrics.get('avg_fluency_score', 0) * 0.2
            return score
        
        best_combo = max(all_combination_metrics, key=score_combination)
        return {
            'parameters': best_combo['params'],
            'score': score_combination(best_combo),
            'key_metrics': {
                'avg_response_relevance': best_combo['metrics'].get('avg_response_relevance', 0),
                'avg_total_time': best_combo['metrics'].get('avg_total_time', 0),
                'avg_response_length': best_combo['metrics'].get('avg_response_length', 0)
            }
        }
    
    def _analyze_parameter_impact(self, all_combination_metrics: List[Dict]) -> Dict:
        """Analyze the impact of different parameters on performance."""
        if not all_combination_metrics:
            return {}
        
        # Group by parameter values and calculate average performance
        parameter_impact = {}
        
        # Get all parameter names
        param_names = set()
        for combo in all_combination_metrics:
            param_names.update(combo['params'].keys())
        
        for param_name in param_names:
            param_values = {}
            for combo in all_combination_metrics:
                param_value = combo['params'].get(param_name)
                if param_value is not None:
                    if param_value not in param_values:
                        param_values[param_value] = []
                    param_values[param_value].append(combo['metrics'].get('avg_response_relevance', 0))
            
            # Calculate average performance for each parameter value
            parameter_impact[param_name] = {}
            for value, scores in param_values.items():
                parameter_impact[param_name][str(value)] = sum(scores) / len(scores) if scores else 0
        
        return parameter_impact
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.mlflow_manager.cleanup_all_runs()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()