import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from .rag_pipeline import HybridRAGPipeline, ChatResponse

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for RAG system."""
    average_confidence: float
    retrieval_coverage: float
    response_relevance: float
    total_queries: int
    rag_mode_usage: float
    chat_mode_usage: float

class RAGEvaluator:
    """Evaluates hybrid RAG system performance."""
    
    def __init__(self, rag_pipeline: HybridRAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def evaluate_retrieval_quality(
        self, 
        test_queries: List[str],
        expected_sources: List[List[str]] = None
    ) -> EvaluationMetrics:
        """Evaluate retrieval quality with test queries."""
        try:
            logger.info(f"Evaluating hybrid RAG system with {len(test_queries)} queries")
            
            responses = []
            confidence_scores = []
            rag_mode_count = 0
            chat_mode_count = 0
            
            for i, query in enumerate(test_queries):
                try:
                    response = self.rag_pipeline.chat(query, force_mode="rag")  # Force RAG for evaluation
                    responses.append(response)
                    confidence_scores.append(response.confidence_score)
                    
                    if response.mode == "rag":
                        rag_mode_count += 1
                    else:
                        chat_mode_count += 1
                    
                    logger.info(f"Query {i+1}/{len(test_queries)} processed")
                    
                except Exception as e:
                    logger.error(f"Error processing query {i+1}: {str(e)}")
                    confidence_scores.append(0.0)
                    chat_mode_count += 1  # Count errors as chat mode
            
            # Calculate metrics
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Retrieval coverage (percentage of queries that found relevant docs)
            successful_retrievals = sum(1 for r in responses if r.sources)
            retrieval_coverage = successful_retrievals / len(test_queries) if test_queries else 0.0
            
            # Response relevance (percentage of high-confidence responses)
            high_confidence_responses = sum(1 for score in confidence_scores if score > 0.7)
            response_relevance = high_confidence_responses / len(test_queries) if test_queries else 0.0
            
            # Mode usage percentages
            total_responses = len(test_queries)
            rag_mode_usage = rag_mode_count / total_responses if total_responses > 0 else 0.0
            chat_mode_usage = chat_mode_count / total_responses if total_responses > 0 else 0.0
            
            metrics = EvaluationMetrics(
                average_confidence=round(avg_confidence, 3),
                retrieval_coverage=round(retrieval_coverage, 3),
                response_relevance=round(response_relevance, 3),
                total_queries=len(test_queries),
                rag_mode_usage=round(rag_mode_usage, 3),
                chat_mode_usage=round(chat_mode_usage, 3)
            )
            
            logger.info(f"Evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def evaluate_mode_detection(
        self, 
        test_cases: List[Tuple[str, str]]  # (query, expected_mode)
    ) -> Dict[str, float]:
        """Evaluate the accuracy of mode detection."""
        try:
            correct_predictions = 0
            total_cases = len(test_cases)
            
            for query, expected_mode in test_cases:
                try:
                    response = self.rag_pipeline.chat(query)
                    if response.mode == expected_mode:
                        correct_predictions += 1
                except Exception as e:
                    logger.error(f"Error evaluating mode detection for query: {query}")
            
            accuracy = correct_predictions / total_cases if total_cases > 0 else 0.0
            
            return {
                "mode_detection_accuracy": round(accuracy, 3),
                "correct_predictions": correct_predictions,
                "total_cases": total_cases
            }
            
        except Exception as e:
            logger.error(f"Error evaluating mode detection: {str(e)}")
            return {"mode_detection_accuracy": 0.0, "correct_predictions": 0, "total_cases": 0}
    
    def generate_evaluation_report(
        self, 
        test_queries: List[str]
    ) -> Dict[str, Any]:
        """Generate detailed evaluation report."""
        try:
            metrics = self.evaluate_retrieval_quality(test_queries)
            
            # Get system stats
            system_stats = self.rag_pipeline.get_system_stats()
            
            # Sample responses for analysis
            sample_responses = []
            for query in test_queries[:3]:  # First 3 queries as samples
                try:
                    response = self.rag_pipeline.chat(query, force_mode="rag")
                    sample_responses.append({
                        "query": query,
                        "answer": response.answer[:200] + "...",
                        "confidence": response.confidence_score,
                        "sources_count": len(response.sources),
                        "mode": response.mode,
                        "reasoning": response.reasoning
                    })
                except Exception as e:
                    logger.error(f"Error getting sample response: {str(e)}")
            
            # Test mode detection with some examples
            mode_test_cases = [
                ("Hello, how are you?", "chat"),
                ("What is the weather today?", "chat"),
                ("What does the document say about this topic?", "rag"),
                ("Summarize the main findings", "rag"),
                ("Tell me a joke", "chat")
            ]
            
            mode_detection_results = self.evaluate_mode_detection(mode_test_cases)
            
            report = {
                "evaluation_metrics": metrics.__dict__,
                "mode_detection": mode_detection_results,
                "system_stats": system_stats,
                "sample_responses": sample_responses,
                "recommendations": self._generate_recommendations(metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if metrics.average_confidence < 0.6:
            recommendations.append("Consider improving document quality or chunking strategy")
        
        if metrics.retrieval_coverage < 0.8:
            recommendations.append("Add more diverse documents to knowledge base")
        
        if metrics.response_relevance < 0.7:
            recommendations.append("Tune similarity threshold or embedding model")
        
        if metrics.rag_mode_usage < 0.5 and metrics.total_queries > 0:
            recommendations.append("Consider adjusting mode detection logic for better RAG utilization")
        
        if not recommendations:
            recommendations.append("Hybrid system performance looks good!")
        
        return recommendations