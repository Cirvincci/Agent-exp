"""
SciDetective Agent Package

A comprehensive scientific research assistant that helps analyze essays,
detect blind spots, generate research ideas, and search scientific literature.
"""

from .agent import SciDetectiveAgent, AgentConfig, AgentResponse
from .text_analysis import TextAnalyzer, AnalysisResult, BlindSpot, EssaySection
from .idea_generator import IdeaGenerator, ResearchIdea, ResearchGap
from .web_searcher import ScientificWebSearcher, Paper, SearchResult
from .visualization import VisualizationGenerator, VisualizationConfig

__version__ = "1.0.0"
__author__ = "SciDetective Team"

# Main exports
__all__ = [
    'SciDetectiveAgent',
    'AgentConfig',
    'AgentResponse',
    'TextAnalyzer',
    'IdeaGenerator',
    'ScientificWebSearcher',
    'VisualizationGenerator',
    'AnalysisResult',
    'ResearchIdea',
    'Paper'
]