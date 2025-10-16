"""
Main Agent Class for SciDetective_Agent

This is the core agent that orchestrates all the modules and provides
a unified interface for scientific essay analysis and research assistance.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# Import our custom modules
try:
    # Try relative imports first
    from .text_analysis import TextAnalyzer, AnalysisResult
    from .idea_generator import IdeaGenerator, ResearchIdea, ResearchGap
    from .web_searcher import ScientificWebSearcher, Paper
    from .visualization import VisualizationGenerator, VisualizationConfig
except ImportError:
    # Fall back to absolute imports
    from text_analysis import TextAnalyzer, AnalysisResult
    from idea_generator import IdeaGenerator, ResearchIdea, ResearchGap
    from web_searcher import ScientificWebSearcher, Paper
    from visualization import VisualizationGenerator, VisualizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for the SciDetective Agent"""
    max_search_results: int = 20
    max_ideas_generated: int = 10
    visualization_style: str = 'scientific'
    enable_interactive_viz: bool = False
    auto_save_results: bool = True
    output_directory: str = 'outputs'

@dataclass
class AgentResponse:
    """Standard response format for agent operations"""
    success: bool
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class SciDetectiveAgent:
    """
    Main SciDetective Agent class that coordinates all analysis and research tasks
    """

    def __init__(self, config: AgentConfig = None):
        """Initialize the SciDetective Agent"""
        self.config = config or AgentConfig()

        # Initialize all modules
        self.text_analyzer = TextAnalyzer()
        self.idea_generator = IdeaGenerator()
        self.web_searcher = ScientificWebSearcher()
        self.visualizer = VisualizationGenerator(
            VisualizationConfig(
                style=self.config.visualization_style,
                interactive=self.config.enable_interactive_viz
            )
        )

        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)

        # Session data
        self.session_data = {
            'last_analysis': None,
            'last_search_results': None,
            'generated_ideas': [],
            'visualizations': {}
        }

        logger.info("SciDetective Agent initialized successfully")

    def analyze_essay(self, text: str = None, file_path: str = None) -> AgentResponse:
        """
        Analyze a scientific essay for structure, blind spots, and quality

        Args:
            text: Essay text content
            file_path: Path to essay file (PDF, DOCX, TXT)

        Returns:
            AgentResponse with analysis results
        """
        try:
            logger.info("Starting essay analysis...")

            # Load document if file path provided
            if file_path and not text:
                text = self.text_analyzer.load_document(file_path)
                logger.info(f"Loaded document from {file_path}")

            if not text:
                return AgentResponse(
                    success=False,
                    message="No text provided for analysis",
                    error="Either text content or file_path must be provided"
                )

            # Perform analysis
            analysis_result = self.text_analyzer.analyze_essay(text)

            # Store in session
            self.session_data['last_analysis'] = analysis_result

            # Prepare response data
            response_data = {
                'analysis_summary': analysis_result.summary,
                'overall_score': analysis_result.overall_score,
                'structure_quality': analysis_result.structure_quality,
                'argument_strength': analysis_result.argument_strength,
                'evidence_quality': analysis_result.evidence_quality,
                'sections_detected': len(analysis_result.sections),
                'blind_spots_found': len(analysis_result.blind_spots),
                'sections': [
                    {
                        'title': section.title,
                        'length': len(section.content),
                        'confidence': section.confidence
                    }
                    for section in analysis_result.sections
                ],
                'blind_spots': [
                    {
                        'type': bs.type,
                        'description': bs.description,
                        'location': bs.location,
                        'severity': bs.severity,
                        'suggestions': bs.suggestions
                    }
                    for bs in analysis_result.blind_spots
                ]
            }

            # Auto-save if enabled
            if self.config.auto_save_results:
                self._save_analysis_results(analysis_result)

            return AgentResponse(
                success=True,
                message=f"Essay analysis completed. Found {len(analysis_result.sections)} sections and {len(analysis_result.blind_spots)} areas for improvement.",
                data=response_data
            )

        except Exception as e:
            logger.error(f"Error in essay analysis: {str(e)}")
            return AgentResponse(
                success=False,
                message="Failed to analyze essay",
                error=str(e)
            )

    async def search_literature(self, query: str, sources: List[str] = None) -> AgentResponse:
        """
        Search scientific literature for relevant papers

        Args:
            query: Search query
            sources: List of sources to search (arxiv, semantic_scholar, pubmed)

        Returns:
            AgentResponse with search results
        """
        try:
            logger.info(f"Searching literature for: {query}")

            # Perform literature search
            search_results = self.web_searcher.search_literature(query, self.config.max_search_results)

            # Store in session
            self.session_data['last_search_results'] = search_results

            # Prepare response data
            response_data = {
                'query': query,
                'total_papers_found': search_results['total_found'],
                'sources_searched': search_results['sources_searched'],
                'papers': [
                    {
                        'title': paper.title,
                        'authors': paper.authors,
                        'abstract': paper.abstract[:300] + '...' if len(paper.abstract) > 300 else paper.abstract,
                        'url': paper.url,
                        'publication_date': paper.publication_date,
                        'journal': paper.journal,
                        'citation_count': paper.citation_count,
                        'source': paper.source
                    }
                    for paper in search_results['papers']
                ],
                'related_keywords': search_results['related_keywords'][:20],
                'search_metadata': search_results['search_metadata']
            }

            return AgentResponse(
                success=True,
                message=f"Literature search completed. Found {search_results['total_found']} papers from {len(search_results['sources_searched'])} sources.",
                data=response_data
            )

        except Exception as e:
            logger.error(f"Error in literature search: {str(e)}")
            return AgentResponse(
                success=False,
                message="Failed to search literature",
                error=str(e)
            )

    def generate_ideas(self, field: str, text: str = None) -> AgentResponse:
        """
        Generate research ideas based on field and optional text analysis

        Args:
            field: Research field
            text: Optional text to analyze for idea generation

        Returns:
            AgentResponse with generated ideas
        """
        try:
            logger.info(f"Generating research ideas for field: {field}")

            # Use provided text or last analyzed essay
            if not text and self.session_data['last_analysis']:
                # Extract text from last analysis (simplified)
                text = " ".join([section.content for section in self.session_data['last_analysis'].sections])

            if not text:
                text = f"Research in {field}"  # Fallback

            # Generate ideas
            ideas, gaps, hypotheses = self.idea_generator.generate_ideas(
                text, field, self.config.max_ideas_generated
            )

            # Store in session
            self.session_data['generated_ideas'] = ideas

            # Prepare response data
            response_data = {
                'field': field,
                'research_ideas': [
                    {
                        'title': idea.title,
                        'description': idea.description,
                        'methodology': idea.methodology,
                        'feasibility': idea.feasibility,
                        'novelty': idea.novelty,
                        'potential_impact': idea.potential_impact,
                        'required_resources': idea.required_resources,
                        'related_fields': idea.related_fields,
                        'estimated_timeline': idea.estimated_timeline,
                        'key_challenges': idea.key_challenges
                    }
                    for idea in ideas
                ],
                'research_gaps': [
                    {
                        'description': gap.description,
                        'gap_type': gap.gap_type,
                        'importance': gap.importance,
                        'research_questions': gap.research_questions
                    }
                    for gap in gaps
                ],
                'hypotheses': hypotheses,
                'generation_metadata': {
                    'ideas_generated': len(ideas),
                    'gaps_identified': len(gaps),
                    'hypotheses_created': len(hypotheses)
                }
            }

            return AgentResponse(
                success=True,
                message=f"Generated {len(ideas)} research ideas and {len(gaps)} research gaps for {field}.",
                data=response_data
            )

        except Exception as e:
            logger.error(f"Error generating ideas: {str(e)}")
            return AgentResponse(
                success=False,
                message="Failed to generate research ideas",
                error=str(e)
            )

    def create_visualization(self, viz_type: str = "dashboard", data: Dict = None) -> AgentResponse:
        """
        Create visualizations based on analysis results

        Args:
            viz_type: Type of visualization (dashboard, trends, concept_map, etc.)
            data: Optional custom data for visualization

        Returns:
            AgentResponse with visualization data
        """
        try:
            logger.info(f"Creating {viz_type} visualization")

            # Prepare data for visualization
            if not data:
                data = {
                    'analysis_result': self.session_data.get('last_analysis'),
                    'blind_spots': self.session_data.get('last_analysis').blind_spots if self.session_data.get('last_analysis') else [],
                    'sections': self.session_data.get('last_analysis').sections if self.session_data.get('last_analysis') else [],
                    'papers': self.session_data.get('last_search_results', {}).get('papers', []),
                    'research_ideas': self.session_data.get('generated_ideas', []),
                    'concepts': self.session_data.get('last_search_results', {}).get('related_keywords', [])
                }

            # Generate visualization
            viz_result = self.visualizer.create_visualization(data, viz_type)

            # Store in session
            self.session_data['visualizations'][viz_type] = viz_result

            # Save visualization if enabled
            if self.config.auto_save_results and viz_result:
                filename = f"{viz_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                if viz_type == "interactive":
                    filename = filename.replace('.png', '.html')

                self.visualizer.save_visualization(viz_result, filename, self.config.output_directory)

            response_data = {
                'visualization_type': viz_type,
                'visualization_data': viz_result,
                'format': 'html' if viz_type == 'interactive' else 'base64_png'
            }

            return AgentResponse(
                success=True,
                message=f"{viz_type.title()} visualization created successfully.",
                data=response_data
            )

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Failed to create {viz_type} visualization",
                error=str(e)
            )

    def get_latest_papers(self, field: str, days_back: int = 30) -> AgentResponse:
        """
        Get the latest papers in a specific field

        Args:
            field: Research field
            days_back: Number of days to look back

        Returns:
            AgentResponse with latest papers
        """
        try:
            logger.info(f"Getting latest papers in {field}")

            latest_papers = self.web_searcher.get_latest_papers(field, days_back, self.config.max_search_results)

            response_data = {
                'field': field,
                'days_back': days_back,
                'papers_found': len(latest_papers),
                'papers': [
                    {
                        'title': paper.title,
                        'authors': paper.authors,
                        'abstract': paper.abstract[:200] + '...' if len(paper.abstract) > 200 else paper.abstract,
                        'url': paper.url,
                        'publication_date': paper.publication_date,
                        'keywords': paper.keywords
                    }
                    for paper in latest_papers
                ]
            }

            return AgentResponse(
                success=True,
                message=f"Found {len(latest_papers)} recent papers in {field}.",
                data=response_data
            )

        except Exception as e:
            logger.error(f"Error getting latest papers: {str(e)}")
            return AgentResponse(
                success=False,
                message="Failed to get latest papers",
                error=str(e)
            )

    def comprehensive_analysis(self, text: str = None, file_path: str = None, field: str = None) -> AgentResponse:
        """
        Perform comprehensive analysis including essay analysis, idea generation, and literature search

        Args:
            text: Essay text content
            file_path: Path to essay file
            field: Research field for idea generation and literature search

        Returns:
            AgentResponse with comprehensive analysis results
        """
        try:
            logger.info("Starting comprehensive analysis...")

            results = {}

            # 1. Analyze essay
            essay_result = self.analyze_essay(text=text, file_path=file_path)
            if not essay_result.success:
                return essay_result
            results['essay_analysis'] = essay_result.data

            # 2. Generate ideas (if field provided)
            if field:
                ideas_result = self.generate_ideas(field, text)
                if ideas_result.success:
                    results['research_ideas'] = ideas_result.data

                # 3. Search literature
                search_query = field
                if text:
                    # Extract key terms for better search
                    concepts = self.idea_generator.extract_key_concepts(text)
                    if concepts:
                        search_query = f"{field} {' '.join(concepts[:3])}"

                search_result = asyncio.run(self.search_literature(search_query))
                if search_result.success:
                    results['literature_search'] = search_result.data

            # 4. Create dashboard visualization
            viz_result = self.create_visualization("dashboard")
            if viz_result.success:
                results['visualization'] = viz_result.data

            return AgentResponse(
                success=True,
                message="Comprehensive analysis completed successfully.",
                data=results
            )

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return AgentResponse(
                success=False,
                message="Failed to complete comprehensive analysis",
                error=str(e)
            )

    def process_command(self, command: str, **kwargs) -> AgentResponse:
        """
        Process various commands for the agent

        Args:
            command: Command to execute
            **kwargs: Additional arguments for the command

        Returns:
            AgentResponse with command results
        """
        try:
            logger.info(f"Processing command: {command}")

            if command == "analyze_essay":
                return self.analyze_essay(
                    text=kwargs.get('text'),
                    file_path=kwargs.get('file_path')
                )

            elif command == "search_literature":
                return asyncio.run(self.search_literature(
                    query=kwargs.get('query', ''),
                    sources=kwargs.get('sources')
                ))

            elif command == "generate_ideas":
                return self.generate_ideas(
                    field=kwargs.get('field', ''),
                    text=kwargs.get('text')
                )

            elif command == "create_visualization":
                return self.create_visualization(
                    viz_type=kwargs.get('viz_type', 'dashboard'),
                    data=kwargs.get('data')
                )

            elif command == "get_latest_papers":
                return self.get_latest_papers(
                    field=kwargs.get('field', ''),
                    days_back=kwargs.get('days_back', 30)
                )

            elif command == "comprehensive_analysis":
                return asyncio.run(self.comprehensive_analysis(
                    text=kwargs.get('text'),
                    file_path=kwargs.get('file_path'),
                    field=kwargs.get('field')
                ))

            elif command == "get_session_data":
                return AgentResponse(
                    success=True,
                    message="Session data retrieved",
                    data=self.session_data
                )

            elif command == "clear_session":
                self.session_data = {
                    'last_analysis': None,
                    'last_search_results': None,
                    'generated_ideas': [],
                    'visualizations': {}
                }
                return AgentResponse(
                    success=True,
                    message="Session data cleared"
                )

            else:
                return AgentResponse(
                    success=False,
                    message=f"Unknown command: {command}",
                    error=f"Available commands: analyze_essay, search_literature, generate_ideas, create_visualization, get_latest_papers, comprehensive_analysis, get_session_data, clear_session"
                )

        except Exception as e:
            logger.error(f"Error processing command {command}: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Failed to process command: {command}",
                error=str(e)
            )

    def _save_analysis_results(self, analysis_result: AnalysisResult):
        """Save analysis results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_results_{timestamp}.json"
            filepath = os.path.join(self.config.output_directory, filename)

            # Convert to serializable format
            data = {
                'timestamp': timestamp,
                'summary': analysis_result.summary,
                'overall_score': analysis_result.overall_score,
                'structure_quality': analysis_result.structure_quality,
                'argument_strength': analysis_result.argument_strength,
                'evidence_quality': analysis_result.evidence_quality,
                'sections': [
                    {
                        'title': section.title,
                        'content_length': len(section.content),
                        'confidence': section.confidence
                    }
                    for section in analysis_result.sections
                ],
                'blind_spots': [
                    {
                        'type': bs.type,
                        'description': bs.description,
                        'location': bs.location,
                        'severity': bs.severity,
                        'suggestions': bs.suggestions
                    }
                    for bs in analysis_result.blind_spots
                ]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Analysis results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")

    def get_help(self) -> AgentResponse:
        """Get help information about available commands and features"""
        help_text = """
SciDetective Agent - Scientific Research Assistant

Available Commands:
1. analyze_essay(text=None, file_path=None) - Analyze essay structure and detect blind spots
2. search_literature(query, sources=None) - Search scientific databases for papers
3. generate_ideas(field, text=None) - Generate research ideas and identify gaps
4. create_visualization(viz_type='dashboard', data=None) - Create various visualizations
5. get_latest_papers(field, days_back=30) - Get recent papers in a field
6. comprehensive_analysis(text=None, file_path=None, field=None) - Full analysis pipeline
7. get_session_data() - Retrieve current session data
8. clear_session() - Clear session data

Visualization Types:
- dashboard: Complete analysis dashboard
- trends: Research publication trends
- concept_map: Concept relationship map
- citation_network: Citation analysis
- research_gaps: Research gaps and ideas
- interactive: Interactive dashboard (if enabled)

Supported File Formats:
- PDF (.pdf)
- Microsoft Word (.docx)
- Plain Text (.txt)

Features:
- NLP-based essay structure detection
- Blind spot identification and suggestions
- Cross-domain research idea generation
- Multi-source literature search (arXiv, Semantic Scholar, PubMed)
- Various visualization types
- Interactive dashboards
- Auto-save functionality
        """

        return AgentResponse(
            success=True,
            message="Help information retrieved",
            data={'help_text': help_text.strip()}
        )