"""
Visualization Generator Module for SciDetective_Agent

This module creates visual representations including:
- Topic comparison charts
- Concept maps and knowledge graphs
- Research trend analysis
- Citation networks
- Timeline visualizations
- Quality assessment dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import io
import base64
from datetime import datetime
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    style: str = 'scientific'  # 'scientific', 'modern', 'minimal'
    color_palette: str = 'viridis'  # matplotlib/seaborn color palette
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    interactive: bool = False  # Whether to generate interactive plots

class VisualizationGenerator:
    """
    Generates various types of visualizations for scientific analysis
    """

    def __init__(self, config: VisualizationConfig = None):
        """Initialize the visualization generator"""
        self.config = config or VisualizationConfig()

        # Set up matplotlib/seaborn styling
        plt.style.use('seaborn-v0_8' if self.config.style == 'scientific' else 'default')
        sns.set_palette(self.config.color_palette)

        # Color schemes for different visualizations
        self.color_schemes = {
            'quality': ['#d32f2f', '#ff9800', '#4caf50'],  # Red, Orange, Green
            'severity': ['#4caf50', '#ff9800', '#d32f2f'],  # Green, Orange, Red
            'categories': px.colors.qualitative.Set3,
            'timeline': px.colors.sequential.Viridis
        }

    def create_essay_analysis_dashboard(self, analysis_result, blind_spots: List, sections: List) -> str:
        """
        Create a comprehensive dashboard for essay analysis

        Args:
            analysis_result: Analysis results from text_analysis module
            blind_spots: List of detected blind spots
            sections: List of essay sections

        Returns:
            Base64 encoded PNG image of the dashboard
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Scientific Essay Analysis Dashboard', fontsize=16, fontweight='bold')

        try:
            # 1. Quality Scores Radar Chart
            self._create_quality_radar(axes[0, 0], analysis_result)

            # 2. Blind Spots by Type
            self._create_blind_spots_chart(axes[0, 1], blind_spots)

            # 3. Section Structure Analysis
            self._create_section_analysis(axes[0, 2], sections)

            # 4. Severity Distribution
            self._create_severity_distribution(axes[1, 0], blind_spots)

            # 5. Section Quality Comparison
            self._create_section_quality_comparison(axes[1, 1], sections, blind_spots)

            # 6. Improvement Recommendations
            self._create_recommendations_chart(axes[1, 2], blind_spots)

            plt.tight_layout()

            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info("Essay analysis dashboard created successfully")
            return image_base64

        except Exception as e:
            logger.error(f"Error creating essay analysis dashboard: {str(e)}")
            plt.close()
            return ""

    def _create_quality_radar(self, ax, analysis_result):
        """Create radar chart for quality metrics"""
        categories = ['Structure', 'Arguments', 'Evidence', 'Overall']
        values = [
            analysis_result.structure_quality,
            analysis_result.argument_strength,
            analysis_result.evidence_quality,
            analysis_result.overall_score
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values_plot = values + [values[0]]  # Complete the circle
        angles_plot = np.append(angles, angles[0])

        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles_plot, values_plot, alpha=0.25, color='#1f77b4')
        ax.set_xticks(angles)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Quality Assessment Radar', fontweight='bold')
        ax.grid(True)

    def _create_blind_spots_chart(self, ax, blind_spots):
        """Create bar chart of blind spots by type"""
        if not blind_spots:
            ax.text(0.5, 0.5, 'No blind spots detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Blind Spots by Type')
            return

        blind_spot_types = [bs.type.replace('_', ' ').title() for bs in blind_spots]
        type_counts = Counter(blind_spot_types)

        types = list(type_counts.keys())
        counts = list(type_counts.values())

        bars = ax.bar(types, counts, color=self.color_schemes['categories'][:len(types)])
        ax.set_title('Blind Spots by Type', fontweight='bold')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')

    def _create_section_analysis(self, ax, sections):
        """Create pie chart of section distribution"""
        if not sections:
            ax.text(0.5, 0.5, 'No sections detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Section Distribution')
            return

        section_names = [section.title.title() for section in sections]
        section_lengths = [len(section.content) for section in sections]

        wedges, texts, autotexts = ax.pie(section_lengths, labels=section_names, autopct='%1.1f%%',
                                         colors=self.color_schemes['categories'][:len(sections)])
        ax.set_title('Section Distribution by Length', fontweight='bold')

    def _create_severity_distribution(self, ax, blind_spots):
        """Create donut chart of severity distribution"""
        if not blind_spots:
            ax.text(0.5, 0.5, 'No issues found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Issue Severity Distribution')
            return

        severities = [bs.severity for bs in blind_spots]
        severity_counts = Counter(severities)

        labels = list(severity_counts.keys())
        sizes = list(severity_counts.values())
        colors = [self.color_schemes['severity'][['low', 'medium', 'high'].index(label)] for label in labels]

        wedges, texts = ax.pie(sizes, labels=labels, colors=colors, wedgeprops=dict(width=0.5))
        ax.set_title('Issue Severity Distribution', fontweight='bold')

    def _create_section_quality_comparison(self, ax, sections, blind_spots):
        """Create section quality comparison chart"""
        if not sections:
            ax.text(0.5, 0.5, 'No sections available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Section Quality Comparison')
            return

        # Calculate quality scores per section
        section_scores = {}
        section_issues = defaultdict(int)

        # Count issues per section
        for bs in blind_spots:
            section_issues[bs.location] += 1

        # Calculate quality scores (inverse of issue count, normalized)
        max_issues = max(section_issues.values()) if section_issues else 1
        for section in sections:
            issues = section_issues.get(section.title, 0)
            quality = 1 - (issues / max(max_issues, 1))
            section_scores[section.title] = quality

        section_names = list(section_scores.keys())
        quality_scores = list(section_scores.values())

        bars = ax.bar(section_names, quality_scores, color=self.color_schemes['categories'][:len(section_names)])
        ax.set_title('Section Quality Scores', fontweight='bold')
        ax.set_ylabel('Quality Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars, quality_scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom')

    def _create_recommendations_chart(self, ax, blind_spots):
        """Create chart showing recommendation categories"""
        if not blind_spots:
            ax.text(0.5, 0.5, 'No recommendations needed', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Improvement Recommendations')
            return

        # Extract recommendation types
        rec_types = []
        for bs in blind_spots:
            for suggestion in bs.suggestions:
                if 'evidence' in suggestion.lower():
                    rec_types.append('Add Evidence')
                elif 'citation' in suggestion.lower():
                    rec_types.append('Add Citations')
                elif 'clarify' in suggestion.lower():
                    rec_types.append('Clarify Arguments')
                elif 'expand' in suggestion.lower():
                    rec_types.append('Expand Content')
                else:
                    rec_types.append('General Improvement')

        rec_counts = Counter(rec_types)
        categories = list(rec_counts.keys())
        counts = list(rec_counts.values())

        bars = ax.barh(categories, counts, color=self.color_schemes['categories'][:len(categories)])
        ax.set_title('Improvement Recommendations', fontweight='bold')
        ax.set_xlabel('Count')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}', ha='left', va='center')

    def create_research_trends_chart(self, papers: List, time_period: str = "monthly") -> str:
        """
        Create visualization of research trends over time

        Args:
            papers: List of papers with publication dates
            time_period: Aggregation period ('daily', 'monthly', 'yearly')

        Returns:
            Base64 encoded PNG image
        """
        try:
            # Filter papers with valid dates
            dated_papers = [p for p in papers if p.publication_date]

            if not dated_papers:
                # Create empty chart
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                ax.text(0.5, 0.5, 'No publication date data available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Research Publication Trends')
            else:
                # Parse dates and create timeline
                dates = []
                for paper in dated_papers:
                    try:
                        date_obj = datetime.strptime(paper.publication_date, '%Y-%m-%d')
                        dates.append(date_obj)
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(paper.publication_date, '%Y')
                            dates.append(date_obj)
                        except ValueError:
                            continue

                if dates:
                    # Create DataFrame for easier manipulation
                    df = pd.DataFrame({'date': dates})

                    # Group by time period
                    if time_period == "yearly":
                        df['period'] = df['date'].dt.year
                    elif time_period == "monthly":
                        df['period'] = df['date'].dt.to_period('M')
                    else:  # daily
                        df['period'] = df['date'].dt.date

                    publication_counts = df.groupby('period').size()

                    # Create the plot
                    fig, ax = plt.subplots(figsize=self.config.figure_size)
                    publication_counts.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=6)
                    ax.set_title(f'Research Publication Trends ({time_period.title()})', fontweight='bold')
                    ax.set_xlabel(f'Time Period ({time_period.title()})')
                    ax.set_ylabel('Number of Publications')
                    ax.grid(True, alpha=0.3)

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info("Research trends chart created successfully")
            return image_base64

        except Exception as e:
            logger.error(f"Error creating research trends chart: {str(e)}")
            plt.close()
            return ""

    def create_concept_map(self, concepts: List[str], relationships: Dict = None) -> str:
        """
        Create a concept map/knowledge graph

        Args:
            concepts: List of key concepts
            relationships: Optional dictionary of concept relationships

        Returns:
            Base64 encoded PNG image
        """
        try:
            if not concepts:
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                ax.text(0.5, 0.5, 'No concepts to visualize',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Concept Map')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                return image_base64

            # Create network graph
            G = nx.Graph()

            # Add nodes for concepts
            for concept in concepts[:20]:  # Limit to 20 concepts for readability
                G.add_node(concept)

            # Add edges based on relationships or create default connections
            if relationships:
                for concept1, related_concepts in relationships.items():
                    if concept1 in concepts:
                        for concept2 in related_concepts:
                            if concept2 in concepts:
                                G.add_edge(concept1, concept2)
            else:
                # Create default connections based on concept similarity (simplified)
                for i, concept1 in enumerate(concepts[:15]):
                    for concept2 in concepts[i+1:min(i+4, len(concepts))]:
                        G.add_edge(concept1, concept2)

            # Create the visualization
            fig, ax = plt.subplots(figsize=self.config.figure_size)

            # Calculate layout
            pos = nx.spring_layout(G, k=3, iterations=50)

            # Draw the network
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                                 node_size=1500, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=1)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')

            ax.set_title('Concept Map', fontweight='bold')
            ax.axis('off')

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info("Concept map created successfully")
            return image_base64

        except Exception as e:
            logger.error(f"Error creating concept map: {str(e)}")
            plt.close()
            return ""

    def create_citation_network(self, papers: List) -> str:
        """
        Create citation network visualization

        Args:
            papers: List of papers with citation information

        Returns:
            Base64 encoded PNG image
        """
        try:
            # Filter papers with citation data
            cited_papers = [p for p in papers if p.citation_count is not None and p.citation_count > 0]

            if not cited_papers:
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                ax.text(0.5, 0.5, 'No citation data available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Citation Network')
            else:
                # Create citation network based on citation counts
                fig, ax = plt.subplots(figsize=self.config.figure_size)

                # Prepare data for scatter plot
                x_data = range(len(cited_papers))
                y_data = [p.citation_count for p in cited_papers]
                sizes = [min(p.citation_count * 10, 500) for p in cited_papers]  # Scale for visualization

                # Create scatter plot
                scatter = ax.scatter(x_data, y_data, s=sizes, alpha=0.6,
                                   c=y_data, cmap='viridis')

                # Add labels for highly cited papers
                for i, paper in enumerate(cited_papers):
                    if paper.citation_count > np.percentile(y_data, 75):  # Top 25%
                        ax.annotate(paper.title[:30] + '...',
                                  (i, paper.citation_count),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.8)

                ax.set_title('Citation Network Analysis', fontweight='bold')
                ax.set_xlabel('Papers (chronological order)')
                ax.set_ylabel('Citation Count')

                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Citations')

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info("Citation network created successfully")
            return image_base64

        except Exception as e:
            logger.error(f"Error creating citation network: {str(e)}")
            plt.close()
            return ""

    def create_research_gap_visualization(self, research_gaps: List, research_ideas: List) -> str:
        """
        Create visualization of research gaps and proposed ideas

        Args:
            research_gaps: List of identified research gaps
            research_ideas: List of generated research ideas

        Returns:
            Base64 encoded PNG image
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Left plot: Research gaps by type and importance
            if research_gaps:
                gap_types = [gap.gap_type for gap in research_gaps]
                gap_importance = [gap.importance for gap in research_gaps]

                # Create a 2D histogram/heatmap
                gap_data = defaultdict(lambda: defaultdict(int))
                for gap_type, importance in zip(gap_types, gap_importance):
                    gap_data[gap_type][importance] += 1

                # Prepare data for heatmap
                types = list(gap_data.keys())
                importances = ['low', 'medium', 'high']
                heatmap_data = np.zeros((len(types), len(importances)))

                for i, gap_type in enumerate(types):
                    for j, importance in enumerate(importances):
                        heatmap_data[i, j] = gap_data[gap_type][importance]

                sns.heatmap(heatmap_data, xticklabels=importances, yticklabels=types,
                           annot=True, fmt='g', ax=axes[0], cmap='YlOrRd')
                axes[0].set_title('Research Gaps by Type and Importance', fontweight='bold')
                axes[0].set_xlabel('Importance Level')
                axes[0].set_ylabel('Gap Type')
            else:
                axes[0].text(0.5, 0.5, 'No research gaps identified',
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Research Gaps Analysis')

            # Right plot: Research ideas by feasibility and novelty
            if research_ideas:
                feasibilities = [idea.feasibility for idea in research_ideas]
                novelties = [idea.novelty for idea in research_ideas]

                # Map text values to numbers for plotting
                level_map = {'low': 1, 'medium': 2, 'high': 3}
                feas_nums = [level_map[f] for f in feasibilities]
                nov_nums = [level_map[n] for n in novelties]

                # Create scatter plot
                scatter = axes[1].scatter(feas_nums, nov_nums, s=100, alpha=0.6,
                                        c=range(len(research_ideas)), cmap='viridis')

                # Add labels
                axes[1].set_xticks([1, 2, 3])
                axes[1].set_xticklabels(['Low', 'Medium', 'High'])
                axes[1].set_yticks([1, 2, 3])
                axes[1].set_yticklabels(['Low', 'Medium', 'High'])
                axes[1].set_xlabel('Feasibility')
                axes[1].set_ylabel('Novelty')
                axes[1].set_title('Research Ideas: Feasibility vs Novelty', fontweight='bold')

                # Add grid
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'No research ideas generated',
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Research Ideas Analysis')

            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            logger.info("Research gap visualization created successfully")
            return image_base64

        except Exception as e:
            logger.error(f"Error creating research gap visualization: {str(e)}")
            plt.close()
            return ""

    def create_interactive_dashboard(self, data: Dict) -> str:
        """
        Create an interactive dashboard using Plotly

        Args:
            data: Dictionary containing various analysis data

        Returns:
            HTML string of the interactive dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Quality Metrics', 'Research Trends',
                              'Citation Analysis', 'Concept Network'),
                specs=[[{"type": "scatterpolar"}, {"type": "scatter"}],
                      [{"type": "bar"}, {"type": "scatter"}]]
            )

            # Add quality radar chart
            if 'quality_scores' in data:
                fig.add_trace(
                    go.Scatterpolar(
                        r=list(data['quality_scores'].values()),
                        theta=list(data['quality_scores'].keys()),
                        fill='toself',
                        name='Quality Scores'
                    ),
                    row=1, col=1
                )

            # Add other plots based on available data
            # This would be expanded based on the specific data structure

            fig.update_layout(
                title="SciDetective Analysis Dashboard",
                showlegend=True,
                height=800
            )

            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            logger.info("Interactive dashboard created successfully")
            return html_str

        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {str(e)}")
            return "<html><body><h1>Error creating dashboard</h1></body></html>"

    def save_visualization(self, image_base64: str, filename: str, output_dir: str = "visualizations") -> bool:
        """
        Save a base64 encoded visualization to file

        Args:
            image_base64: Base64 encoded image data
            filename: Output filename
            output_dir: Output directory

        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(image_base64))

            logger.info(f"Visualization saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return False

    def create_visualization(self, data: Dict, viz_type: str = "dashboard") -> str:
        """
        Main method to create visualizations based on data and type

        Args:
            data: Analysis data dictionary
            viz_type: Type of visualization to create

        Returns:
            Base64 encoded image or HTML string for interactive visualizations
        """
        logger.info(f"Creating {viz_type} visualization")

        try:
            if viz_type == "dashboard":
                return self.create_essay_analysis_dashboard(
                    data.get('analysis_result'),
                    data.get('blind_spots', []),
                    data.get('sections', [])
                )
            elif viz_type == "trends":
                return self.create_research_trends_chart(data.get('papers', []))
            elif viz_type == "concept_map":
                return self.create_concept_map(data.get('concepts', []))
            elif viz_type == "citation_network":
                return self.create_citation_network(data.get('papers', []))
            elif viz_type == "research_gaps":
                return self.create_research_gap_visualization(
                    data.get('research_gaps', []),
                    data.get('research_ideas', [])
                )
            elif viz_type == "interactive":
                return self.create_interactive_dashboard(data)
            else:
                logger.warning(f"Unknown visualization type: {viz_type}")
                return ""

        except Exception as e:
            logger.error(f"Error creating {viz_type} visualization: {str(e)}")
            return ""