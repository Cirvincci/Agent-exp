"""
SciDetective Agent MCP Server

This module provides MCP (Model Context Protocol) tools for the SciDetective Agent,
allowing external systems to interact with the agent's capabilities.
"""

import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Import our agent
from .the_agent import SciDetectiveAgent, AgentConfig

# Initialize the agent
agent_config = AgentConfig(
    max_search_results=20,
    max_ideas_generated=10,
    visualization_style='scientific',
    enable_interactive_viz=True,
    auto_save_results=True,
    output_directory='outputs'
)

sci_detective = SciDetectiveAgent(agent_config)

# Initialize FastMCP server
mcp = FastMCP("scidetective", host="0.0.0.0", port=50002)

@mcp.tool()
def analyze_essay(text: str = None, file_path: str = None) -> str:
    """
    Analyze a scientific essay for structure, blind spots, and quality.

    Args:
        text: Essay text content (optional if file_path provided)
        file_path: Path to essay file (PDF, DOCX, TXT) (optional if text provided)

    Returns:
        JSON string with analysis results including sections, blind spots, and quality scores
    """
    try:
        result = sci_detective.analyze_essay(text=text, file_path=file_path)
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to analyze essay',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def search_literature(query: str, max_results: int = 10) -> str:
    """
    Search scientific literature across multiple databases.

    Args:
        query: Search query for scientific papers
        max_results: Maximum number of results to return (default: 10)

    Returns:
        JSON string with search results from arXiv, Semantic Scholar, and PubMed
    """
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(sci_detective.search_literature(query))
        finally:
            loop.close()

        # Limit results if requested
        if result.success and result.data and 'papers' in result.data:
            result.data['papers'] = result.data['papers'][:max_results]

        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to search literature',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def generate_ideas(field: str, context_text: str = None) -> str:
    """
    Generate research ideas and identify research gaps in a scientific field.

    Args:
        field: Scientific field or research area
        context_text: Optional context text to inform idea generation

    Returns:
        JSON string with generated research ideas, gaps, and hypotheses
    """
    try:
        result = sci_detective.generate_ideas(field=field, text=context_text)
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to generate ideas',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def create_visualization(viz_type: str = "dashboard") -> str:
    """
    Create visualizations based on previous analysis results.

    Args:
        viz_type: Type of visualization (dashboard, trends, concept_map, citation_network, research_gaps, interactive)

    Returns:
        JSON string with visualization data (base64 encoded PNG or HTML)
    """
    try:
        result = sci_detective.create_visualization(viz_type=viz_type)
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to create visualization',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def get_latest_papers(field: str, days_back: int = 30) -> str:
    """
    Get the latest papers in a specific research field.

    Args:
        field: Research field to search
        days_back: Number of days to look back for recent papers (default: 30)

    Returns:
        JSON string with recent papers in the specified field
    """
    try:
        result = sci_detective.get_latest_papers(field=field, days_back=days_back)
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to get latest papers',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def comprehensive_analysis(text: str = None, file_path: str = None, field: str = None) -> str:
    """
    Perform comprehensive analysis including essay analysis, idea generation, and literature search.

    Args:
        text: Essay text content (optional if file_path provided)
        file_path: Path to essay file (optional if text provided)
        field: Research field for idea generation and literature search (optional)

    Returns:
        JSON string with complete analysis results including visualizations
    """
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                sci_detective.comprehensive_analysis(text=text, file_path=file_path, field=field)
            )
        finally:
            loop.close()

        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to perform comprehensive analysis',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def get_agent_help() -> str:
    """
    Get help information about available commands and features of the SciDetective Agent.

    Returns:
        JSON string with help information and usage examples
    """
    try:
        result = sci_detective.get_help()
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to get help information',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def clear_session() -> str:
    """
    Clear the current session data including analysis results and cached data.

    Returns:
        JSON string confirming session has been cleared
    """
    try:
        result = sci_detective.process_command("clear_session")
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to clear session',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def get_session_data() -> str:
    """
    Retrieve current session data including last analysis, search results, and generated ideas.

    Returns:
        JSON string with current session state
    """
    try:
        result = sci_detective.process_command("get_session_data")
        return json.dumps({
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'error': result.error
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to get session data',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def extract_key_concepts(text: str) -> str:
    """
    Extract key scientific concepts from text.

    Args:
        text: Text to analyze for key concepts

    Returns:
        JSON string with extracted concepts and their frequencies
    """
    try:
        concepts = sci_detective.idea_generator.extract_key_concepts(text)
        return json.dumps({
            'success': True,
            'message': f'Extracted {len(concepts)} key concepts',
            'data': {
                'concepts': concepts,
                'total_concepts': len(concepts)
            }
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to extract key concepts',
            'error': str(e)
        }, indent=2)

@mcp.tool()
def detect_research_gaps(text: str, field: str) -> str:
    """
    Identify potential research gaps from text in a specific field.

    Args:
        text: Research text to analyze
        field: Research field context

    Returns:
        JSON string with identified research gaps and their characteristics
    """
    try:
        gaps = sci_detective.idea_generator.identify_research_gaps(text, field)
        gap_data = [
            {
                'description': gap.description,
                'gap_type': gap.gap_type,
                'importance': gap.importance,
                'research_questions': gap.research_questions
            }
            for gap in gaps
        ]

        return json.dumps({
            'success': True,
            'message': f'Identified {len(gaps)} research gaps',
            'data': {
                'research_gaps': gap_data,
                'total_gaps': len(gaps),
                'field': field
            }
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': 'Failed to detect research gaps',
            'error': str(e)
        }, indent=2)

if __name__ == "__main__":
    # Initialize and run the server
    print("Starting SciDetective Agent MCP Server...")
    print("Available tools:")
    print("- analyze_essay: Analyze scientific essays")
    print("- search_literature: Search scientific databases")
    print("- generate_ideas: Generate research ideas")
    print("- create_visualization: Create various visualizations")
    print("- get_latest_papers: Get recent papers in a field")
    print("- comprehensive_analysis: Full analysis pipeline")
    print("- extract_key_concepts: Extract concepts from text")
    print("- detect_research_gaps: Identify research gaps")
    print("- get_agent_help: Get help information")
    print("- clear_session: Clear session data")
    print("- get_session_data: Get current session data")
    print(f"Server running on port 50002...")

    try:
        mcp.run(transport='sse')
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")