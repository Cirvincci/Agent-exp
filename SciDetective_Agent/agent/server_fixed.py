"""
SciDetective Agent MCP Server

This module provides MCP (Model Context Protocol) tools for the SciDetective Agent,
allowing external systems to interact with the agent's capabilities.
"""

import json
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import FastMCP
try:
    from mcp.server.fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    print("Warning: FastMCP not available. Please install with: pip install fastmcp")

# Import our agent with error handling
try:
    from the_agent import SciDetectiveAgent, AgentConfig
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SciDetectiveAgent: {e}")
    AGENT_AVAILABLE = False

if not FASTMCP_AVAILABLE:
    print("Cannot run MCP server without FastMCP. Exiting...")
    sys.exit(1)

if not AGENT_AVAILABLE:
    print("Cannot run server without SciDetectiveAgent. Exiting...")
    sys.exit(1)

# Initialize the agent
agent_config = AgentConfig(
    max_search_results=20,
    max_ideas_generated=10,
    visualization_style='scientific',
    enable_interactive_viz=True,
    auto_save_results=True,
    output_directory='outputs'
)

try:
    sci_detective = SciDetectiveAgent(agent_config)
    print("✅ SciDetective Agent initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize SciDetective Agent: {e}")
    sys.exit(1)

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

def main():
    """Main function to start the server"""
    print("🔬 Starting SciDetective Agent MCP Server...")
    print("=" * 50)
    print("Available tools:")
    print("- analyze_essay: Analyze scientific essays")
    print("- get_agent_help: Get help information")
    print(f"🚀 Server running on port 50002...")
    print("=" * 50)

    try:
        mcp.run(transport='sse')
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("Please check that all dependencies are installed correctly")

if __name__ == "__main__":
    main()
