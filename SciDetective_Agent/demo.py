#!/usr/bin/env python3
"""
SciDetective Agent Demo Script

This script demonstrates the key functionality of the SciDetective Agent
including essay analysis, research idea generation, literature search, and visualization.
"""

import asyncio
import json
import os
import sys

# Add the agent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))

from agent import SciDetectiveAgent, AgentConfig

def print_separator(title):
    """Print a formatted separator for demo sections"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def print_results(result, detailed=False):
    """Print formatted results from agent operations"""
    if result.success:
        print(f"✅ {result.message}")
        if detailed and result.data:
            print("\nKey Results:")
            for key, value in result.data.items():
                if isinstance(value, (int, float, str)):
                    print(f"  • {key}: {value}")
                elif isinstance(value, list):
                    print(f"  • {key}: {len(value)} items")
    else:
        print(f"❌ {result.message}")
        if result.error:
            print(f"   Error: {result.error}")

async def main():
    """Main demo function"""
    print_separator("SciDetective Agent Demo")
    print("🔬 Welcome to the SciDetective Agent demonstration!")
    print("This demo will showcase the agent's key capabilities.")

    # Initialize agent with custom configuration
    config = AgentConfig(
        max_search_results=5,  # Limit for demo
        max_ideas_generated=3,
        visualization_style='scientific',
        enable_interactive_viz=False,  # Disable for demo
        auto_save_results=True,
        output_directory='demo_outputs'
    )

    agent = SciDetectiveAgent(config)
    print(f"✅ Agent initialized with config: {config.output_directory}")

    # Demo 1: Essay Analysis
    print_separator("Demo 1: Essay Analysis")
    print("📝 Analyzing a sample scientific essay...")

    sample_essay = """
    Abstract: This study investigates the potential of quantum computing in solving complex optimization problems.

    Introduction: Quantum computing represents a paradigm shift in computational capabilities. However, current implementations face significant challenges in maintaining quantum coherence.

    Methods: We implemented several quantum algorithms using a 20-qubit quantum processor. The algorithms were tested on various optimization problems.

    Results: Our results show promising improvements in certain problem classes. However, the data is limited and more research is needed.

    Discussion: The findings suggest that quantum computing might be useful for optimization. Nevertheless, technical limitations remain a barrier.

    Conclusion: Quantum computing shows potential but requires further development. Additional research is necessary to fully understand its capabilities.
    """

    analysis_result = agent.analyze_essay(text=sample_essay)
    print_results(analysis_result, detailed=True)

    if analysis_result.success:
        print(f"\n📊 Analysis Summary:")
        data = analysis_result.data
        print(f"  • Overall Score: {data['overall_score']:.2f}/1.0")
        print(f"  • Sections Detected: {data['sections_detected']}")
        print(f"  • Blind Spots Found: {data['blind_spots_found']}")

        if data['blind_spots']:
            print(f"\n⚠️  Key Issues Found:")
            for i, bs in enumerate(data['blind_spots'][:3], 1):
                print(f"  {i}. {bs['type'].replace('_', ' ').title()}: {bs['description'][:80]}...")

    # Demo 2: Research Idea Generation
    print_separator("Demo 2: Research Idea Generation")
    print("💡 Generating research ideas for quantum computing...")

    ideas_result = agent.generate_ideas(field="quantum computing", text=sample_essay)
    print_results(ideas_result, detailed=True)

    if ideas_result.success:
        print(f"\n🧠 Generated Ideas:")
        for i, idea in enumerate(ideas_result.data['research_ideas'], 1):
            print(f"  {i}. {idea['title']}")
            print(f"     Feasibility: {idea['feasibility']} | Novelty: {idea['novelty']}")
            print(f"     Timeline: {idea['estimated_timeline']}")

    # Demo 3: Literature Search
    print_separator("Demo 3: Literature Search")
    print("🔍 Searching for recent papers on quantum optimization...")

    search_result = await agent.search_literature("quantum optimization algorithms")
    print_results(search_result, detailed=True)

    if search_result.success:
        print(f"\n📚 Found Papers:")
        for i, paper in enumerate(search_result.data['papers'][:3], 1):
            print(f"  {i}. {paper['title'][:60]}...")
            print(f"     Authors: {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
            print(f"     Source: {paper['source']} | Date: {paper['publication_date']}")

    # Demo 4: Latest Papers
    print_separator("Demo 4: Latest Papers")
    print("📅 Getting the latest papers in quantum computing...")

    latest_result = agent.get_latest_papers("quantum computing", days_back=7)
    print_results(latest_result, detailed=True)

    if latest_result.success:
        print(f"\n🆕 Recent Papers ({latest_result.data['papers_found']} found):")
        for i, paper in enumerate(latest_result.data['papers'][:2], 1):
            print(f"  {i}. {paper['title'][:60]}...")
            print(f"     Date: {paper['publication_date']}")

    # Demo 5: Visualization
    print_separator("Demo 5: Visualization Creation")
    print("📊 Creating analysis dashboard...")

    viz_result = agent.create_visualization("dashboard")
    print_results(viz_result)

    if viz_result.success:
        print("✅ Visualization created and saved to outputs directory")

    # Demo 6: Comprehensive Analysis
    print_separator("Demo 6: Comprehensive Analysis")
    print("🔄 Running complete analysis pipeline...")

    comprehensive_result = await agent.comprehensive_analysis(
        text=sample_essay,
        field="quantum computing"
    )
    print_results(comprehensive_result)

    if comprehensive_result.success:
        print("\n📋 Comprehensive Analysis Complete!")
        print("   • Essay analyzed ✅")
        print("   • Ideas generated ✅")
        print("   • Literature searched ✅")
        print("   • Visualizations created ✅")

    # Demo 7: Session Management
    print_separator("Demo 7: Session Management")
    print("💾 Checking session data...")

    session_result = agent.process_command("get_session_data")
    if session_result.success:
        session_data = session_result.data
        print("📊 Session Summary:")
        print(f"  • Last analysis: {'✅' if session_data.get('last_analysis') else '❌'}")
        print(f"  • Search results: {'✅' if session_data.get('last_search_results') else '❌'}")
        print(f"  • Generated ideas: {len(session_data.get('generated_ideas', []))}")
        print(f"  • Visualizations: {len(session_data.get('visualizations', {}))}")

    # Demo 8: Help System
    print_separator("Demo 8: Help System")
    print("❓ Getting help information...")

    help_result = agent.get_help()
    if help_result.success:
        print("📖 Help system available with comprehensive documentation")
        print("   Use agent.get_help() for detailed command information")

    # Summary
    print_separator("Demo Complete")
    print("🎉 SciDetective Agent demonstration completed successfully!")
    print(f"📁 Check the '{config.output_directory}' directory for saved outputs")
    print("\n🚀 Ready to analyze your scientific research!")

    # Optional: Clean up demo session
    print("\n🧹 Cleaning up demo session...")
    clear_result = agent.process_command("clear_session")
    if clear_result.success:
        print("✅ Session cleared")

if __name__ == "__main__":
    print("Starting SciDetective Agent Demo...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Please check that all dependencies are installed:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")