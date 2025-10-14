# SciDetective Agent

## 🧠 Purpose
SciDetective_Agent is an autonomous research assistant that helps users analyze their scientific essays, detect blind spots, generate new research ideas, and gather cutting-edge insights from the latest literature. It provides visual comparisons and conceptual diagrams to enhance understanding.

## 🎯 Goals
1. **Essay Analysis**: Accept user input (full essay, topic keywords, PDF, or DOCX) and analyze structure, arguments, and evidence quality
2. **Blind Spot Detection**: Identify logical flaws, unsupported assumptions, or unexplored angles in scientific essays
3. **Literature Search**: Search the internet (arXiv, Semantic Scholar, PubMed) for the latest related papers
4. **Research Ideas**: Generate creative, feasible new research directions in the same field
5. **Visualization**: Create visual outputs (topic maps, comparison charts, dashboards) for better insights
6. **Interactive Interface**: Chat-style interface with "Essay Analysis Mode" and "Idea Discovery Mode"

## 🧩 Functional Modules

### 📝 Text Analysis Module (`text_analysis.py`)
- **Structure Detection**: Automatically identifies scientific paper sections (abstract, methods, results, discussion)
- **Blind Spot Detection**: Uses NLP patterns to find missing arguments, weak evidence, and logical gaps
- **Quality Assessment**: Calculates scores for structure, argument strength, and evidence quality
- **Document Loading**: Supports PDF, DOCX, and TXT file formats

### 💡 Idea Generator Module (`idea_generator.py`)
- **Cross-Domain Innovation**: Applies techniques from one field to another (e.g., physics principles to biology)
- **Research Gap Identification**: Finds unexplored areas and methodological gaps
- **Hypothesis Generation**: Creates testable research hypotheses
- **Feasibility Assessment**: Evaluates novelty, feasibility, and potential impact of ideas

### 🔍 Scientific Web Searcher (`web_searcher.py`)
- **Multi-Source Search**: Integrates arXiv, Semantic Scholar, and PubMed APIs
- **Recent Paper Tracking**: Finds the latest publications in specific fields
- **Citation Analysis**: Analyzes citation networks and paper impact
- **Rate Limiting**: Respects API limits with intelligent request management

### 📊 Visualization Generator (`visualization.py`)
- **Analysis Dashboards**: Comprehensive quality assessment visualizations
- **Research Trends**: Publication timeline and trend analysis
- **Concept Maps**: Knowledge graphs showing concept relationships
- **Citation Networks**: Visual analysis of paper citations and impact
- **Interactive Charts**: Plotly-based interactive visualizations

### 🤖 Main Agent Class (`agent.py`)
- **Command Routing**: Processes different analysis commands
- **Session Management**: Maintains analysis state across interactions
- **Comprehensive Analysis**: Orchestrates all modules for complete analysis
- **Auto-Save**: Automatically saves results and visualizations

### 🌐 MCP Server (`server.py`)
- **Tool Integration**: Exposes agent capabilities as MCP tools
- **API Interface**: Provides JSON-based communication
- **Error Handling**: Robust error management and reporting

## 🚀 Installation and Setup

### Prerequisites
```bash
# Required Python packages
pip install -r requirements.txt
```

### Required Dependencies
```text
# Core ML and NLP
spacy>=3.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
networkx>=2.8.0
pandas>=1.4.0
numpy>=1.21.0

# Document processing
PyPDF2>=3.0.0
python-docx>=0.8.11

# Web and API
requests>=2.28.0
aiohttp>=3.8.0
arxiv>=1.4.0

# MCP and Agent Framework
fastmcp
google-adk
litellm
openai

# Environment and utilities
python-dotenv
pydantic
asyncio
```

### spaCy Model Installation
```bash
python -m spacy download en_core_web_sm
```

## 📋 Usage Examples

### 🧩 Example Workflow

**Scenario**: "Analyze my essay about plasma confinement in fusion reactors."

1. **Agent receives essay** →
2. **Runs essay analysis** → Finds weak theoretical justification →
3. **Retrieves latest arXiv papers** →
4. **Suggests stronger models** →
5. **Provides comparison table and visual diagram**

### 💻 Code Examples

#### 1. Basic Essay Analysis
```python
from agent import SciDetectiveAgent

# Initialize agent
agent = SciDetectiveAgent()

# Analyze essay from file
result = agent.analyze_essay(file_path="my_essay.pdf")
print(f"Overall Score: {result.data['overall_score']}")
print(f"Blind Spots Found: {result.data['blind_spots_found']}")
```

#### 2. Literature Search
```python
# Search recent papers
search_result = await agent.search_literature("plasma confinement fusion")
print(f"Found {search_result.data['total_papers_found']} papers")

# Get latest papers in field
latest = agent.get_latest_papers("fusion energy", days_back=30)
```

#### 3. Research Idea Generation
```python
# Generate ideas for a field
ideas_result = agent.generate_ideas("fusion energy", text=essay_text)
print(f"Generated {len(ideas_result.data['research_ideas'])} new ideas")

for idea in ideas_result.data['research_ideas']:
    print(f"- {idea['title']}: {idea['feasibility']} feasibility")
```

#### 4. Create Visualizations
```python
# Create analysis dashboard
viz_result = agent.create_visualization("dashboard")

# Create concept map
concept_viz = agent.create_visualization("concept_map")

# Create research trends chart
trends_viz = agent.create_visualization("trends")
```

#### 5. Comprehensive Analysis
```python
# Full pipeline analysis
comprehensive = await agent.comprehensive_analysis(
    file_path="research_paper.pdf",
    field="quantum computing"
)

# Results include essay analysis, literature search, ideas, and visualizations
print(comprehensive.data.keys())
# Output: ['essay_analysis', 'research_ideas', 'literature_search', 'visualization']
```

## 🛠️ MCP Server Tools

The agent exposes the following tools via MCP:

| Tool | Description | Parameters |
|------|-------------|------------|
| `analyze_essay` | Analyze essay structure and quality | `text`, `file_path` |
| `search_literature` | Search scientific databases | `query`, `max_results` |
| `generate_ideas` | Generate research ideas | `field`, `context_text` |
| `create_visualization` | Create various visualizations | `viz_type` |
| `get_latest_papers` | Get recent papers in field | `field`, `days_back` |
| `comprehensive_analysis` | Full analysis pipeline | `text`, `file_path`, `field` |
| `extract_key_concepts` | Extract concepts from text | `text` |
| `detect_research_gaps` | Identify research gaps | `text`, `field` |

### Running the MCP Server
```bash
cd D:\Agents\SciDetective_Agent\agent
python server.py
```

## 📊 Visualization Types

### 1. Analysis Dashboard
- Quality metrics radar chart
- Blind spots by type
- Section distribution
- Severity analysis
- Improvement recommendations

### 2. Research Trends
- Publication timeline analysis
- Field evolution tracking
- Citation trend analysis

### 3. Concept Maps
- Knowledge graphs
- Concept relationships
- Cross-domain connections

### 4. Citation Networks
- Paper impact visualization
- Citation flow analysis
- Influential work identification

### 5. Research Gap Analysis
- Gap type categorization
- Importance assessment
- Idea feasibility mapping

## 🎮 Interactive Modes

### Essay Analysis Mode
```python
# Analyze structure, detect blind spots, assess quality
result = agent.analyze_essay(text=essay_content)

# Get specific recommendations
for blind_spot in result.data['blind_spots']:
    print(f"Issue: {blind_spot['type']}")
    print(f"Location: {blind_spot['location']}")
    print(f"Suggestions: {blind_spot['suggestions']}")
```

### Idea Discovery Mode
```python
# Generate cross-domain ideas
ideas = agent.generate_ideas(field="materials science")

# Explore research gaps
gaps = agent.detect_research_gaps(text=research_text, field="nanotechnology")

# Get latest developments
latest = agent.get_latest_papers("quantum materials", days_back=7)
```

## 📁 Output Structure

```
outputs/
├── analysis_results_20241014_143052.json    # Analysis results
├── dashboard_20241014_143053.png            # Visualization files
├── concept_map_20241014_143054.png
├── research_trends_20241014_143055.png
└── visualizations/                          # Additional visualizations
    ├── interactive_dashboard.html
    └── citation_network.png
```

## 🔧 Configuration

### Agent Configuration
```python
from agent import AgentConfig

config = AgentConfig(
    max_search_results=20,        # Max papers per search
    max_ideas_generated=10,       # Max research ideas
    visualization_style='scientific',  # Visualization theme
    enable_interactive_viz=True,  # Enable interactive charts
    auto_save_results=True,       # Auto-save outputs
    output_directory='outputs'    # Output folder
)

agent = SciDetectiveAgent(config)
```

### Visualization Configuration
```python
from agent.visualization import VisualizationConfig

viz_config = VisualizationConfig(
    style='scientific',           # 'scientific', 'modern', 'minimal'
    color_palette='viridis',      # Matplotlib color palette
    figure_size=(12, 8),         # Figure dimensions
    dpi=300,                     # Image resolution
    interactive=True             # Interactive plots
)
```

## 🐛 Error Handling and Logging

The agent includes comprehensive error handling and logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Error handling example
try:
    result = agent.analyze_essay(file_path="nonexistent.pdf")
    if not result.success:
        print(f"Error: {result.error}")
        print(f"Message: {result.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📈 Performance Considerations

- **Rate Limiting**: Automatic API rate limiting for external services
- **Async Operations**: Non-blocking literature searches
- **Caching**: Session-based result caching
- **Memory Management**: Efficient handling of large documents
- **Parallel Processing**: Concurrent searches across multiple databases

## 🔒 API Keys and Configuration

Create a `.env` file in the agent directory:

```env
# Optional: API keys for enhanced functionality
SEMANTIC_SCHOLAR_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Model configuration
MODEL=deepseek/deepseek-chat
```

## 🤝 Contributing

The modular design makes it easy to extend functionality:

1. **Add new analysis methods** in `text_analysis.py`
2. **Extend idea generation** in `idea_generator.py`
3. **Add new data sources** in `web_searcher.py`
4. **Create custom visualizations** in `visualization.py`
5. **Add new commands** in `agent.py`

## 📚 Advanced Features

### Custom Analysis Patterns
```python
# Add custom blind spot detection patterns
analyzer = agent.text_analyzer
analyzer.weak_argument_indicators.append(r'\bunverified\s+claim\b')
```

### Custom Visualization Themes
```python
# Create custom color schemes
agent.visualizer.color_schemes['custom'] = ['#FF6B6B', '#4ECDC4', '#45B7D1']
```

### Cross-Domain Pattern Extension
```python
# Add new cross-domain patterns
agent.idea_generator.cross_domain_patterns['psychology_to_ai'] = [
    'cognitive bias modeling in machine learning',
    'emotional intelligence in artificial systems'
]
```

## 🎯 Use Cases

1. **Academic Researchers**: Analyze papers, find research gaps, generate ideas
2. **Graduate Students**: Improve thesis quality, discover new directions
3. **Research Groups**: Collaborative analysis, trend monitoring
4. **Scientific Writers**: Quality assessment, structure improvement
5. **Funding Agencies**: Evaluate research proposals, identify innovative ideas

## 📞 Support and Documentation

- **Help Command**: Use `agent.get_help()` for interactive assistance
- **Session Management**: Track analysis history with `get_session_data()`
- **Clear Session**: Reset state with `clear_session()`
- **Error Reporting**: Detailed error messages and suggested fixes

---

**SciDetective Agent v1.0.0** - Your AI-powered scientific research companion 🔬✨