"""
Idea Generator Module for SciDetective_Agent

This module generates creative and feasible research ideas using:
- Cross-domain knowledge synthesis
- Pattern recognition from literature
- Hypothesis generation
- Research direction suggestions
"""

import re
import random
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResearchIdea:
    """Represents a generated research idea"""
    title: str
    description: str
    methodology: str
    feasibility: str  # 'high', 'medium', 'low'
    novelty: str  # 'high', 'medium', 'low'
    potential_impact: str  # 'high', 'medium', 'low'
    required_resources: List[str]
    related_fields: List[str]
    estimated_timeline: str
    key_challenges: List[str]

@dataclass
class ResearchGap:
    """Represents an identified research gap"""
    description: str
    gap_type: str  # 'methodological', 'theoretical', 'empirical', 'technological'
    importance: str  # 'high', 'medium', 'low'
    research_questions: List[str]

class IdeaGenerator:
    """
    Generates creative research ideas and identifies research gaps
    """

    def __init__(self):
        """Initialize the idea generator with knowledge bases"""

        # Cross-domain knowledge patterns
        self.cross_domain_patterns = {
            'physics_to_biology': [
                'quantum mechanics principles in biological systems',
                'thermodynamics in cellular processes',
                'wave mechanics in neural networks',
                'phase transitions in ecosystem dynamics'
            ],
            'computer_science_to_medicine': [
                'machine learning for drug discovery',
                'algorithmic approaches to disease diagnosis',
                'network analysis in epidemiology',
                'optimization methods in treatment planning'
            ],
            'materials_science_to_energy': [
                'novel materials for energy storage',
                'biomimetic approaches to energy conversion',
                'nanomaterials for solar cells',
                'smart materials for energy harvesting'
            ],
            'psychology_to_economics': [
                'behavioral factors in market dynamics',
                'cognitive biases in decision making',
                'social psychology in financial systems',
                'neuroscience applications in economics'
            ],
            'ecology_to_engineering': [
                'bio-inspired design principles',
                'ecosystem models for system optimization',
                'natural selection algorithms',
                'swarm intelligence applications'
            ]
        }

        # Research methodology templates
        self.methodology_templates = {
            'experimental': [
                'controlled laboratory experiments with {variable} manipulation',
                'field studies measuring {outcome} under {conditions}',
                'randomized controlled trials comparing {treatment} vs {control}',
                'longitudinal studies tracking {parameter} over {timeframe}'
            ],
            'computational': [
                'machine learning models trained on {dataset}',
                'simulation studies using {model} framework',
                'data mining analysis of {data_source}',
                'mathematical modeling of {system} dynamics'
            ],
            'observational': [
                'cross-sectional analysis of {population}',
                'retrospective study of {historical_data}',
                'meta-analysis of existing {research_area} studies',
                'systematic review of {topic} literature'
            ],
            'mixed_methods': [
                'combined quantitative and qualitative analysis',
                'multi-scale modeling approach',
                'interdisciplinary collaboration study',
                'triangulation of {method1}, {method2}, and {method3}'
            ]
        }

        # Innovation triggers
        self.innovation_triggers = [
            'What if we applied {technique} from {field1} to {field2}?',
            'How might {recent_technology} change our understanding of {phenomenon}?',
            'Could {biological_process} inspire new approaches to {engineering_problem}?',
            'What would happen if we combined {method1} with {method2}?',
            'How can we address the limitation of {current_approach}?',
            'What new insights emerge when viewing {problem} through {perspective} lens?'
        ]

        # Research field keywords
        self.field_keywords = {
            'artificial_intelligence': ['machine learning', 'neural networks', 'deep learning', 'reinforcement learning'],
            'biotechnology': ['genetic engineering', 'protein synthesis', 'biomarkers', 'cell therapy'],
            'materials_science': ['nanomaterials', 'composites', 'superconductors', 'metamaterials'],
            'environmental_science': ['climate change', 'pollution', 'conservation', 'sustainability'],
            'neuroscience': ['brain imaging', 'neural plasticity', 'cognition', 'consciousness'],
            'energy': ['renewable energy', 'battery technology', 'fuel cells', 'energy efficiency'],
            'medicine': ['personalized medicine', 'immunotherapy', 'diagnostics', 'drug delivery'],
            'robotics': ['autonomous systems', 'human-robot interaction', 'bio-inspired robotics', 'swarm robotics']
        }

        # Current research trends (these would ideally be updated from literature)
        self.trending_topics = [
            'quantum computing applications',
            'CRISPR gene editing ethics',
            'artificial general intelligence',
            'personalized medicine',
            'sustainable materials',
            'space colonization technology',
            'brain-computer interfaces',
            'autonomous vehicle safety'
        ]

    def extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract key scientific concepts from text

        Args:
            text: Input text to analyze

        Returns:
            List of extracted key concepts
        """
        # Simple keyword extraction - in practice, use more sophisticated NLP
        concepts = []

        # Scientific terms patterns
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:algorithm|method|model|theory|principle)\b',
            r'\b(?:quantum|molecular|cellular|neural|genetic|computational)\s+\w+\b',
            r'\b\w+(?:ology|ics|ism|tion|ity|ness)\b',
            r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\s+(?:effect|law|theorem|hypothesis)\b'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([match.lower().strip() for match in matches])

        # Remove duplicates and sort by frequency
        concept_counts = Counter(concepts)
        return [concept for concept, count in concept_counts.most_common(20)]

    def identify_research_gaps(self, text: str, field: str) -> List[ResearchGap]:
        """
        Identify potential research gaps from the text

        Args:
            text: Research text to analyze
            field: Primary research field

        Returns:
            List of identified research gaps
        """
        gaps = []

        # Look for gap indicators
        gap_indicators = [
            r'(?:limited|insufficient|lack of|no)\s+(?:research|studies|data|understanding)',
            r'(?:future|further|additional)\s+(?:research|investigation|study)\s+(?:is\s+)?(?:needed|required)',
            r'(?:remains|still)\s+(?:unclear|unknown|unexplored)',
            r'(?:potential|opportunity)\s+for\s+(?:research|investigation)',
            r'(?:gap|limitation|weakness)\s+in\s+(?:current|existing)\s+(?:knowledge|understanding)'
        ]

        for pattern in gap_indicators:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                # Extract context around the gap
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()

                # Determine gap type based on context
                gap_type = self._classify_gap_type(context)

                gaps.append(ResearchGap(
                    description=f"Research gap identified: {context[:200]}...",
                    gap_type=gap_type,
                    importance=self._assess_gap_importance(context, field),
                    research_questions=self._generate_research_questions(context, field)
                ))

        return gaps

    def _classify_gap_type(self, context: str) -> str:
        """Classify the type of research gap"""
        context_lower = context.lower()

        if any(word in context_lower for word in ['method', 'technique', 'approach', 'procedure']):
            return 'methodological'
        elif any(word in context_lower for word in ['theory', 'model', 'framework', 'concept']):
            return 'theoretical'
        elif any(word in context_lower for word in ['data', 'experiment', 'observation', 'measurement']):
            return 'empirical'
        elif any(word in context_lower for word in ['technology', 'tool', 'instrument', 'device']):
            return 'technological'
        else:
            return 'general'

    def _assess_gap_importance(self, context: str, field: str) -> str:
        """Assess the importance of a research gap"""
        # Simple heuristic - in practice, use more sophisticated analysis
        context_lower = context.lower()

        high_importance_indicators = ['critical', 'essential', 'urgent', 'significant', 'major']
        medium_importance_indicators = ['important', 'relevant', 'useful', 'beneficial']

        if any(indicator in context_lower for indicator in high_importance_indicators):
            return 'high'
        elif any(indicator in context_lower for indicator in medium_importance_indicators):
            return 'medium'
        else:
            return 'low'

    def _generate_research_questions(self, context: str, field: str) -> List[str]:
        """Generate research questions based on context"""
        questions = []

        # Simple question generation templates
        question_templates = [
            f"How can we better understand the mechanisms underlying {field}?",
            f"What new methodologies could advance research in {field}?",
            f"How might emerging technologies impact {field}?",
            f"What are the long-term implications of current findings in {field}?"
        ]

        # Select relevant questions
        questions.extend(random.sample(question_templates, min(2, len(question_templates))))

        return questions

    def generate_cross_domain_ideas(self, primary_field: str, concepts: List[str]) -> List[ResearchIdea]:
        """
        Generate research ideas by combining concepts across domains

        Args:
            primary_field: The main research field
            concepts: Key concepts from the research

        Returns:
            List of generated research ideas
        """
        ideas = []

        # Find relevant cross-domain patterns
        relevant_patterns = []
        for pattern_key, patterns in self.cross_domain_patterns.items():
            if any(keyword in primary_field.lower() for keyword in pattern_key.split('_')):
                relevant_patterns.extend(patterns)

        # Generate ideas based on cross-domain patterns
        for pattern in relevant_patterns[:5]:  # Limit to top 5
            idea = self._create_research_idea_from_pattern(pattern, primary_field, concepts)
            if idea:
                ideas.append(idea)

        # Generate ideas based on methodology combinations
        for methodology_type, templates in self.methodology_templates.items():
            template = random.choice(templates)
            idea = self._create_methodology_based_idea(template, primary_field, concepts)
            if idea:
                ideas.append(idea)

        return ideas

    def _create_research_idea_from_pattern(self, pattern: str, field: str, concepts: List[str]) -> Optional[ResearchIdea]:
        """Create a research idea based on a cross-domain pattern"""
        # Select relevant concepts
        selected_concepts = random.sample(concepts, min(3, len(concepts))) if concepts else []

        # Generate title
        title = f"Applying {pattern} to {field}"
        if selected_concepts:
            title += f" focusing on {', '.join(selected_concepts[:2])}"

        # Generate description
        description = f"This research explores how {pattern} can be leveraged to advance understanding in {field}."
        if selected_concepts:
            description += f" Specifically, we investigate the role of {', '.join(selected_concepts)} in this context."

        # Assess feasibility, novelty, and impact
        feasibility = self._assess_feasibility(pattern, field)
        novelty = self._assess_novelty(pattern, field)
        impact = self._assess_potential_impact(pattern, field)

        return ResearchIdea(
            title=title,
            description=description,
            methodology=self._suggest_methodology(pattern, field),
            feasibility=feasibility,
            novelty=novelty,
            potential_impact=impact,
            required_resources=self._identify_required_resources(pattern, field),
            related_fields=self._identify_related_fields(pattern, field),
            estimated_timeline=self._estimate_timeline(feasibility, novelty),
            key_challenges=self._identify_challenges(pattern, field)
        )

    def _create_methodology_based_idea(self, template: str, field: str, concepts: List[str]) -> Optional[ResearchIdea]:
        """Create a research idea based on methodology template"""
        # Fill in template variables
        variables = {
            'variable': random.choice(concepts) if concepts else 'key parameter',
            'outcome': f'{field} performance metrics',
            'conditions': 'controlled laboratory conditions',
            'treatment': 'novel intervention approach',
            'control': 'standard treatment protocol',
            'parameter': random.choice(concepts) if concepts else 'system parameter',
            'timeframe': '12-month period',
            'dataset': f'{field} research database',
            'model': 'computational framework',
            'data_source': f'{field} literature',
            'system': f'{field} system',
            'population': f'{field} research subjects',
            'historical_data': f'historical {field} records',
            'research_area': field,
            'topic': field,
            'method1': 'experimental analysis',
            'method2': 'computational modeling',
            'method3': 'theoretical framework'
        }

        methodology = template
        for var, value in variables.items():
            methodology = methodology.replace(f'{{{var}}}', value)

        title = f"Novel {field} research using {methodology.split()[0]} approach"
        description = f"This study proposes {methodology} to advance {field} research."

        return ResearchIdea(
            title=title,
            description=description,
            methodology=methodology,
            feasibility='medium',
            novelty='medium',
            potential_impact='medium',
            required_resources=['research team', 'equipment', 'funding'],
            related_fields=[field, 'methodology'],
            estimated_timeline='6-18 months',
            key_challenges=['data collection', 'methodology validation']
        )

    def _assess_feasibility(self, pattern: str, field: str) -> str:
        """Assess the feasibility of a research idea"""
        # Simple heuristic - in practice, use more sophisticated analysis
        if 'quantum' in pattern.lower() or 'advanced' in pattern.lower():
            return 'low'
        elif 'computational' in pattern.lower() or 'analysis' in pattern.lower():
            return 'high'
        else:
            return 'medium'

    def _assess_novelty(self, pattern: str, field: str) -> str:
        """Assess the novelty of a research idea"""
        # Check against trending topics
        pattern_lower = pattern.lower()
        if any(trend in pattern_lower for trend in [t.lower() for t in self.trending_topics]):
            return 'medium'  # Trending but not necessarily novel
        else:
            return 'high'  # Assume novel if not trending

    def _assess_potential_impact(self, pattern: str, field: str) -> str:
        """Assess the potential impact of a research idea"""
        # Heuristic based on field importance and pattern scope
        high_impact_fields = ['medicine', 'energy', 'environment', 'ai']
        if any(hif in field.lower() for hif in high_impact_fields):
            return 'high'
        else:
            return 'medium'

    def _suggest_methodology(self, pattern: str, field: str) -> str:
        """Suggest appropriate methodology for the research idea"""
        if 'computational' in pattern.lower():
            return 'Computational modeling and simulation studies'
        elif 'biological' in pattern.lower():
            return 'Experimental laboratory studies with biological systems'
        elif 'engineering' in pattern.lower():
            return 'Prototype development and testing'
        else:
            return 'Mixed-methods approach combining theoretical and empirical research'

    def _identify_required_resources(self, pattern: str, field: str) -> List[str]:
        """Identify required resources for the research"""
        base_resources = ['research team', 'funding', 'literature access']

        if 'computational' in pattern.lower():
            base_resources.extend(['high-performance computing', 'software licenses'])
        if 'experimental' in pattern.lower():
            base_resources.extend(['laboratory equipment', 'experimental materials'])
        if 'biological' in pattern.lower():
            base_resources.extend(['biological samples', 'specialized lab facilities'])

        return list(set(base_resources))

    def _identify_related_fields(self, pattern: str, field: str) -> List[str]:
        """Identify related research fields"""
        related = [field]

        pattern_lower = pattern.lower()
        if 'quantum' in pattern_lower:
            related.append('physics')
        if 'biological' in pattern_lower:
            related.append('biology')
        if 'computational' in pattern_lower:
            related.append('computer science')
        if 'materials' in pattern_lower:
            related.append('materials science')

        return list(set(related))

    def _estimate_timeline(self, feasibility: str, novelty: str) -> str:
        """Estimate research timeline based on feasibility and novelty"""
        if feasibility == 'low' or novelty == 'high':
            return '2-5 years'
        elif feasibility == 'medium':
            return '1-3 years'
        else:
            return '6-18 months'

    def _identify_challenges(self, pattern: str, field: str) -> List[str]:
        """Identify key challenges for the research"""
        challenges = ['funding acquisition', 'team coordination']

        pattern_lower = pattern.lower()
        if 'quantum' in pattern_lower:
            challenges.append('technical complexity')
        if 'biological' in pattern_lower:
            challenges.append('ethical considerations')
        if 'computational' in pattern_lower:
            challenges.append('computational resources')

        return challenges

    def generate_hypothesis(self, field: str, concepts: List[str], gaps: List[ResearchGap]) -> List[str]:
        """
        Generate research hypotheses based on field, concepts, and gaps

        Args:
            field: Research field
            concepts: Key concepts
            gaps: Identified research gaps

        Returns:
            List of generated hypotheses
        """
        hypotheses = []

        # Hypothesis templates
        templates = [
            "We hypothesize that {concept1} significantly influences {concept2} in {field} systems.",
            "We propose that combining {concept1} with {concept2} will enhance {field} outcomes.",
            "We predict that {concept1} acts as a mediator between {concept2} and {field} performance.",
            "We hypothesize that novel applications of {concept1} can address current limitations in {field}.",
            "We propose that {concept1}-based approaches will outperform traditional {field} methods."
        ]

        # Generate hypotheses from concepts
        if len(concepts) >= 2:
            for template in templates:
                concept1 = random.choice(concepts)
                concept2 = random.choice([c for c in concepts if c != concept1])

                hypothesis = template.format(
                    concept1=concept1,
                    concept2=concept2,
                    field=field
                )
                hypotheses.append(hypothesis)

        # Generate hypotheses from research gaps
        for gap in gaps[:3]:  # Limit to top 3 gaps
            hypothesis = f"We hypothesize that addressing the {gap.gap_type} gap in {gap.description[:50]}... will significantly advance {field} research."
            hypotheses.append(hypothesis)

        return hypotheses

    def generate_ideas(self, text: str, field: str, max_ideas: int = 10) -> Tuple[List[ResearchIdea], List[ResearchGap], List[str]]:
        """
        Main method to generate research ideas

        Args:
            text: Research text to analyze
            field: Primary research field
            max_ideas: Maximum number of ideas to generate

        Returns:
            Tuple of (research_ideas, research_gaps, hypotheses)
        """
        logger.info(f"Generating research ideas for field: {field}")

        # Extract key concepts
        concepts = self.extract_key_concepts(text)
        logger.info(f"Extracted {len(concepts)} key concepts")

        # Identify research gaps
        gaps = self.identify_research_gaps(text, field)
        logger.info(f"Identified {len(gaps)} research gaps")

        # Generate cross-domain ideas
        ideas = self.generate_cross_domain_ideas(field, concepts)

        # Limit to max_ideas
        ideas = ideas[:max_ideas]

        # Generate hypotheses
        hypotheses = self.generate_hypothesis(field, concepts, gaps)

        logger.info(f"Generated {len(ideas)} research ideas and {len(hypotheses)} hypotheses")

        return ideas, gaps, hypotheses