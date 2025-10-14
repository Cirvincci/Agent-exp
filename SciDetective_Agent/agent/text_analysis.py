"""
Text Analysis Module for SciDetective_Agent

This module provides NLP-based analysis for scientific essays including:
- Structure detection (abstract, methods, results, discussion)
- Blind spot detection
- Argument strength assessment
- Content quality evaluation
"""

import re
import spacy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import PyPDF2
import docx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EssaySection:
    """Represents a section of a scientific essay"""
    title: str
    content: str
    start_position: int
    end_position: int
    confidence: float

@dataclass
class BlindSpot:
    """Represents a detected blind spot in the essay"""
    type: str  # 'logical_gap', 'missing_evidence', 'weak_argument', 'unexplored_angle'
    description: str
    location: str  # section or paragraph reference
    severity: str  # 'low', 'medium', 'high'
    suggestions: List[str]

@dataclass
class AnalysisResult:
    """Complete analysis result of an essay"""
    sections: List[EssaySection]
    blind_spots: List[BlindSpot]
    structure_quality: float
    argument_strength: float
    evidence_quality: float
    overall_score: float
    summary: str

class TextAnalyzer:
    """
    Main text analysis class for scientific essays
    """

    def __init__(self):
        """Initialize the text analyzer with NLP models"""
        try:
            # Load spaCy model for English
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define section patterns for scientific papers
        self.section_patterns = {
            'abstract': [
                r'\babstract\b',
                r'\bsummary\b',
                r'\boverview\b'
            ],
            'introduction': [
                r'\bintroduction\b',
                r'\bbackground\b',
                r'\bmotivation\b'
            ],
            'methods': [
                r'\bmethods?\b',
                r'\bmethodology\b',
                r'\bexperimental\s+setup\b',
                r'\bapproach\b',
                r'\bprocedure\b'
            ],
            'results': [
                r'\bresults?\b',
                r'\bfindings?\b',
                r'\bobservations?\b',
                r'\bdata\b'
            ],
            'discussion': [
                r'\bdiscussion\b',
                r'\banalysis\b',
                r'\binterpretation\b'
            ],
            'conclusion': [
                r'\bconclusions?\b',
                r'\bsummary\b',
                r'\bfinal\s+remarks?\b'
            ],
            'references': [
                r'\breferences?\b',
                r'\bbibliography\b',
                r'\bcitations?\b'
            ]
        }

        # Keywords indicating weak arguments or missing evidence
        self.weak_argument_indicators = [
            r'\bmight\s+be\b',
            r'\bcould\s+be\b',
            r'\bpossibly\b',
            r'\bperhaps\b',
            r'\bit\s+seems\b',
            r'\bappears\s+to\b',
            r'\bsuggests?\s+that\b',
            r'\bwithout\s+evidence\b',
            r'\bno\s+data\b',
            r'\bunproven\b'
        ]

        # Keywords indicating strong evidence
        self.strong_evidence_indicators = [
            r'\bdemonstrated\b',
            r'\bproven\b',
            r'\bconfirmed\b',
            r'\bverified\b',
            r'\bestablished\b',
            r'\bstatistically\s+significant\b',
            r'\bp\s*<\s*0\.05\b',
            r'\bdata\s+shows?\b',
            r'\bresults\s+indicate\b'
        ]

    def load_document(self, file_path: str) -> str:
        """
        Load text from various document formats

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content
        """
        try:
            if file_path.lower().endswith('.pdf'):
                return self._load_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                return self._load_docx(file_path)
            elif file_path.lower().endswith('.txt'):
                return self._load_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def _load_pdf(self, file_path: str) -> str:
        """Load text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
        return text

    def _load_docx(self, file_path: str) -> str:
        """Load text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error reading DOCX: {str(e)}")
            raise
        return text

    def _load_txt(self, file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT: {str(e)}")
            raise

    def detect_sections(self, text: str) -> List[EssaySection]:
        """
        Detect and extract sections from the essay text

        Args:
            text: The essay text to analyze

        Returns:
            List of detected sections
        """
        sections = []
        text_lower = text.lower()

        # Find section boundaries
        section_boundaries = []

        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in matches:
                    section_boundaries.append({
                        'name': section_name,
                        'start': match.start(),
                        'confidence': 0.8  # Base confidence
                    })

        # Sort boundaries by position
        section_boundaries.sort(key=lambda x: x['start'])

        # Extract section content
        for i, boundary in enumerate(section_boundaries):
            start_pos = boundary['start']

            # Find end position (start of next section or end of text)
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1]['start']
            else:
                end_pos = len(text)

            # Extract content
            content = text[start_pos:end_pos].strip()

            # Skip very short sections
            if len(content) < 50:
                continue

            sections.append(EssaySection(
                title=boundary['name'],
                content=content,
                start_position=start_pos,
                end_position=end_pos,
                confidence=boundary['confidence']
            ))

        return sections

    def detect_blind_spots(self, text: str, sections: List[EssaySection]) -> List[BlindSpot]:
        """
        Detect potential blind spots in the essay

        Args:
            text: The essay text
            sections: Detected sections

        Returns:
            List of detected blind spots
        """
        blind_spots = []

        # 1. Check for weak arguments
        blind_spots.extend(self._detect_weak_arguments(text, sections))

        # 2. Check for missing evidence
        blind_spots.extend(self._detect_missing_evidence(text, sections))

        # 3. Check for logical gaps
        blind_spots.extend(self._detect_logical_gaps(text, sections))

        # 4. Check for unexplored angles
        blind_spots.extend(self._detect_unexplored_angles(text, sections))

        return blind_spots

    def _detect_weak_arguments(self, text: str, sections: List[EssaySection]) -> List[BlindSpot]:
        """Detect weak or unsupported arguments"""
        blind_spots = []

        for pattern in self.weak_argument_indicators:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                # Find which section this belongs to
                section_name = self._find_section_for_position(match.start(), sections)

                blind_spots.append(BlindSpot(
                    type='weak_argument',
                    description=f"Weak argument indicator found: '{match.group()}'",
                    location=section_name or 'unknown section',
                    severity='medium',
                    suggestions=[
                        "Provide stronger evidence to support this claim",
                        "Use more definitive language if evidence supports it",
                        "Consider adding experimental data or citations"
                    ]
                ))

        return blind_spots

    def _detect_missing_evidence(self, text: str, sections: List[EssaySection]) -> List[BlindSpot]:
        """Detect claims that lack supporting evidence"""
        blind_spots = []

        # Look for strong claims without nearby evidence
        claim_patterns = [
            r'\b(proves?|demonstrates?|shows?|confirms?|establishes?)\s+that\b',
            r'\bclearly\s+\w+\b',
            r'\bobviously\s+\w+\b',
            r'\bundoubtedly\s+\w+\b'
        ]

        for pattern in claim_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                # Check if there's supporting evidence nearby
                context_start = max(0, match.start() - 200)
                context_end = min(len(text), match.end() + 200)
                context = text[context_start:context_end].lower()

                has_evidence = any(re.search(evidence_pattern, context)
                                 for evidence_pattern in self.strong_evidence_indicators)

                if not has_evidence:
                    section_name = self._find_section_for_position(match.start(), sections)

                    blind_spots.append(BlindSpot(
                        type='missing_evidence',
                        description=f"Strong claim without supporting evidence: '{match.group()}'",
                        location=section_name or 'unknown section',
                        severity='high',
                        suggestions=[
                            "Add statistical data to support this claim",
                            "Include relevant citations or references",
                            "Provide experimental evidence",
                            "Consider moderating the language if evidence is limited"
                        ]
                    ))

        return blind_spots

    def _detect_logical_gaps(self, text: str, sections: List[EssaySection]) -> List[BlindSpot]:
        """Detect logical inconsistencies or gaps in reasoning"""
        blind_spots = []

        # Check for contradictory statements
        contradiction_patterns = [
            (r'\bhowever\b', r'\bnevertheless\b'),
            (r'\balthough\b', r'\bbut\b'),
            (r'\bcontrary\s+to\b', r'\bin\s+contrast\b')
        ]

        # This is a simplified implementation
        # In a real system, you'd use more sophisticated NLP analysis

        paragraphs = text.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50:
                continue

            # Look for contradiction indicators
            has_contradiction = any(
                re.search(pattern1, paragraph, re.IGNORECASE) and
                re.search(pattern2, paragraph, re.IGNORECASE)
                for pattern1, pattern2 in contradiction_patterns
            )

            if has_contradiction:
                blind_spots.append(BlindSpot(
                    type='logical_gap',
                    description=f"Potential logical inconsistency in paragraph {i+1}",
                    location=f"paragraph {i+1}",
                    severity='medium',
                    suggestions=[
                        "Review the logical flow of arguments",
                        "Clarify the relationship between contrasting points",
                        "Ensure conclusions follow from premises"
                    ]
                ))

        return blind_spots

    def _detect_unexplored_angles(self, text: str, sections: List[EssaySection]) -> List[BlindSpot]:
        """Detect potentially unexplored research angles"""
        blind_spots = []

        # Look for sections that are notably short
        section_lengths = {section.title: len(section.content) for section in sections}
        avg_length = sum(section_lengths.values()) / len(section_lengths) if section_lengths else 0

        for section_name, length in section_lengths.items():
            if length < avg_length * 0.5 and length > 0:  # Significantly shorter than average
                blind_spots.append(BlindSpot(
                    type='unexplored_angle',
                    description=f"The {section_name} section appears underdeveloped",
                    location=section_name,
                    severity='medium',
                    suggestions=[
                        f"Expand the {section_name} section with more detail",
                        "Consider additional perspectives or approaches",
                        "Include more comprehensive analysis"
                    ]
                ))

        return blind_spots

    def _find_section_for_position(self, position: int, sections: List[EssaySection]) -> Optional[str]:
        """Find which section contains the given text position"""
        for section in sections:
            if section.start_position <= position <= section.end_position:
                return section.title
        return None

    def calculate_quality_scores(self, text: str, sections: List[EssaySection],
                                blind_spots: List[BlindSpot]) -> Tuple[float, float, float, float]:
        """
        Calculate quality scores for the essay

        Returns:
            Tuple of (structure_quality, argument_strength, evidence_quality, overall_score)
        """
        # Structure quality based on section completeness
        expected_sections = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
        found_sections = [section.title for section in sections]
        structure_quality = len(set(found_sections) & set(expected_sections)) / len(expected_sections)

        # Argument strength based on blind spots
        weak_argument_count = sum(1 for bs in blind_spots if bs.type == 'weak_argument')
        total_sentences = len(re.findall(r'[.!?]+', text))
        argument_strength = max(0, 1 - (weak_argument_count / max(total_sentences * 0.1, 1)))

        # Evidence quality based on evidence indicators
        strong_evidence_count = sum(1 for pattern in self.strong_evidence_indicators
                                  if re.search(pattern, text, re.IGNORECASE))
        missing_evidence_count = sum(1 for bs in blind_spots if bs.type == 'missing_evidence')
        evidence_quality = strong_evidence_count / max(strong_evidence_count + missing_evidence_count, 1)

        # Overall score
        overall_score = (structure_quality + argument_strength + evidence_quality) / 3

        return structure_quality, argument_strength, evidence_quality, overall_score

    def analyze_essay(self, text: str) -> AnalysisResult:
        """
        Perform complete analysis of an essay

        Args:
            text: The essay text to analyze

        Returns:
            Complete analysis result
        """
        logger.info("Starting essay analysis...")

        # Detect sections
        sections = self.detect_sections(text)
        logger.info(f"Detected {len(sections)} sections")

        # Detect blind spots
        blind_spots = self.detect_blind_spots(text, sections)
        logger.info(f"Detected {len(blind_spots)} potential blind spots")

        # Calculate quality scores
        structure_quality, argument_strength, evidence_quality, overall_score = \
            self.calculate_quality_scores(text, sections, blind_spots)

        # Generate summary
        summary = self._generate_summary(sections, blind_spots, overall_score)

        return AnalysisResult(
            sections=sections,
            blind_spots=blind_spots,
            structure_quality=structure_quality,
            argument_strength=argument_strength,
            evidence_quality=evidence_quality,
            overall_score=overall_score,
            summary=summary
        )

    def _generate_summary(self, sections: List[EssaySection], blind_spots: List[BlindSpot],
                         overall_score: float) -> str:
        """Generate a summary of the analysis"""
        summary_parts = []

        # Overall assessment
        if overall_score >= 0.8:
            summary_parts.append("The essay demonstrates strong structure and argumentation.")
        elif overall_score >= 0.6:
            summary_parts.append("The essay has good foundation but could benefit from improvements.")
        else:
            summary_parts.append("The essay has significant areas for improvement.")

        # Section analysis
        if sections:
            summary_parts.append(f"Identified {len(sections)} main sections: {', '.join([s.title for s in sections[:5]])}.")

        # Blind spot summary
        if blind_spots:
            blind_spot_types = {}
            for bs in blind_spots:
                blind_spot_types[bs.type] = blind_spot_types.get(bs.type, 0) + 1

            summary_parts.append(f"Found {len(blind_spots)} areas for improvement:")
            for bs_type, count in blind_spot_types.items():
                summary_parts.append(f"- {count} {bs_type.replace('_', ' ')} issues")
        else:
            summary_parts.append("No significant blind spots detected.")

        return " ".join(summary_parts)