"""
Intelligent Match Analyzer with LLM-Powered Analysis

Features:
- Multi-strategy matching (exact, fuzzy, composite)
- Match scoring and ranking
- LLM-powered match recommendation
- Batch processing with concurrency
- Detailed match explanations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from difflib import SequenceMatcher

from src.core.reltio_client import ReltioClient, ReltioConfig
from src.parsers.file_parser import ParsedRecord

logger = logging.getLogger(__name__)


class MatchConfidence(Enum):
    """Match confidence levels"""
    EXACT = "exact"           # 100% match on unique identifier
    HIGH = "high"             # 90-99% confidence
    MEDIUM = "medium"         # 70-89% confidence
    LOW = "low"               # 50-69% confidence
    UNCERTAIN = "uncertain"   # < 50% confidence
    NO_MATCH = "no_match"     # No matches found


class MatchReason(Enum):
    """Reasons for match determination"""
    NPI_EXACT = "Exact NPI match"
    DEA_EXACT = "Exact DEA match"
    NAME_EXACT = "Exact name match"
    NAME_FUZZY = "Fuzzy name match"
    NAME_ADDRESS = "Name + Address match"
    NAME_SPECIALTY = "Name + Specialty match"
    COMPOSITE = "Multiple attribute match"
    LLM_ANALYSIS = "LLM-powered analysis"
    USER_SPECIFIED = "User-specified merge target"


@dataclass
class MatchCandidate:
    """A potential match candidate from Reltio"""
    entity_uri: str
    entity_label: str
    entity_type: str
    attributes: Dict[str, Any]
    match_score: float  # 0-100
    confidence: MatchConfidence
    match_reasons: List[MatchReason]
    matched_attributes: Dict[str, Tuple[Any, Any]]  # attr -> (input_value, reltio_value)
    reltio_match_score: Optional[float] = None  # Score from Reltio's match engine


@dataclass
class MatchResult:
    """Complete match analysis result for a single input record"""
    input_record: ParsedRecord
    candidates: List[MatchCandidate]
    best_match: Optional[MatchCandidate]
    recommendation: str  # merge, review, skip, no_match
    llm_analysis: Optional[str] = None
    processing_time_ms: float = 0
    errors: List[str] = field(default_factory=list)


class MatchScorer:
    """Calculate match scores between input attributes and Reltio entities"""

    # Attribute weights for scoring
    ATTRIBUTE_WEIGHTS = {
        "NPI": 100,          # Unique identifier - highest weight
        "DEA": 90,           # Strong identifier
        "LicenseNumber": 80,
        "LastName": 60,
        "FirstName": 50,
        "Specialty": 40,
        "City": 30,
        "State": 25,
        "PostalCode": 35,
        "Phone": 30,
        "Email": 45,
        "Organization": 35,
    }

    @classmethod
    def calculate_similarity(cls, value1: Any, value2: Any) -> float:
        """Calculate similarity between two values (0-1)"""
        if value1 is None or value2 is None:
            return 0.0

        str1 = str(value1).lower().strip()
        str2 = str(value2).lower().strip()

        if not str1 or not str2:
            return 0.0

        # Exact match
        if str1 == str2:
            return 1.0

        # Fuzzy match using SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()

    @classmethod
    def score_match(
        cls,
        input_attrs: Dict[str, Any],
        reltio_attrs: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Tuple[Any, Any]], List[MatchReason]]:
        """
        Score a match between input attributes and Reltio entity

        Returns:
            Tuple of (score, matched_attributes, reasons)
        """
        total_weight = 0
        weighted_score = 0
        matched_attrs = {}
        reasons = []

        # Flatten Reltio attributes (they come nested)
        flat_reltio = cls._flatten_reltio_attrs(reltio_attrs)

        for attr, input_value in input_attrs.items():
            if input_value is None:
                continue

            weight = cls.ATTRIBUTE_WEIGHTS.get(attr, 20)
            reltio_value = flat_reltio.get(attr)

            if reltio_value is not None:
                similarity = cls.calculate_similarity(input_value, reltio_value)

                if similarity > 0.5:  # Only count meaningful matches
                    matched_attrs[attr] = (input_value, reltio_value)
                    weighted_score += weight * similarity
                    total_weight += weight

                    # Determine reason
                    if attr == "NPI" and similarity == 1.0:
                        reasons.append(MatchReason.NPI_EXACT)
                    elif attr == "DEA" and similarity == 1.0:
                        reasons.append(MatchReason.DEA_EXACT)
                    elif attr in ["FirstName", "LastName"] and similarity == 1.0:
                        reasons.append(MatchReason.NAME_EXACT)
                    elif attr in ["FirstName", "LastName"] and similarity > 0.8:
                        reasons.append(MatchReason.NAME_FUZZY)

        # Calculate final score
        if total_weight > 0:
            score = (weighted_score / total_weight) * 100
        else:
            score = 0

        # Add composite reason if multiple matches
        if len(matched_attrs) >= 3 and MatchReason.COMPOSITE not in reasons:
            reasons.append(MatchReason.COMPOSITE)

        return score, matched_attrs, reasons

    @classmethod
    def _flatten_reltio_attrs(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested Reltio attribute structure"""
        flat = {}

        if not attrs:
            return flat

        for attr_name, attr_values in attrs.items():
            if isinstance(attr_values, list) and attr_values:
                # Get OV or first value
                for av in attr_values:
                    if isinstance(av, dict):
                        if av.get("ov", False) or len(attr_values) == 1:
                            value = av.get("value")
                            if isinstance(value, dict):
                                # Nested attribute
                                for sub_name, sub_values in value.items():
                                    if isinstance(sub_values, list) and sub_values:
                                        for sv in sub_values:
                                            if isinstance(sv, dict) and sv.get("value"):
                                                flat[sub_name] = sv["value"]
                                                break
                            else:
                                flat[attr_name] = value
                            break

        return flat

    @classmethod
    def determine_confidence(cls, score: float, reasons: List[MatchReason]) -> MatchConfidence:
        """Determine confidence level based on score and reasons"""
        # Exact identifier match = highest confidence
        if MatchReason.NPI_EXACT in reasons or MatchReason.DEA_EXACT in reasons:
            return MatchConfidence.EXACT

        if score >= 90:
            return MatchConfidence.HIGH
        elif score >= 70:
            return MatchConfidence.MEDIUM
        elif score >= 50:
            return MatchConfidence.LOW
        elif score > 0:
            return MatchConfidence.UNCERTAIN
        else:
            return MatchConfidence.NO_MATCH


class LLMAnalyzer:
    """LLM-powered match analysis for intelligent recommendations"""

    def __init__(self, api_key: str, provider: str = "openai"):
        """
        Initialize LLM analyzer

        Args:
            api_key: API key for LLM provider
            provider: 'openai' or 'anthropic'
        """
        self.api_key = api_key
        self.provider = provider
        self._client = None

    async def analyze_match(
        self,
        input_record: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[int]]:
        """
        Use LLM to analyze potential matches and recommend action

        Args:
            input_record: The input HCP record
            candidates: List of potential matches from Reltio

        Returns:
            Tuple of (analysis_text, recommended_candidate_index or None)
        """
        if not candidates:
            return "No potential matches found in Reltio.", None

        # Build prompt
        prompt = self._build_analysis_prompt(input_record, candidates)

        try:
            if self.provider == "openai":
                return await self._analyze_with_openai(prompt)
            elif self.provider == "anthropic":
                return await self._analyze_with_anthropic(prompt)
            else:
                return "LLM analysis not configured.", None
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"LLM analysis unavailable: {str(e)}", None

    def _build_analysis_prompt(
        self,
        input_record: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> str:
        """Build the analysis prompt for the LLM"""
        prompt = """You are a Healthcare Data Quality expert analyzing potential HCP (Healthcare Professional) matches.

INPUT RECORD (from Data Steward's file):
"""
        prompt += json.dumps(input_record, indent=2)

        prompt += """

POTENTIAL MATCHES FROM RELTIO MDM:
"""
        for i, candidate in enumerate(candidates, 1):
            prompt += f"\n--- Candidate {i} ---\n"
            prompt += json.dumps(candidate, indent=2)

        prompt += """

ANALYSIS TASK:
1. Compare the input record against each candidate
2. Consider: NPI (unique identifier), Name similarity, Specialty, Location, Credentials
3. Account for data quality issues: typos, abbreviations, formatting differences
4. Determine if any candidate is likely the SAME PERSON as the input record

RESPOND WITH:
1. Brief analysis (2-3 sentences) explaining your reasoning
2. RECOMMENDATION: One of [MERGE_WITH_1, MERGE_WITH_2, ..., REVIEW_MANUALLY, NO_MATCH]
3. CONFIDENCE: One of [HIGH, MEDIUM, LOW]

Format your response as:
ANALYSIS: <your analysis>
RECOMMENDATION: <recommendation>
CONFIDENCE: <confidence level>
"""
        return prompt

    async def _analyze_with_openai(self, prompt: str) -> Tuple[str, Optional[int]]:
        """Analyze using OpenAI"""
        try:
            import openai
        except ImportError:
            return "OpenAI package not installed", None

        client = openai.AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a healthcare data quality expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        result = response.choices[0].message.content
        recommended_idx = self._parse_recommendation(result)

        return result, recommended_idx

    async def _analyze_with_anthropic(self, prompt: str) -> Tuple[str, Optional[int]]:
        """Analyze using Anthropic"""
        try:
            import anthropic
        except ImportError:
            return "Anthropic package not installed", None

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text
        recommended_idx = self._parse_recommendation(result)

        return result, recommended_idx

    def _parse_recommendation(self, llm_response: str) -> Optional[int]:
        """Parse LLM response to extract recommended candidate index"""
        # Look for MERGE_WITH_N pattern
        match = re.search(r'MERGE_WITH_(\d+)', llm_response)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-indexed

        return None


class MatchAnalyzer:
    """
    Main match analysis engine

    Combines:
    - Direct Reltio search
    - Attribute-based scoring
    - Optional LLM analysis
    """

    def __init__(
        self,
        reltio_config: ReltioConfig,
        llm_api_key: Optional[str] = None,
        llm_provider: str = "openai",
        use_llm: bool = True
    ):
        """
        Initialize match analyzer

        Args:
            reltio_config: Reltio connection configuration
            llm_api_key: Optional API key for LLM analysis
            llm_provider: 'openai' or 'anthropic'
            use_llm: Whether to use LLM for analysis (requires api_key)
        """
        self.reltio_config = reltio_config
        self.scorer = MatchScorer()
        self.use_llm = use_llm and llm_api_key is not None

        if self.use_llm:
            self.llm_analyzer = LLMAnalyzer(llm_api_key, llm_provider)
        else:
            self.llm_analyzer = None

    async def analyze_record(
        self,
        record: ParsedRecord,
        reltio_client: ReltioClient,
        entity_type: str = "HCP"
    ) -> MatchResult:
        """
        Analyze a single record and find potential matches

        Args:
            record: Parsed input record
            reltio_client: Active Reltio client
            entity_type: Entity type to search

        Returns:
            MatchResult with candidates and recommendation
        """
        import time
        start_time = time.time()

        candidates = []
        errors = []

        try:
            # Strategy 1: If merge target specified, verify it exists
            if record.merge_target:
                target_entity = await self._verify_merge_target(
                    record.merge_target, reltio_client
                )
                if target_entity:
                    candidate = self._create_candidate_from_entity(
                        target_entity, record, [MatchReason.USER_SPECIFIED]
                    )
                    candidate.match_score = 100
                    candidate.confidence = MatchConfidence.EXACT
                    candidates.append(candidate)

            # Strategy 2: Search by identifiers (NPI, DEA, etc.)
            if record.identifiers and not candidates:
                id_candidates = await self._search_by_identifiers(
                    record, reltio_client, entity_type
                )
                candidates.extend(id_candidates)

            # Strategy 3: Search by attributes if no identifier matches
            if not candidates:
                attr_candidates = await self._search_by_attributes(
                    record, reltio_client, entity_type
                )
                candidates.extend(attr_candidates)

            # Deduplicate and sort by score
            candidates = self._deduplicate_candidates(candidates)
            candidates.sort(key=lambda c: c.match_score, reverse=True)

            # Limit to top candidates
            candidates = candidates[:10]

            # Optional: LLM analysis for top candidates
            llm_analysis = None
            if self.use_llm and candidates and self.llm_analyzer:
                llm_analysis, llm_recommended_idx = await self._run_llm_analysis(
                    record, candidates
                )

                # Update best match based on LLM recommendation
                if llm_recommended_idx is not None and llm_recommended_idx < len(candidates):
                    candidates[llm_recommended_idx].match_reasons.append(MatchReason.LLM_ANALYSIS)
                    # Boost the LLM-recommended candidate
                    candidates[llm_recommended_idx].match_score = min(
                        candidates[llm_recommended_idx].match_score + 10, 100
                    )
                    candidates.sort(key=lambda c: c.match_score, reverse=True)

        except Exception as e:
            logger.error(f"Error analyzing record {record.row_number}: {e}")
            errors.append(str(e))

        # Determine best match and recommendation
        best_match = candidates[0] if candidates else None
        recommendation = self._determine_recommendation(best_match)

        processing_time = (time.time() - start_time) * 1000

        return MatchResult(
            input_record=record,
            candidates=candidates,
            best_match=best_match,
            recommendation=recommendation,
            llm_analysis=llm_analysis,
            processing_time_ms=processing_time,
            errors=errors
        )

    async def analyze_batch(
        self,
        records: List[ParsedRecord],
        reltio_client: ReltioClient,
        entity_type: str = "HCP",
        on_progress: callable = None
    ) -> List[MatchResult]:
        """
        Analyze multiple records concurrently

        Args:
            records: List of parsed records
            reltio_client: Active Reltio client
            entity_type: Entity type
            on_progress: Optional callback(completed, total)

        Returns:
            List of MatchResults
        """
        results = []
        total = len(records)

        # Process in batches to control concurrency
        batch_size = self.reltio_config.max_concurrent_requests

        for i in range(0, total, batch_size):
            batch = records[i:i + batch_size]

            tasks = [
                self.analyze_record(record, reltio_client, entity_type)
                for record in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Create error result
                    results.append(MatchResult(
                        input_record=batch[j],
                        candidates=[],
                        best_match=None,
                        recommendation="error",
                        errors=[str(result)]
                    ))
                else:
                    results.append(result)

            if on_progress:
                on_progress(min(i + batch_size, total), total)

        return results

    async def _verify_merge_target(
        self,
        target_id: str,
        client: ReltioClient
    ) -> Optional[Dict[str, Any]]:
        """Verify a specified merge target exists"""
        try:
            return await client.get_entity(target_id)
        except Exception:
            return None

    async def _search_by_identifiers(
        self,
        record: ParsedRecord,
        client: ReltioClient,
        entity_type: str
    ) -> List[MatchCandidate]:
        """Search Reltio using record identifiers"""
        candidates = []

        for id_type, id_value in record.identifiers.items():
            if id_type == "EntityURI":
                continue  # Skip entity URIs

            # Build search filter
            if id_type == "NPI":
                filter_expr = f"equals(attributes.NPI,'{id_value}')"
            elif id_type == "DEA":
                filter_expr = f"equals(attributes.DEA,'{id_value}')"
            elif id_type == "SourceID":
                filter_expr = f"equals(crosswalks.value,'{id_value}')"
            else:
                continue

            try:
                results = await client.search_entities(
                    filter_expr=filter_expr,
                    entity_type=entity_type,
                    max_results=5
                )

                for entity in (results if isinstance(results, list) else []):
                    candidate = self._create_candidate_from_entity(
                        entity, record
                    )
                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Identifier search failed for {id_type}: {e}")

        return candidates

    async def _search_by_attributes(
        self,
        record: ParsedRecord,
        client: ReltioClient,
        entity_type: str
    ) -> List[MatchCandidate]:
        """Search Reltio using record attributes"""
        candidates = []

        # Build combined attribute search
        all_attrs = {**record.identifiers, **record.attributes}

        try:
            results = await client.find_matches_for_attributes(
                attributes=all_attrs,
                entity_type=entity_type
            )

            for entity in results:
                candidate = self._create_candidate_from_entity(entity, record)
                candidates.append(candidate)

        except Exception as e:
            logger.warning(f"Attribute search failed: {e}")

        return candidates

    def _create_candidate_from_entity(
        self,
        entity: Dict[str, Any],
        record: ParsedRecord,
        additional_reasons: List[MatchReason] = None
    ) -> MatchCandidate:
        """Create a MatchCandidate from a Reltio entity"""
        # Extract attributes
        attrs = entity.get("attributes", {})

        # Combine input identifiers and attributes for scoring
        input_attrs = {**record.identifiers, **record.attributes}

        # Calculate match score
        score, matched_attrs, reasons = self.scorer.score_match(input_attrs, attrs)

        if additional_reasons:
            reasons.extend(additional_reasons)

        confidence = self.scorer.determine_confidence(score, reasons)

        return MatchCandidate(
            entity_uri=entity.get("uri", ""),
            entity_label=entity.get("label", "Unknown"),
            entity_type=entity.get("type", "").split("/")[-1],
            attributes=attrs,
            match_score=score,
            confidence=confidence,
            match_reasons=reasons,
            matched_attributes=matched_attrs,
            reltio_match_score=entity.get("relevanceScores", {}).get("matchScore")
        )

    def _deduplicate_candidates(
        self,
        candidates: List[MatchCandidate]
    ) -> List[MatchCandidate]:
        """Remove duplicate candidates by entity URI"""
        seen = set()
        unique = []

        for candidate in candidates:
            if candidate.entity_uri not in seen:
                seen.add(candidate.entity_uri)
                unique.append(candidate)

        return unique

    def _determine_recommendation(
        self,
        best_match: Optional[MatchCandidate]
    ) -> str:
        """Determine recommended action based on best match"""
        if not best_match:
            return "no_match"

        if best_match.confidence == MatchConfidence.EXACT:
            return "merge"
        elif best_match.confidence == MatchConfidence.HIGH:
            return "merge"
        elif best_match.confidence == MatchConfidence.MEDIUM:
            return "review"
        elif best_match.confidence == MatchConfidence.LOW:
            return "review"
        else:
            return "skip"

    async def _run_llm_analysis(
        self,
        record: ParsedRecord,
        candidates: List[MatchCandidate]
    ) -> Tuple[Optional[str], Optional[int]]:
        """Run LLM analysis on candidates"""
        if not self.llm_analyzer:
            return None, None

        # Prepare data for LLM
        input_data = {**record.identifiers, **record.attributes}

        candidate_data = []
        for c in candidates[:5]:  # Limit to top 5 for LLM
            flat_attrs = MatchScorer._flatten_reltio_attrs(c.attributes)
            candidate_data.append({
                "uri": c.entity_uri,
                "label": c.entity_label,
                "match_score": c.match_score,
                "attributes": flat_attrs
            })

        return await self.llm_analyzer.analyze_match(input_data, candidate_data)
