"""
query_processor.py
Query preprocessing utilities including fiscal period extraction and query expansion.

This module addresses the temporal query handling issue where the system
retrieved data from wrong fiscal periods due to semantic similarity.
"""

import re
from typing import Dict, List, Optional, Tuple


class FiscalPeriodExtractor:
    """
    Extract and normalize fiscal period references from queries.

    Microsoft Fiscal Calendar:
    - Fiscal year runs July 1 - June 30
    - Q1 = July-September
    - Q2 = October-December
    - Q3 = January-March
    - Q4 = April-June

    Examples:
        >>> extractor = FiscalPeriodExtractor()
        >>> extractor.extract("What was revenue in Q2 FY2025?")
        {'fiscal_year': 'FY2025', 'quarter': 'Q2', 'raw': 'Q2 FY2025'}
    """

    # Microsoft fiscal calendar mapping
    MSFT_FISCAL_CALENDAR = {
        'Q1': {'months': 'July-September', 'end_month': 'September'},
        'Q2': {'months': 'October-December', 'end_month': 'December'},
        'Q3': {'months': 'January-March', 'end_month': 'March'},
        'Q4': {'months': 'April-June', 'end_month': 'June'},
    }

    QUARTER_WORD_MAP = {
        'first': 'Q1', 'second': 'Q2', 'third': 'Q3', 'fourth': 'Q4',
        '1st': 'Q1', '2nd': 'Q2', '3rd': 'Q3', '4th': 'Q4',
    }

    def extract(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract fiscal period information from a query.

        Args:
            query: The user's question

        Returns:
            Dictionary with keys:
            - fiscal_year: e.g., 'FY2024', 'FY2025'
            - quarter: e.g., 'Q1', 'Q2', 'Q3', 'Q4'
            - raw: The combined period string, e.g., 'Q1 FY2025'
            - doc_type: Inferred document type ('10-K' or '10-Q')
        """
        query_lower = query.lower()
        result = {
            'fiscal_year': None,
            'quarter': None,
            'raw': None,
            'doc_type': None,
        }

        # Pattern 1: Q# FY#### (most specific, highest priority)
        match = re.search(r'q([1-4])\s*fy\s*(\d{4})', query_lower)
        if match:
            result['quarter'] = f"Q{match.group(1)}"
            result['fiscal_year'] = f"FY{match.group(2)}"
            result['raw'] = f"Q{match.group(1)} FY{match.group(2)}"
            result['doc_type'] = '10-Q' if result['quarter'] != 'Q4' else '10-K'
            return result

        # Pattern 2: FY#### alone (annual report)
        match = re.search(r'\bfy\s*(\d{4})\b', query_lower)
        if match:
            result['fiscal_year'] = f"FY{match.group(1)}"
            result['raw'] = f"FY{match.group(1)}"
            result['doc_type'] = '10-K'
            return result

        # Pattern 3: "fiscal year 2024" or "fiscal 2024"
        match = re.search(r'fiscal\s*(?:year\s*)?(\d{4})', query_lower)
        if match:
            result['fiscal_year'] = f"FY{match.group(1)}"
            result['raw'] = f"FY{match.group(1)}"
            result['doc_type'] = '10-K'
            return result

        # Pattern 4: Written quarters ("first quarter 2024", "Q1 2024")
        for word, quarter in self.QUARTER_WORD_MAP.items():
            if word in query_lower:
                result['quarter'] = quarter
                # Look for year nearby
                year_match = re.search(r'(\d{4})', query)
                if year_match:
                    result['fiscal_year'] = f"FY{year_match.group(1)}"
                    result['raw'] = f"{quarter} FY{year_match.group(1)}"
                result['doc_type'] = '10-Q'
                return result

        # Pattern 5: Just Q# without year
        match = re.search(r'\bq([1-4])\b', query_lower)
        if match:
            result['quarter'] = f"Q{match.group(1)}"
            result['doc_type'] = '10-Q'
            # No fiscal year specified

        return result

    def to_metadata_filter(self, extracted: Dict) -> Optional[Dict]:
        """
        Convert extracted fiscal period to ChromaDB metadata filter.

        Args:
            extracted: Output from extract()

        Returns:
            ChromaDB where clause dictionary, or None if no filter needed
        """
        filters = []

        # Exact period match (highest precision)
        if extracted.get('raw'):
            # For "Q1 FY2025", we want fiscal_period to contain "Q1 FY2025"
            return {"fiscal_period": {"$eq": extracted['raw']}}

        # Fiscal year only match
        if extracted.get('fiscal_year'):
            return {"fiscal_period": {"$contains": extracted['fiscal_year']}}

        # No temporal filter
        return None

    def to_relaxed_filter(self, extracted: Dict) -> Optional[Dict]:
        """
        Create a relaxed filter that matches fiscal year but not necessarily quarter.
        Useful as fallback when exact match returns too few results.
        """
        if extracted.get('fiscal_year'):
            return {"fiscal_period": {"$contains": extracted['fiscal_year']}}
        return None

    def get_calendar_context(self, extracted: Dict) -> str:
        """
        Generate calendar context for query expansion.

        Returns human-readable date ranges for the fiscal period.
        """
        if not extracted.get('quarter') or not extracted.get('fiscal_year'):
            return ""

        quarter = extracted['quarter']
        fy_str = extracted['fiscal_year']
        fy_num = int(fy_str.replace('FY', ''))

        cal_info = self.MSFT_FISCAL_CALENDAR.get(quarter, {})
        months = cal_info.get('months', '')
        end_month = cal_info.get('end_month', '')

        # Calculate calendar year for quarter end
        # Q1 (Jul-Sep) and Q2 (Oct-Dec) are in calendar year = FY - 1
        # Q3 (Jan-Mar) and Q4 (Apr-Jun) are in calendar year = FY
        if quarter in ['Q1', 'Q2']:
            cal_year = fy_num - 1
        else:
            cal_year = fy_num

        # Build context string
        quarter_end_dates = {
            'Q1': f"September 30, {fy_num - 1}",
            'Q2': f"December 31, {fy_num - 1}",
            'Q3': f"March 31, {fy_num}",
            'Q4': f"June 30, {fy_num}",
        }

        end_date = quarter_end_dates.get(quarter, '')

        return f"{months} {cal_year}, quarter ended {end_date}"


class TemporalQueryExpander:
    """
    Expand queries with explicit temporal context to improve retrieval.

    This helps the retriever find relevant documents by adding:
    - Calendar month ranges
    - Quarter end dates
    - Document type hints
    """

    def __init__(self):
        self.extractor = FiscalPeriodExtractor()

    def expand(self, query: str) -> Tuple[str, Dict]:
        """
        Expand query with temporal context.

        Args:
            query: Original user query

        Returns:
            Tuple of (expanded_query, fiscal_info)
        """
        fiscal_info = self.extractor.extract(query)

        if not fiscal_info.get('raw'):
            return query, fiscal_info

        expansions = []

        # Add calendar context
        calendar_context = self.extractor.get_calendar_context(fiscal_info)
        if calendar_context:
            expansions.append(calendar_context)

        # Add document type hint
        if fiscal_info.get('doc_type'):
            if fiscal_info['doc_type'] == '10-Q':
                expansions.append("quarterly report 10-Q")
            else:
                expansions.append("annual report 10-K")

        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            return expanded, fiscal_info

        return query, fiscal_info


class QueryPreprocessor:
    """
    Main query preprocessing class that combines all preprocessing steps.
    """

    def __init__(self):
        self.fiscal_extractor = FiscalPeriodExtractor()
        self.query_expander = TemporalQueryExpander()

    def process(self, query: str) -> Dict:
        """
        Full query preprocessing pipeline.

        Returns:
            Dictionary with:
            - original_query: The original query
            - expanded_query: Query with temporal expansion
            - fiscal_info: Extracted fiscal period information
            - metadata_filter: ChromaDB filter for retrieval
            - relaxed_filter: Fallback filter if strict returns too few results
        """
        # Extract fiscal period
        fiscal_info = self.fiscal_extractor.extract(query)

        # Expand query
        expanded_query, _ = self.query_expander.expand(query)

        # Create metadata filters
        metadata_filter = self.fiscal_extractor.to_metadata_filter(fiscal_info)
        relaxed_filter = self.fiscal_extractor.to_relaxed_filter(fiscal_info)

        return {
            'original_query': query,
            'expanded_query': expanded_query,
            'fiscal_info': fiscal_info,
            'metadata_filter': metadata_filter,
            'relaxed_filter': relaxed_filter,
        }
