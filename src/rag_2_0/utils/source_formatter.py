"""
Source formatting utilities for clean, informative citations.
"""

from typing import List, Dict, Any
import re
from datetime import datetime


class SourceFormatter:
    """Format document sources into clean, informative citations."""
    
    def __init__(self):
        self.journal_patterns = {
            'JSM': 'Journal of Sport Management',
            'JIS': 'Journal of Issues in Intercollegiate Athletics', 
            'JIAA': 'Journal of Issues in Intercollegiate Athletics',
            'JIIA': 'Journal of Issues in Intercollegiate Athletics',
            'SMR': 'Sport Management Review',
            'JASM': 'Journal of Applied Sport Management',
            'IJSM': 'International Journal of Sport Management',
            'IJHMS': 'International Journal of Human Movement Science',
            'RSJ': 'Recreational Sports Journal',
            'JCD': 'Journal of Career Development',
            'JSSAE': 'Journal of Student Services Administration and Evaluation',
            'EM': 'Event Management',
            'GI': 'Gender Issues',
            'JSFD': 'Journal of Systemics, Cybernetics and Informatics'
        }

    def extract_publication_info(self, title: str) -> Dict[str, str]:
        """Extract publication information from document title."""
        info = {
            'year': '',
            'journal': '',
            'authors': '',
            'clean_title': title
        }
        
        # Extract year (4 digits)
        year_match = re.search(r'20\d{2}', title)
        if year_match:
            info['year'] = year_match.group()
        
        # Extract journal abbreviation and expand it
        for abbrev, full_name in self.journal_patterns.items():
            if f'.{abbrev}.' in title or title.startswith(f'{abbrev}.') or f'{abbrev}.' in title:
                info['journal'] = full_name
                break
        
        # Extract authors (usually after year)
        if info['year']:
            parts = title.split(info['year'])
            if len(parts) > 1:
                author_part = parts[1].split('.')[0] if '.' in parts[1] else parts[1]
                # Clean up author names
                author_part = author_part.strip(' .')
                if author_part and len(author_part) < 50:  # Reasonable author name length
                    info['authors'] = author_part
        
        # Create clean title (remove file extensions, clean up)
        clean_title = title
        clean_title = re.sub(r'\.pdf$', '', clean_title, flags=re.IGNORECASE)
        clean_title = re.sub(r'^20\d{2}\.', '', clean_title)  # Remove leading year
        clean_title = re.sub(r'\.(JSM|JIS|JIAA|JIIA|SMR|JASM|IJSM|IJHMS|RSJ|JCD|JSSAE|EM|GI|JSFD)\.', '', clean_title)
        info['clean_title'] = clean_title.strip(' .')
        
        return info

    def format_document_citation(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Format a single document into a clean citation."""
        title = metadata.get('title', metadata.get('source_file', 'Unknown Document'))
        source_url = metadata.get('source', '')
        full_path = metadata.get('full_path', '')
        size = metadata.get('size', '')
        
        # Extract publication info
        pub_info = self.extract_publication_info(title)
        
        # Determine document type
        doc_type = 'Research Paper'
        if 'Trade.Journals' in full_path:
            doc_type = 'Trade Publication'
        elif any(j in title for j in self.journal_patterns.keys()):
            doc_type = 'Journal Article'
        elif '.pdf' in title.lower():
            doc_type = 'PDF Document'
        elif 'docs.google.com' in source_url:
            doc_type = 'Google Document'
        
        # Format size
        size_str = ''
        if size and size.isdigit():
            size_mb = round(int(size) / 1048576, 1) if int(size) > 1048576 else None
            if size_mb and size_mb > 0.1:
                size_str = f" ({size_mb}MB)"
        
        # Create formatted citation
        citation = {
            'display_title': pub_info['clean_title'] or title,
            'doc_type': doc_type,
            'journal': pub_info['journal'],
            'year': pub_info['year'],
            'authors': pub_info['authors'],
            'size_info': size_str,
            'url': source_url,
            'category': self._get_category(full_path),
            'raw_title': title
        }
        
        return citation

    def _get_category(self, full_path: str) -> str:
        """Extract category from document path."""
        if 'Articles' in full_path:
            return 'Research Articles'
        elif 'Trade.Journals' in full_path:
            return 'Trade Publications'
        else:
            return 'Documents'

    def format_sources_section(self, retrieved_docs_metadata: List[Dict[str, Any]]) -> str:
        """Format multiple sources into a clean, organized section."""
        if not retrieved_docs_metadata:
            return ""
        
        citations = []
        for doc_meta in retrieved_docs_metadata:
            citation = self.format_document_citation(doc_meta.get('metadata', {}))
            citations.append(citation)
        
        # Group by category
        categories = {}
        for citation in citations:
            category = citation['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(citation)
        
        # Build formatted output
        output_lines = ["\n\n📚 **Sources:**"]
        
        for category, docs in categories.items():
            if len(categories) > 1:  # Only show category headers if multiple categories
                output_lines.append(f"\n**{category}:**")
            
            for doc in docs:
                # Create a clean, informative citation
                citation_line = f"• **{doc['display_title']}**"
                
                # Add additional info
                details = []
                if doc['year']:
                    details.append(doc['year'])
                if doc['journal']:
                    details.append(doc['journal'])
                elif doc['doc_type'] != 'Research Paper':
                    details.append(doc['doc_type'])
                
                if details:
                    citation_line += f" ({', '.join(details)})"
                
                if doc['size_info']:
                    citation_line += doc['size_info']
                
                # Add link if available
                if doc['url']:
                    citation_line += f" [[View Document]({doc['url']})]"
                
                output_lines.append(citation_line)
        
        return "\n".join(output_lines)

    def format_sources_compact(self, retrieved_docs_metadata: List[Dict[str, Any]]) -> str:
        """Format sources in a more compact format for shorter responses."""
        if not retrieved_docs_metadata:
            return ""
        
        citations = []
        for doc_meta in retrieved_docs_metadata:
            citation = self.format_document_citation(doc_meta.get('metadata', {}))
            
            # Create compact citation
            compact = citation['display_title']
            if citation['year']:
                compact += f" ({citation['year']})"
            citations.append(compact)
        
        if len(citations) == 1:
            return f"\n\n📄 **Source:** {citations[0]}"
        else:
            return f"\n\n📚 **Sources:** {' • '.join(citations[:3])}" + (f" • +{len(citations)-3} more" if len(citations) > 3 else "")

    def format_inline_citations(self, retrieved_docs_metadata: List[Dict[str, Any]]) -> Dict[int, str]:
        """Create inline citation markers that can be embedded in text."""
        citations = {}
        for i, doc_meta in enumerate(retrieved_docs_metadata, 1):
            citation = self.format_document_citation(doc_meta.get('metadata', {}))
            short_cite = citation['display_title']
            if citation['year']:
                short_cite += f" ({citation['year']})"
            citations[i] = short_cite
        
        return citations 