"""Text preprocessing utilities for Jira issues."""
import re
from typing import Dict, Any, Optional
from html import unescape


def clean_html(text: str) -> str:
    """Remove HTML tags and entities from text.
    
    Args:
        text: Text potentially containing HTML
        
    Returns:
        Cleaned text with HTML removed
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = unescape(text)
    return text


def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text.
    
    Args:
        text: Text containing markdown
        
    Returns:
        Cleaned text without markdown formatting
    """
    if not text:
        return ""
    
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove markdown bold/italic **text** or *text*
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    # Remove markdown code blocks ```code```
    text = re.sub(r'```[^`]+```', ' ', text)
    # Remove inline code `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_text_from_jira_issue(issue_data: Dict[str, Any]) -> str:
    """Extract and combine text from Jira issue for embedding.
    
    Combines summary, description, and optionally comments into
    a single text string suitable for generating embeddings.
    
    Args:
        issue_data: Dictionary containing Jira issue data
        
    Returns:
        Combined and cleaned text from the issue
    """
    fields = issue_data.get('fields', {})
    
    # Extract summary
    summary = fields.get('summary', '')
    
    # Extract description
    description = fields.get('description', '')
    if isinstance(description, dict):
        # Handle Atlassian Document Format (ADF)
        description = extract_text_from_adf(description)
    
    # Combine summary and description
    combined_text = f"{summary}. {description}"
    
    # Clean the text
    combined_text = clean_html(combined_text)
    combined_text = clean_markdown(combined_text)
    combined_text = normalize_whitespace(combined_text)
    
    return combined_text


def extract_text_from_adf(adf_content: Dict[str, Any]) -> str:
    """Extract plain text from Atlassian Document Format (ADF).
    
    Args:
        adf_content: ADF content dictionary
        
    Returns:
        Extracted plain text
    """
    if not isinstance(adf_content, dict):
        return str(adf_content)
    
    text_parts = []
    
    def traverse(node: Dict[str, Any]):
        """Recursively traverse ADF nodes to extract text."""
        if isinstance(node, dict):
            # Extract text from text nodes
            if node.get('type') == 'text':
                text_parts.append(node.get('text', ''))
            
            # Traverse content
            if 'content' in node:
                for child in node['content']:
                    traverse(child)
            
            # Traverse other nested structures
            for key, value in node.items():
                if key != 'content' and isinstance(value, (dict, list)):
                    traverse(value)
        
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(adf_content)
    return ' '.join(text_parts)


def preprocess_issue_for_embedding(issue_data: Dict[str, Any]) -> str:
    """Main preprocessing function for Jira issues.
    
    This is the primary function to use for preparing issue text
    before generating embeddings.
    
    Args:
        issue_data: Complete Jira issue data
        
    Returns:
        Preprocessed text ready for embedding
    """
    return extract_text_from_jira_issue(issue_data)
