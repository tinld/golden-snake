"""
Utility functions for text processing
"""
import re
from typing import List
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese text
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep Vietnamese diacritics
    text = re.sub(r'[^\w\s\u0100-\u017F]', '', text)
    
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for Vietnamese
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    return text.split()


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    Extract keywords from text (simple frequency-based)
    
    Args:
        text: Input text
        top_k: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    words = text.lower().split()
    
    # Simple stopwords for Vietnamese
    stopwords = {
        'là', 'cái', 'chiếc', 'những', 'các', 'có', 'được', 'được',
        'như', 'hoặc', 'và', 'mà', 'nếu', 'thì', 'để', 'tại', 'từ',
        'giữa', 'này', 'kia', 'cái', 'chiếc', 'tôi', 'bạn', 'anh',
        'chị', 'nó', 'nơi', 'lúc', 'khi', 'giờ', 'bao', 'bến'
    }
    
    # Filter stopwords and count
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    word_freq = {}
    for word in filtered:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top k
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_k]]


if __name__ == "__main__":
    text = "Làm sao sửa lỗi này? Tôi gặp vấn đề kỹ thuật lớn!"
    print(f"Original: {text}")
    print(f"Normalized: {normalize_text(text)}")
    print(f"Tokens: {tokenize(text)}")
    print(f"Keywords: {extract_keywords(text, top_k=3)}")
