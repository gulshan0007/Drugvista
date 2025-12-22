"""
DRUGVISTA - Embeddings Module
=============================
Generates embeddings for documents and queries using sentence-transformers.

AWS Alignment:
- In production, this would use Amazon Bedrock Embeddings (e.g., amazon.titan-embed-text-v1)
- Or Amazon SageMaker hosting a custom embedding model
- Currently uses sentence-transformers for local execution
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Sentence transformers for local embeddings
# In production: Use boto3 to call Amazon Bedrock
from sentence_transformers import SentenceTransformer

# Model selection - using a lightweight, effective model
# AWS equivalent: amazon.titan-embed-text-v1 or cohere.embed-english-v3
MODEL_NAME = "all-MiniLM-L6-v2"

# Global model instance (singleton pattern for efficiency)
_model = None


def get_model() -> SentenceTransformer:
    """
    Get or initialize the embedding model.
    Uses singleton pattern to avoid reloading model.
    
    AWS Production:
        Replace with Bedrock client initialization:
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    return _model


def embed_text(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the embedding vector
        
    AWS Production:
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps({'inputText': text})
        )
        return json.loads(response['body'].read())['embedding']
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts (batch processing).
    
    Args:
        texts: List of input texts
        
    Returns:
        List of embedding vectors
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def parse_document(file_path: Path) -> Dict[str, Any]:
    """
    Parse a document file with YAML-style metadata header.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with 'metadata' and 'content' keys
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML-style header between --- markers
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata_str = parts[1].strip()
            text_content = parts[2].strip()
            
            # Parse simple YAML metadata
            metadata = {}
            for line in metadata_str.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            
            return {
                'metadata': metadata,
                'content': text_content,
                'file_path': str(file_path)
            }
    
    # Fallback: no metadata header
    return {
        'metadata': {'id': file_path.stem},
        'content': content,
        'file_path': str(file_path)
    }


def load_all_documents(data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load all documents from the data directory.
    
    AWS Production:
        - Documents would be stored in S3
        - Use boto3 to list and download from S3 bucket
        
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='drugvista-documents')
        for obj in response['Contents']:
            doc = s3.get_object(Bucket='drugvista-documents', Key=obj['Key'])
            content = doc['Body'].read().decode('utf-8')
    """
    # Get the data directory path relative to this file
    base_path = Path(__file__).parent.parent / data_dir
    
    documents = []
    
    # Scan all subdirectories
    for subdir in ['papers', 'clinical_trials', 'market']:
        dir_path = base_path / subdir
        if dir_path.exists():
            for file_path in dir_path.glob('*.txt'):
                doc = parse_document(file_path)
                doc['metadata']['source_type'] = subdir
                documents.append(doc)
    
    print(f"Loaded {len(documents)} documents from {data_dir}/")
    return documents


def embed_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for all documents.
    
    Args:
        documents: List of parsed documents
        
    Returns:
        Documents with 'embedding' field added
    """
    print("Generating embeddings for documents...")
    
    # Extract text content for embedding
    texts = [doc['content'] for doc in documents]
    
    # Generate embeddings in batch
    embeddings = embed_texts(texts)
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc['embedding'] = embedding
    
    print(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
    return documents


if __name__ == "__main__":
    # Test embedding generation
    print("Testing DRUGVISTA Embeddings Module")
    print("=" * 50)
    
    # Test single embedding
    test_text = "Semaglutide reduces cardiovascular risk in diabetic patients"
    embedding = embed_text(test_text)
    print(f"\nTest embedding dimension: {len(embedding)}")
    
    # Load and embed all documents
    docs = load_all_documents()
    embedded_docs = embed_documents(docs)
    
    print(f"\nSuccessfully embedded {len(embedded_docs)} documents")
    for doc in embedded_docs[:3]:
        print(f"  - {doc['metadata'].get('id', 'unknown')}: {doc['metadata'].get('topic', 'N/A')}")

