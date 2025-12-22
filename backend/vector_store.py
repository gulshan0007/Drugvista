"""
DRUGVISTA - Vector Store Module
================================
FAISS-based vector store for efficient similarity search.

AWS Alignment:
- In production, would use Amazon OpenSearch with vector search
- Or Amazon Kendra for enterprise search
- Or store FAISS index in S3 and load on Lambda cold start
- Currently uses local FAISS for demonstration
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss

from embeddings import embed_text, embed_documents, load_all_documents


class VectorStore:
    """
    FAISS-based vector store for document retrieval.
    
    AWS Production Architecture:
        - Index stored in S3: s3://drugvista-indices/faiss_index.bin
        - Metadata in DynamoDB: drugvista-document-metadata
        - Could migrate to OpenSearch Serverless for managed solution
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.index_path = Path(__file__).parent / "faiss_index.bin"
        self.metadata_path = Path(__file__).parent / "documents_metadata.json"
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from embedded documents.
        
        Args:
            documents: List of documents with 'embedding' field
            
        AWS Production:
            - Would batch upload embeddings to OpenSearch
            - Or use SageMaker for index building at scale
        """
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        # Extract embeddings and convert to numpy array
        embeddings = np.array([doc['embedding'] for doc in documents], dtype='float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product after normalization = Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # Store document metadata (without embeddings to save space)
        self.documents = []
        for doc in documents:
            doc_copy = {
                'metadata': doc['metadata'],
                'content': doc['content'],
                'file_path': doc.get('file_path', '')
            }
            self.documents.append(doc_copy)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save(self) -> None:
        """
        Save index and metadata to disk.
        
        AWS Production:
            s3 = boto3.client('s3')
            s3.upload_file(self.index_path, 'drugvista-indices', 'faiss_index.bin')
            s3.put_object(
                Bucket='drugvista-indices',
                Key='documents_metadata.json',
                Body=json.dumps(self.documents)
            )
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save document metadata as JSON
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2)
        
        print(f"Saved index to {self.index_path}")
        print(f"Saved metadata to {self.metadata_path}")
    
    def load(self) -> bool:
        """
        Load index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
            
        AWS Production:
            s3 = boto3.client('s3')
            s3.download_file('drugvista-indices', 'faiss_index.bin', '/tmp/faiss_index.bin')
            # Load index from /tmp for Lambda
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_type: Optional filter by document type (paper, clinical_trial, market)
            
        Returns:
            List of matching documents with similarity scores
            
        AWS Production:
            - Use OpenSearch k-NN query
            - Or Kendra Query API with filters
        """
        if self.index is None:
            raise ValueError("Index not initialized. Load or build index first.")
        
        # Generate query embedding
        query_embedding = np.array([embed_text(query)], dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        # Search with extra results if filtering
        search_k = top_k * 3 if filter_type else top_k
        
        # Perform similarity search
        scores, indices = self.index.search(query_embedding, min(search_k, len(self.documents)))
        
        # Gather results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
                
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(score)
            
            # Apply type filter if specified
            if filter_type:
                doc_type = doc['metadata'].get('type', '')
                if doc_type != filter_type:
                    continue
            
            results.append(doc)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_category(
        self, 
        query: str, 
        top_k: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search and return results grouped by category.
        
        Args:
            query: Search query text
            top_k: Number of results per category
            
        Returns:
            Dictionary with results grouped by type
        """
        all_results = self.search(query, top_k=top_k * 3)
        
        categorized = {
            'papers': [],
            'clinical_trials': [],
            'market': []
        }
        
        for doc in all_results:
            doc_type = doc['metadata'].get('type', '')
            category = {
                'paper': 'papers',
                'clinical_trial': 'clinical_trials',
                'market': 'market'
            }.get(doc_type, 'papers')
            
            if len(categorized[category]) < top_k:
                categorized[category].append(doc)
        
        return categorized


def initialize_vector_store(force_rebuild: bool = False) -> VectorStore:
    """
    Initialize the vector store, loading existing or building new.
    
    Args:
        force_rebuild: If True, rebuild even if existing index found
        
    Returns:
        Initialized VectorStore instance
    """
    store = VectorStore()
    
    if not force_rebuild and store.load():
        print("Loaded existing vector store")
        return store
    
    print("Building new vector store...")
    
    # Load all documents
    documents = load_all_documents()
    
    # Generate embeddings
    embedded_docs = embed_documents(documents)
    
    # Build and save index
    store.build_index(embedded_docs)
    store.save()
    
    return store


# Singleton instance for the application
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or initialize the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = initialize_vector_store()
    return _vector_store


if __name__ == "__main__":
    # Test vector store
    print("Testing DRUGVISTA Vector Store")
    print("=" * 50)
    
    # Force rebuild for testing
    store = initialize_vector_store(force_rebuild=True)
    
    # Test search
    test_queries = [
        "diabetes cardiovascular outcomes",
        "gene therapy for genetic diseases",
        "cancer immunotherapy market trends"
    ]
    
    for query in test_queries:
        print(f"\n\nQuery: '{query}'")
        print("-" * 40)
        
        results = store.search(query, top_k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. [{doc['metadata'].get('type', 'unknown')}] {doc['metadata'].get('id', 'N/A')}")
            print(f"   Topic: {doc['metadata'].get('topic', 'N/A')}")
            print(f"   Score: {doc['similarity_score']:.4f}")
            print(f"   Preview: {doc['content'][:100]}...")

