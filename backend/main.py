"""
DRUGVISTA - FastAPI Backend
============================
REST API for pharmaceutical intelligence analysis.

Endpoints:
- POST /analyze: Main analysis endpoint
- GET /health: Health check
- GET /documents: List indexed documents

AWS Alignment:
- In production, this would be deployed as:
  - AWS Lambda with API Gateway (serverless)
  - Or ECS Fargate for container deployment
  - Or EC2 behind ALB for traditional deployment
- API Gateway would handle authentication, rate limiting, CORS
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import get_vector_store, initialize_vector_store
from rag_pipeline import RAGPipeline, AnalysisResult


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint."""
    query: str = Field(
        ..., 
        description="Disease name, molecule name, or research abstract to analyze",
        min_length=3,
        max_length=5000
    )
    use_chain_of_thought: bool = Field(
        default=True,
        description="Use multi-step reasoning (slower but more thorough)"
    )


class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint."""
    clinical_viability: str = Field(
        ..., 
        description="High, Medium, or Low"
    )
    key_evidence: List[str] = Field(
        ..., 
        description="Document IDs supporting the analysis"
    )
    major_risks: List[str] = Field(
        ..., 
        description="Key risk factors identified"
    )
    market_signal: str = Field(
        ..., 
        description="Strong, Moderate, or Weak"
    )
    recommendation: str = Field(
        ..., 
        description="Proceed, Investigate Further, or Drop"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence in the analysis (0.0-1.0)"
    )
    explanation: str = Field(
        ..., 
        description="Human-readable explanation"
    )


class DocumentInfo(BaseModel):
    """Document metadata for listing."""
    id: str
    type: str
    topic: str
    source_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store_ready: bool
    documents_indexed: int
    llm_configured: bool


# =============================================================================
# APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes vector store on startup.
    
    AWS Production:
        - Load FAISS index from S3 on Lambda cold start
        - Initialize Bedrock client
        - Warm up embedding model
    """
    print("=" * 50)
    print("DRUGVISTA Backend Starting...")
    print("=" * 50)
    
    # Initialize vector store (will build if not exists)
    print("\nInitializing vector store...")
    try:
        store = initialize_vector_store()
        print(f"Vector store ready with {store.index.ntotal} documents")
    except Exception as e:
        print(f"Warning: Vector store initialization failed: {e}")
    
    # Check LLM configuration
    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key configured ✓")
    else:
        print("WARNING: OPENAI_API_KEY not set - LLM analysis will fail")
        print("Set with: export OPENAI_API_KEY='your-key'")
    
    print("\n" + "=" * 50)
    print("DRUGVISTA Ready for requests!")
    print("=" * 50 + "\n")
    
    yield
    
    # Cleanup on shutdown
    print("\nDRUGVISTA Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="DRUGVISTA API",
    description="AI co-pilot for molecular, clinical, and market intelligence",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
# AWS Production: Configure via API Gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    AWS Production:
        - Used by ALB/ECS health checks
        - CloudWatch synthetic canaries
    """
    try:
        store = get_vector_store()
        docs_count = store.index.ntotal if store.index else 0
        store_ready = docs_count > 0
    except Exception:
        store_ready = False
        docs_count = 0
    
    return HealthResponse(
        status="healthy",
        vector_store_ready=store_ready,
        documents_indexed=docs_count,
        llm_configured=bool(os.getenv("OPENAI_API_KEY"))
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main analysis endpoint.
    
    Accepts a query (disease, molecule, or abstract) and returns
    structured intelligence with clinical, risk, and market insights.
    
    AWS Production:
        - Lambda function with 30-second timeout
        - API Gateway with request validation
        - Results cached in ElastiCache
        - CloudWatch metrics for monitoring
    """
    # Validate LLM configuration
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="LLM not configured. Set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(use_chain_of_thought=request.use_chain_of_thought)
        
        # Run analysis
        result = pipeline.analyze(request.query)
        
        # Convert to response
        return AnalyzeResponse(
            clinical_viability=result.clinical_viability,
            key_evidence=result.key_evidence,
            major_risks=result.major_risks,
            market_signal=result.market_signal,
            recommendation=result.recommendation,
            confidence_score=result.confidence_score,
            explanation=result.explanation
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all indexed documents.
    
    Useful for debugging and understanding the knowledge base.
    """
    try:
        store = get_vector_store()
        
        documents = []
        for doc in store.documents:
            documents.append(DocumentInfo(
                id=doc['metadata'].get('id', 'unknown'),
                type=doc['metadata'].get('type', 'unknown'),
                topic=doc['metadata'].get('topic', 'N/A'),
                source_type=doc['metadata'].get('source_type', 'unknown')
            ))
        
        return documents
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.get("/search")
async def search_documents(query: str, top_k: int = 5):
    """
    Search for relevant documents without LLM analysis.
    
    Useful for testing retrieval quality.
    """
    try:
        store = get_vector_store()
        results = store.search(query, top_k=top_k)
        
        return {
            "query": query,
            "results": [
                {
                    "id": doc['metadata'].get('id', 'unknown'),
                    "type": doc['metadata'].get('type', 'unknown'),
                    "topic": doc['metadata'].get('topic', 'N/A'),
                    "score": doc['similarity_score'],
                    "preview": doc['content'][:300] + "..."
                }
                for doc in results
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print(f"\nStarting DRUGVISTA Backend on port {port}...")
    print(f"API docs available at: http://localhost:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Enable hot reload for development
    )

