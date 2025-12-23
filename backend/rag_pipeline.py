"""
DRUGVISTA - RAG Pipeline Module
=================================
Implements the core RAG pipeline with multi-step reasoning.

Pipeline Flow:
1. Query embedding + vector search
2. Context understanding (LLM step 1)
3. Clinical reasoning (LLM step 2)
4. Market reasoning (LLM step 3)
5. Decision synthesis (LLM step 4)

AWS Alignment:
- In production, LLM calls would go to Amazon Bedrock
- Pipeline orchestration would use AWS Step Functions or Lambda
- Each step could be a separate Lambda function for scalability
"""

import os
import json
import re
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# OpenAI for LLM (hackathon-friendly, easy API key setup)
# AWS Production: Use boto3 bedrock-runtime client
import openai

from vector_store import get_vector_store, VectorStore
from prompts import (
    SYSTEM_PROMPT,
    CONTEXT_UNDERSTANDING_PROMPT,
    CLINICAL_REASONING_PROMPT,
    MARKET_REASONING_PROMPT,
    DECISION_SYNTHESIS_PROMPT,
    SINGLE_SHOT_PROMPT,
    format_documents_for_prompt
)


@dataclass
class AnalysisResult:
    """Structured result from the RAG pipeline."""
    clinical_viability: str
    key_evidence: List[str]
    major_risks: List[str]
    market_signal: str
    recommendation: str
    confidence_score: float
    explanation: str
    raw_reasoning: Optional[Dict[str, str]] = None


class RAGPipeline:
    """
    RAG Pipeline for pharmaceutical intelligence.
    
    AWS Production Architecture:
        - Step Functions state machine orchestrating Lambda functions
        - Each reasoning step as a separate Lambda
        - Bedrock for LLM inference
        - Results cached in ElastiCache for repeated queries
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        model: str = "gpt-3.5-turbo",  # Use gpt-4 for better reasoning if available
        use_chain_of_thought: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store instance (will create if None)
            model: OpenAI model to use
            use_chain_of_thought: If True, use multi-step reasoning
        """
        self.vector_store = vector_store or get_vector_store()
        self.model = model
        self.use_chain_of_thought = use_chain_of_thought
        
        # Initialize OpenAI client
        # AWS Production: bedrock = boto3.client('bedrock-runtime')
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set. LLM calls will fail.")
            self.client = None
        else:
            # Clear any proxy env vars that might interfere
            import httpx
            self.client = openai.OpenAI(
                api_key=api_key,
                http_client=httpx.Client()
            )
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM with given prompts.
        
        AWS Production (Bedrock with Claude):
            response = bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 2048,
                    'messages': [
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'system': system_prompt
                })
            )
            return json.loads(response['body'].read())['content'][0]['text']
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY environment variable.")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent outputs
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    # Minimum similarity score threshold - below this, docs are not relevant
    RELEVANCE_THRESHOLD = 0.25  # Lowered for better recall
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant documents for the query.
        
        AWS Production:
            - Could use OpenSearch k-NN for larger scale
            - Add re-ranking with cross-encoder model
            - Cache frequent queries in ElastiCache
        """
        # Get categorized results
        categorized = self.vector_store.search_by_category(query, top_k=2)
        
        # Flatten for formatting
        all_docs = (
            categorized['papers'] + 
            categorized['clinical_trials'] + 
            categorized['market']
        )
        
        # Filter documents by relevance threshold
        relevant_docs = [
            doc for doc in all_docs 
            if doc.get('similarity_score', 0) >= self.RELEVANCE_THRESHOLD
        ]
        
        # Calculate average relevance score
        avg_score = sum(d.get('similarity_score', 0) for d in all_docs) / len(all_docs) if all_docs else 0
        max_score = max((d.get('similarity_score', 0) for d in all_docs), default=0)
        
        # Determine if query is within scope of knowledge base
        # Relaxed condition: at least 1 relevant doc OR max score >= 0.3
        is_relevant = len(relevant_docs) >= 1 and max_score >= 0.3
        
        # Debug logging
        print(f"  [Retrieval] max_score={max_score:.3f}, relevant_docs={len(relevant_docs)}, is_relevant={is_relevant}")
        
        return {
            'all_documents': relevant_docs if is_relevant else all_docs,
            'categorized': categorized,
            'formatted': format_documents_for_prompt(relevant_docs) if is_relevant else format_documents_for_prompt(all_docs),
            'is_relevant': is_relevant,
            'avg_score': avg_score,
            'max_score': max_score,
            'relevant_count': len(relevant_docs)
        }
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return default structure
        return {
            "clinical_viability": "Medium",
            "key_evidence": [],
            "major_risks": ["Unable to parse LLM response"],
            "market_signal": "Moderate",
            "recommendation": "Investigate Further",
            "confidence_score": 0.3,
            "explanation": response[:500] if response else "Analysis failed."
        }
    
    def _no_data_response(self, query: str, context: Dict[str, Any]) -> AnalysisResult:
        """
        Return a response when query is outside the knowledge base scope.
        Prevents hallucination by being explicit about data limitations.
        """
        return AnalysisResult(
            clinical_viability="Unknown",
            key_evidence=[],
            major_risks=["Insufficient data in knowledge base"],
            market_signal="Unknown",
            recommendation="No Data Available",
            confidence_score=0.0,
            explanation=f"The query '{query}' does not match any documents in the DRUGVISTA knowledge base. "
                       f"The system currently contains information about: GLP-1 agonists (Semaglutide, Tirzepatide), "
                       f"CAR-T therapy (Tisagenlecleucel), PCSK9 inhibitors (Evolocumab), Alzheimer's treatments "
                       f"(Lecanemab, Donanemab), CRISPR gene therapy, PD-1 inhibitors (Pembrolizumab), "
                       f"JAK inhibitors (Tofacitinib), mRNA vaccines, SGLT2 inhibitors (Dapagliflozin), "
                       f"and Bispecific antibodies (Teclistamab). Please query one of these topics for accurate analysis.",
            raw_reasoning={"retrieval_scores": f"max={context.get('max_score', 0):.3f}, avg={context.get('avg_score', 0):.3f}"}
        )
    
    def analyze_single_shot(self, query: str) -> AnalysisResult:
        """
        Single-shot analysis (faster, less detailed).
        
        Use when:
        - Quick results needed
        - Simple queries
        - API rate limits are a concern
        """
        # Retrieve context
        context = self.retrieve_context(query)
        
        # Check if query is within knowledge base scope
        if not context.get('is_relevant', False):
            print(f"  [WARNING] Query outside knowledge base scope (max_score={context.get('max_score', 0):.3f})")
            return self._no_data_response(query, context)
        
        # Build prompt
        prompt = SINGLE_SHOT_PROMPT.format(
            query=query,
            documents=context['formatted']
        )
        
        # Call LLM
        response = self._call_llm(SYSTEM_PROMPT, prompt)
        
        # Parse response
        parsed = self._parse_json_response(response)
        
        return AnalysisResult(
            clinical_viability=parsed.get('clinical_viability', 'Medium'),
            key_evidence=parsed.get('key_evidence', []),
            major_risks=parsed.get('major_risks', []),
            market_signal=parsed.get('market_signal', 'Moderate'),
            recommendation=parsed.get('recommendation', 'Investigate Further'),
            confidence_score=parsed.get('confidence_score', 0.5),
            explanation=parsed.get('explanation', 'Analysis completed.')
        )
    
    def analyze_chain_of_thought(self, query: str) -> AnalysisResult:
        """
        Multi-step chain-of-thought analysis (thorough, detailed).
        
        AWS Production:
            - Each step could be a separate Lambda function
            - Step Functions would orchestrate the chain
            - Intermediate results stored in DynamoDB
        """
        reasoning_chain = {}
        
        # Step 0: Retrieve context
        context = self.retrieve_context(query)
        
        # Check if query is within knowledge base scope
        if not context.get('is_relevant', False):
            print(f"  [WARNING] Query outside knowledge base scope (max_score={context.get('max_score', 0):.3f})")
            return self._no_data_response(query, context)
        
        # Step 1: Context Understanding
        print("  Step 1/4: Understanding context...")
        context_prompt = CONTEXT_UNDERSTANDING_PROMPT.format(
            query=query,
            documents=context['formatted']
        )
        context_summary = self._call_llm(SYSTEM_PROMPT, context_prompt)
        reasoning_chain['context_understanding'] = context_summary
        
        # Step 2: Clinical Reasoning
        print("  Step 2/4: Clinical reasoning...")
        clinical_prompt = CLINICAL_REASONING_PROMPT.format(
            query=query,
            context_summary=context_summary,
            documents=context['formatted']
        )
        clinical_assessment = self._call_llm(SYSTEM_PROMPT, clinical_prompt)
        reasoning_chain['clinical_reasoning'] = clinical_assessment
        
        # Step 3: Market Reasoning
        print("  Step 3/4: Market reasoning...")
        market_docs = format_documents_for_prompt(context['categorized']['market'])
        market_prompt = MARKET_REASONING_PROMPT.format(
            query=query,
            clinical_assessment=clinical_assessment,
            market_documents=market_docs if market_docs.strip() else "No specific market documents retrieved."
        )
        market_assessment = self._call_llm(SYSTEM_PROMPT, market_prompt)
        reasoning_chain['market_reasoning'] = market_assessment
        
        # Step 4: Decision Synthesis
        print("  Step 4/4: Synthesizing decision...")
        synthesis_prompt = DECISION_SYNTHESIS_PROMPT.format(
            query=query,
            clinical_assessment=clinical_assessment,
            market_assessment=market_assessment
        )
        synthesis = self._call_llm(SYSTEM_PROMPT, synthesis_prompt)
        reasoning_chain['decision_synthesis'] = synthesis
        
        # Parse final result
        parsed = self._parse_json_response(synthesis)
        
        return AnalysisResult(
            clinical_viability=parsed.get('clinical_viability', 'Medium'),
            key_evidence=parsed.get('key_evidence', []),
            major_risks=parsed.get('major_risks', []),
            market_signal=parsed.get('market_signal', 'Moderate'),
            recommendation=parsed.get('recommendation', 'Investigate Further'),
            confidence_score=parsed.get('confidence_score', 0.5),
            explanation=parsed.get('explanation', 'Analysis completed with chain-of-thought reasoning.'),
            raw_reasoning=reasoning_chain
        )
    
    def analyze(self, query: str) -> AnalysisResult:
        """
        Main analysis entry point.
        
        Automatically selects single-shot or chain-of-thought based on configuration.
        """
        print(f"Analyzing query: '{query}'")
        
        if self.use_chain_of_thought:
            return self.analyze_chain_of_thought(query)
        else:
            return self.analyze_single_shot(query)
    
    def to_response_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert AnalysisResult to API response dictionary."""
        return {
            "clinical_viability": result.clinical_viability,
            "key_evidence": result.key_evidence,
            "major_risks": result.major_risks,
            "market_signal": result.market_signal,
            "recommendation": result.recommendation,
            "confidence_score": result.confidence_score,
            "explanation": result.explanation
        }


# Singleton pipeline instance
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or initialize the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


if __name__ == "__main__":
    # Test the pipeline
    print("Testing DRUGVISTA RAG Pipeline")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: Set OPENAI_API_KEY to test LLM functionality")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        print("\nTesting retrieval only...")
        
        pipeline = RAGPipeline(use_chain_of_thought=False)
        context = pipeline.retrieve_context("semaglutide diabetes cardiovascular")
        
        print(f"\nRetrieved {len(context['all_documents'])} documents:")
        for doc in context['all_documents']:
            print(f"  - {doc['metadata'].get('id')}: {doc['similarity_score']:.4f}")
    else:
        pipeline = RAGPipeline(use_chain_of_thought=True)
        
        result = pipeline.analyze("What is the clinical viability of semaglutide for diabetes and cardiovascular protection?")
        
        print("\n" + "=" * 50)
        print("ANALYSIS RESULT")
        print("=" * 50)
        print(json.dumps(pipeline.to_response_dict(result), indent=2))

