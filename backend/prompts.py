"""
DRUGVISTA - Prompts Module
===========================
Multi-step reasoning prompts for clinical, market, and decision synthesis.

The LLM reasoning pipeline follows 4 steps:
1. Context Understanding - Parse and categorize retrieved documents
2. Clinical Reasoning - Analyze efficacy, safety, and viability
3. Market Reasoning - Assess commercial potential and competitive landscape
4. Decision Synthesis - Combine insights into actionable recommendation

AWS Alignment:
- In production, prompts would be stored in AWS Systems Manager Parameter Store
- Or versioned in S3 for A/B testing different prompt strategies
- Bedrock would serve the LLM (Claude, Titan, or Llama)
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are DRUGVISTA, an expert AI co-pilot for pharmaceutical and biomedical intelligence.

Your role is to analyze molecules, diseases, and research abstracts to provide:
- Clinical viability assessments
- Safety and risk analysis
- Market and commercial insights
- Evidence-based recommendations

IMPORTANT RULES:
1. Only use information from the provided context documents
2. Cite specific evidence from documents when making claims
3. Be explicit about uncertainty when evidence is limited
4. Never hallucinate clinical data or statistics
5. Provide balanced analysis including both opportunities and risks

You serve pharmaceutical researchers, investors, and decision-makers who need accurate, actionable intelligence."""

# =============================================================================
# STEP 1: CONTEXT UNDERSTANDING
# =============================================================================

CONTEXT_UNDERSTANDING_PROMPT = """Analyze the following query and retrieved documents.

USER QUERY: {query}

RETRIEVED DOCUMENTS:
{documents}

---

Task: Provide a structured summary of the available evidence.

For each document, extract:
1. Document ID and type (paper/clinical_trial/market)
2. Key topic (disease, molecule, or therapeutic area)
3. Most relevant findings for the query
4. Limitations or caveats mentioned

Then synthesize:
- What aspects of the query are well-covered by the evidence?
- What aspects have limited or no evidence?
- Are there any contradictions between documents?

Format your response as a structured analysis that will feed into clinical and market reasoning."""

# =============================================================================
# STEP 2: CLINICAL REASONING
# =============================================================================

CLINICAL_REASONING_PROMPT = """Based on the context analysis, perform clinical reasoning.

QUERY: {query}

CONTEXT SUMMARY:
{context_summary}

RETRIEVED EVIDENCE:
{documents}

---

Analyze the clinical aspects:

1. EFFICACY ASSESSMENT
   - What clinical endpoints were studied?
   - What were the effect sizes (if available)?
   - How does this compare to existing treatments?

2. SAFETY PROFILE
   - What adverse events were reported?
   - What is the severity and frequency of key risks?
   - Are there specific populations at higher risk?

3. CLINICAL VIABILITY
   - Is the evidence from rigorous studies (RCT, Phase 3)?
   - What is the regulatory status?
   - What are the main barriers to clinical adoption?

4. RISK FLAGS
   - List specific safety concerns that warrant attention
   - Note any trial failures or regulatory warnings
   - Identify patient populations that may be contraindicated

Provide your clinical assessment with explicit citations to document IDs."""

# =============================================================================
# STEP 3: MARKET REASONING  
# =============================================================================

MARKET_REASONING_PROMPT = """Based on the context and clinical analysis, perform market reasoning.

QUERY: {query}

CLINICAL ASSESSMENT:
{clinical_assessment}

MARKET-RELATED EVIDENCE:
{market_documents}

---

Analyze the commercial and market aspects:

1. MARKET OPPORTUNITY
   - What is the estimated market size (if mentioned)?
   - What is the target patient population?
   - What is the current standard of care?

2. COMPETITIVE LANDSCAPE
   - Who are the key competitors?
   - What are the differentiation factors?
   - Are there upcoming competitive threats?

3. COMMERCIAL CHALLENGES
   - What are the pricing and reimbursement dynamics?
   - Are there manufacturing or supply chain issues?
   - What is the payer/access outlook?

4. MARKET SIGNAL ASSESSMENT
   - Strong: Clear market need, favorable dynamics, limited competition
   - Moderate: Good opportunity but significant challenges
   - Weak: Major barriers or unfavorable market conditions

Provide your market assessment with explicit citations to evidence."""

# =============================================================================
# STEP 4: DECISION SYNTHESIS
# =============================================================================

DECISION_SYNTHESIS_PROMPT = """Synthesize all analyses into a final recommendation.

QUERY: {query}

CLINICAL ASSESSMENT:
{clinical_assessment}

MARKET ASSESSMENT:
{market_assessment}

---

Provide a comprehensive decision synthesis:

1. SUMMARY OF KEY FINDINGS
   - 2-3 sentences capturing the most important insights

2. CLINICAL VIABILITY RATING
   - High: Strong efficacy, acceptable safety, clear regulatory path
   - Medium: Promising but significant uncertainties or risks
   - Low: Major efficacy, safety, or regulatory concerns

3. KEY EVIDENCE
   - List the document IDs that most strongly support your assessment

4. MAJOR RISKS
   - List 2-4 specific risk factors that decision-makers should consider

5. MARKET SIGNAL
   - Strong: Proceed with high priority
   - Moderate: Proceed with caution, monitor developments
   - Weak: Significant concerns, further investigation needed

6. RECOMMENDATION
   - Proceed: Evidence supports moving forward
   - Investigate Further: Promising but needs more data
   - Drop: Significant concerns outweigh potential

7. CONFIDENCE SCORE
   - 0.0-1.0 based on quantity and quality of available evidence
   - >0.8: High confidence, robust evidence
   - 0.5-0.8: Moderate confidence, some gaps
   - <0.5: Low confidence, limited evidence

8. HUMAN-READABLE EXPLANATION
   - 3-5 sentences for presentation to stakeholders

Format your response as valid JSON matching this structure:
{{
    "clinical_viability": "High | Medium | Low",
    "key_evidence": ["doc_id_1", "doc_id_2"],
    "major_risks": ["risk_1", "risk_2"],
    "market_signal": "Strong | Moderate | Weak",
    "recommendation": "Proceed | Investigate Further | Drop",
    "confidence_score": 0.0,
    "explanation": "Human readable summary..."
}}"""

# =============================================================================
# SINGLE-SHOT PROMPT (FALLBACK)
# =============================================================================

SINGLE_SHOT_PROMPT = """You are DRUGVISTA, an AI co-pilot for pharmaceutical intelligence.

Analyze the following query using the retrieved documents.

USER QUERY: {query}

RETRIEVED DOCUMENTS:
{documents}

---

Provide a comprehensive analysis covering:
1. Clinical viability (efficacy, safety, regulatory status)
2. Risk factors and safety concerns
3. Market dynamics and commercial potential
4. Final recommendation

Your response MUST be valid JSON in this exact format:
{{
    "clinical_viability": "High | Medium | Low",
    "key_evidence": ["doc_id_1", "doc_id_2"],
    "major_risks": ["risk_1", "risk_2"],
    "market_signal": "Strong | Moderate | Weak",
    "recommendation": "Proceed | Investigate Further | Drop",
    "confidence_score": 0.0,
    "explanation": "Human readable summary of 3-5 sentences..."
}}

Base your analysis ONLY on the provided documents. Cite document IDs as evidence.
If information is limited, reflect this in a lower confidence score."""


def format_documents_for_prompt(documents: list) -> str:
    """Format retrieved documents for inclusion in prompts."""
    formatted = []
    for i, doc in enumerate(documents, 1):
        doc_id = doc['metadata'].get('id', f'doc_{i}')
        doc_type = doc['metadata'].get('type', 'unknown')
        topic = doc['metadata'].get('topic', 'N/A')
        content = doc['content'][:1500]  # Truncate long documents
        
        formatted.append(f"""
--- Document {i} ---
ID: {doc_id}
Type: {doc_type}
Topic: {topic}
Content:
{content}
""")
    
    return "\n".join(formatted)


def get_prompt_chain():
    """Return the full prompt chain for multi-step reasoning."""
    return {
        'system': SYSTEM_PROMPT,
        'context_understanding': CONTEXT_UNDERSTANDING_PROMPT,
        'clinical_reasoning': CLINICAL_REASONING_PROMPT,
        'market_reasoning': MARKET_REASONING_PROMPT,
        'decision_synthesis': DECISION_SYNTHESIS_PROMPT
    }

