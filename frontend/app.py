"""
DRUGVISTA - Streamlit Frontend
===============================
Simple, elegant UI for pharmaceutical intelligence analysis.

Features:
- Single text input for queries
- Real-time analysis with loading states
- Structured results display
- Responsive design

AWS Alignment:
- In production, could be hosted on:
  - AWS Amplify for static hosting
  - ECS Fargate for containerized deployment
  - Or CloudFront + S3 for pure static assets (if using React)
"""

import os
import requests
import streamlit as st
from typing import Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

# Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="DRUGVISTA | AI Pharma Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f23 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #00d4ff, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 0;
    }
    
    .tagline {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .card-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #00d4ff;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    
    /* Status badges */
    .badge-high { color: #00ff88; }
    .badge-medium { color: #ffbb00; }
    .badge-low { color: #ff4444; }
    .badge-strong { color: #00ff88; }
    .badge-moderate { color: #ffbb00; }
    .badge-weak { color: #ff4444; }
    
    /* Risk tags */
    .risk-tag {
        display: inline-block;
        background: rgba(255, 68, 68, 0.2);
        border: 1px solid rgba(255, 68, 68, 0.4);
        color: #ff6b6b;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    
    /* Evidence tags */
    .evidence-tag {
        display: inline-block;
        background: rgba(0, 212, 255, 0.2);
        border: 1px solid rgba(0, 212, 255, 0.4);
        color: #00d4ff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1));
        border: 2px solid rgba(0, 255, 136, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .recommendation-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #888;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Confidence meter */
    .confidence-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Explanation box */
    .explanation-box {
        background: rgba(255, 255, 255, 0.02);
        border-left: 4px solid #00d4ff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #ccc;
        line-height: 1.8;
    }
    
    /* Input styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
        color: #000 !important;
        font-weight: 700 !important;
        padding: 0.8rem 3rem !important;
        border-radius: 30px !important;
        border: none !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_backend_health() -> Dict[str, Any]:
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def analyze_query(query: str, use_chain_of_thought: bool = True) -> Dict[str, Any]:
    """Send query to backend for analysis."""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={
                "query": query,
                "use_chain_of_thought": use_chain_of_thought
            },
            timeout=120  # LLM calls can take time
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json().get("detail", "Unknown error")}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_badge_class(value: str) -> str:
    """Get CSS class for status badges."""
    value_lower = value.lower()
    if value_lower in ['high', 'strong', 'proceed']:
        return 'badge-high'
    elif value_lower in ['medium', 'moderate', 'investigate further']:
        return 'badge-medium'
    else:
        return 'badge-low'


# =============================================================================
# MAIN UI
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">DRUGVISTA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">AI co-pilot for molecular, clinical, and market intelligence</p>', unsafe_allow_html=True)
    
    # Separator
    st.markdown("---")
    
    # Input section
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        query = st.text_area(
            "Enter a disease name, molecule name, or paste a research abstract:",
            height=120,
            placeholder="Example: Semaglutide for cardiovascular protection in diabetic patients\n\nOr paste an abstract from a research paper..."
        )
        
        # Analysis options
        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            use_cot = st.checkbox("Use detailed analysis (slower)", value=True)
        
        # Analyze button
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔬 ANALYZE", use_container_width=True)
    
    # Analysis results
    if analyze_clicked and query:
        with st.spinner("🧬 Analyzing... This may take 30-60 seconds for detailed analysis."):
            result = analyze_query(query, use_chain_of_thought=use_cot)
        
        if result["success"]:
            data = result["data"]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Recommendation box (prominent)
            rec_class = get_badge_class(data['recommendation'])
            st.markdown(f"""
            <div class="recommendation-box">
                <div class="recommendation-label">Recommendation</div>
                <div class="recommendation-value {rec_class}">{data['recommendation'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Main metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                viability_class = get_badge_class(data['clinical_viability'])
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Clinical Viability</div>
                    <div class="card-value {viability_class}">{data['clinical_viability']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                market_class = get_badge_class(data['market_signal'])
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Market Signal</div>
                    <div class="card-value {market_class}">{data['market_signal']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                confidence = data['confidence_score']
                confidence_pct = int(confidence * 100)
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Confidence</div>
                    <div class="card-value">{confidence_pct}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                risk_count = len(data.get('major_risks', []))
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Risk Factors</div>
                    <div class="card-value">{risk_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Explanation
            st.markdown(f"""
            <div class="explanation-box">
                <strong>Analysis Summary:</strong><br><br>
                {data['explanation']}
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors and evidence
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="result-card">
                    <div class="card-header">⚠️ Risk Flags</div>
                </div>
                """, unsafe_allow_html=True)
                
                risks = data.get('major_risks', [])
                if risks:
                    risk_html = " ".join([f'<span class="risk-tag">{risk}</span>' for risk in risks])
                    st.markdown(risk_html, unsafe_allow_html=True)
                else:
                    st.markdown("*No major risks identified*")
            
            with col2:
                st.markdown("""
                <div class="result-card">
                    <div class="card-header">📚 Key Evidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                evidence = data.get('key_evidence', [])
                if evidence:
                    evidence_html = " ".join([f'<span class="evidence-tag">{doc}</span>' for doc in evidence])
                    st.markdown(evidence_html, unsafe_allow_html=True)
                else:
                    st.markdown("*No specific documents cited*")
            
            # Raw JSON (collapsible)
            with st.expander("📋 View Raw JSON Response"):
                st.json(data)
        
        else:
            st.error(f"❌ Analysis failed: {result['error']}")
            st.info("Make sure the backend is running and OPENAI_API_KEY is set.")
    
    elif analyze_clicked:
        st.warning("Please enter a query to analyze.")
    
    # Footer with status
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Backend status check
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        health = check_backend_health()
        if health.get("status") == "healthy":
            docs = health.get("documents_indexed", 0)
            llm = "✓" if health.get("llm_configured") else "✗"
            st.success(f"✓ Backend connected | {docs} documents indexed | LLM: {llm}")
        else:
            st.error("✗ Backend not available. Start with: cd backend && python main.py")
        
        st.markdown("""
        <div style="text-align: center; color: #555; font-size: 0.8rem; margin-top: 1rem;">
            Built for AWS ImpactX Challenge @ IIT Bombay<br>
            Powered by RAG + Multi-step LLM Reasoning
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

