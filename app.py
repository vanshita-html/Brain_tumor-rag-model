import os
import streamlit as st
from PIL import Image

# Import helper modules (unmodified logic)
from model_utils import load_tumor_model, preprocess_image, predict_tumor
from rag_utils import ask_rag_assistant, load_vector_store

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Restrained, Editorial Custom CSS
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
        color: #111827;
    }
    
    .stApp {
        background-color: #FAFAFA;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1140px;
    }
    
    /* Sidebar Restyling - Plain white background, no drop shadows */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    /* Clear Chat History Button - Solid accent fill, white text, 4-6px corner radius, full width */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        width: 100% !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #1D4ED8 !important;
    }
    
    /* Secondary / Prompt buttons */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
        border: 1px solid #E5E7EB;
        background-color: #FFFFFF;
        color: #374151;
        box-shadow: none !important;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
        background-color: #EFF6FF;
    }
    
    /* Plain text Tabs with accent underline */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #E5E7EB !important;
        gap: 24px !important;
        padding: 0 !important;
        margin-bottom: 24px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: #6B7280 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 8px 4px 12px 4px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    /* Hide default chat message container styles if any */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #6B7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Expander styling - flat thin border */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #374151 !important;
        box-shadow: none !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #E5E7EB !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        background-color: #FFFFFF !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #6B7280;
        font-size: 13px;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Title Block (Direct on page background, split 2-color heading)
# -----------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="font-size: 50px; font-weight: 800; color: #111827; margin: 0 0 4px 0; line-height: 1.1; letter-spacing: -0.02em;">
        Brain Tumor <span style="color: #2563EB;">Assistant</span>
    </h1>
    <p style="font-size: 16px; color: #6B7280; margin: 0;">
        AI-Powered MRI Slice Classification & Knowledge-Grounded Patient Assistant
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Initialize Model & Vector Store
# -----------------------------------------------------------------------------
with st.spinner("Initializing system..."):
    try:
        model = load_tumor_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
        
    try:
        vector_store = load_vector_store()
    except Exception as e:
        st.warning(f"Vector store note: {e}")

# -----------------------------------------------------------------------------
# Application Tabs
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Tumor Prediction", "Ask the Assistant"])

# =============================================================================
# TAB 1: TUMOR PREDICTION PANEL
# =============================================================================
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.markdown('<h2 style="font-size: 21px; font-weight: 700; color: #111827; margin-bottom: 12px;">Upload Brain MRI Scan</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 14px; color: #6B7280; margin-bottom: 16px;">Select an axial/coronal brain MRI slice image file (JPG, JPEG, or PNG).</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                img_preview = Image.open(uploaded_file)
                st.markdown('<div style="border: 1px solid #E5E7EB; border-radius: 8px; background: #FFFFFF; padding: 12px; margin-top: 12px;">', unsafe_allow_html=True)
                st.image(
                    img_preview,
                    caption=f"Uploaded Image: {uploaded_file.name}",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to open uploaded image file: {e}")
        else:
            st.markdown("""
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; background: #FFFFFF; padding: 24px; text-align: center; color: #6B7280; font-size: 14px;">
                Drag and drop or browse an MRI scan to perform CNN tumor prediction.
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<h2 style="font-size: 21px; font-weight: 700; color: #111827; margin-bottom: 12px;">Inference & Analysis</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None and model_loaded:
            with st.spinner("Running model prediction..."):
                try:
                    pil_img, img_tensor = preprocess_image(uploaded_file)
                    prediction = predict_tumor(model, img_tensor)
                    
                    label = prediction['label']
                    conf = prediction['confidence']
                    desc = prediction['description']
                    
                    # Restyled Result Box: Plain 1px-bordered box (no colored icon badges)
                    st.markdown(f"""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                        <div style="font-size: 12px; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">PREDICTED CLASS</div>
                        <div style="font-size: 22px; font-weight: 800; color: #111827;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Metric (flat, no extra progress-bar graphics)
                    st.metric(
                        label="Confidence Score",
                        value=f"{conf:.2f}%"
                    )
                    
                    # Short Description Text at 14-15px gray
                    st.markdown(f"""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 18px; font-size: 14px; line-height: 1.6; color: #4B5563; margin-top: 16px;">
                        <strong style="color: #111827;">Overview:</strong> {desc}
                    </div>
                    """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error during model inference: {e}")
        elif uploaded_file is None:
            st.markdown("""
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; background: #FFFFFF; padding: 24px; text-align: center; color: #6B7280; font-size: 14px;">
                Awaiting image upload to compute classification.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Classification model is not available.")

# =============================================================================
# TAB 2: RAG CHATBOT PANEL
# =============================================================================
with tab2:
    st.markdown('<h2 style="font-size: 21px; font-weight: 700; color: #111827; margin-bottom: 8px;">Ask the Assistant</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 14px; color: #6B7280; margin-bottom: 20px;">Ask questions regarding brain tumor symptoms, diagnosis, treatment options, and WHO classification.</p>', unsafe_allow_html=True)
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am your brain tumor information assistant. "
                    "I can answer questions based on reference medical guides "
                    "(WHO CNS Classification, NCI Patient Guides, and clinical diagnostic literature). "
                    "How can I help you today?"
                ),
                "sources": []
            }
        ]
        
    # Quick Prompt Buttons
    chip_cols = st.columns(3)
    quick_query = None
    if chip_cols[0].button("Brain tumor symptoms?"):
        quick_query = "What are common symptoms of a brain tumor?"
    if chip_cols[1].button("WHO grading system?"):
        quick_query = "How does the WHO grade brain tumors?"
    if chip_cols[2].button("Meningioma treatments?"):
        quick_query = "What treatment options exist for Meningiomas?"
        
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    # Sidebar Content (Plain white, restyled headers, full-width accent button)
    with st.sidebar:
        st.markdown('<h3 style="font-size: 16px; font-weight: 700; color: #111827; margin-top: 8px; margin-bottom: 12px;">Chat Options</h3>', unsafe_allow_html=True)
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. Feel free to ask a new question.",
                    "sources": []
                }
            ]
            st.rerun()
            
        st.markdown('<hr style="border: none; border-top: 1px solid #E5E7EB; margin: 20px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 10px;">Reference Knowledge Base</h3>', unsafe_allow_html=True)
        st.markdown("""
        <ul style="padding-left: 18px; margin: 0 0 16px 0; font-size: 14px; color: #374151; line-height: 1.6;">
            <li>WHO CNS Classification (2021)</li>
            <li>NCI Adult CNS Tumors Treatment Guide</li>
            <li>Early Diagnosis from MRI Reports (IJDDT)</li>
        </ul>
        """, unsafe_allow_html=True)
        
        api_status = "Connected" if os.getenv("GEMINI_API_KEY") else "Passage Fallback Mode"
        st.markdown(f'<p style="font-size: 13px; color: #6B7280; margin: 0;"><strong>Gemini Status:</strong> {api_status}</p>', unsafe_allow_html=True)

    # Display Chat History with Custom HTML Blocks (No default emojis/avatars/bubbles)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="font-size: 11px; font-weight: 700; color: #6B7280; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">USER QUERY</div>
                <div style="background-color: #111827; color: #FFFFFF; font-weight: 600; padding: 14px 18px; border-radius: 6px; font-size: 14px; line-height: 1.5;">
                    {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-bottom: 16px;">
                <div style="font-size: 11px; font-weight: 700; color: #2563EB; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">ASSISTANT (GROUNDED ANSWER)</div>
                <div style="background-color: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 0 6px 6px 0; padding: 16px 20px; color: #111827; font-size: 14px; line-height: 1.6;">
                    {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Plain flat Source Citations expander
            if msg.get("sources"):
                with st.expander("Sources & References"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{src['doc_name']}** *(Page {src['page']})*")
                        st.markdown(f"<div style='color: #6B7280; font-size: 13px; margin-bottom: 8px;'><em>\"{src['text']}\"</em></div>", unsafe_allow_html=True)

    # Chat input box
    user_input = st.chat_input("Ask a question about brain tumors...")
    prompt_to_process = quick_query if quick_query else user_input
    
    if prompt_to_process:
        # Append user prompt
        st.session_state.messages.append({
            "role": "user",
            "content": prompt_to_process,
            "sources": []
        })
        
        # Display User Query block directly
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 11px; font-weight: 700; color: #6B7280; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">USER QUERY</div>
            <div style="background-color: #111827; color: #FFFFFF; font-weight: 600; padding: 14px 18px; border-radius: 6px; font-size: 14px; line-height: 1.5;">
                {prompt_to_process}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate Assistant Response (unmodified logic)
        with st.spinner("Retrieving reference context..."):
            rag_result = ask_rag_assistant(
                query=prompt_to_process,
                chat_history=st.session_state.messages[:-1]
            )
            
            answer_text = rag_result['answer']
            sources = rag_result['sources']
            
            # Display Assistant Answer block directly
            st.markdown(f"""
            <div style="margin-bottom: 16px;">
                <div style="font-size: 11px; font-weight: 700; color: #2563EB; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">ASSISTANT (GROUNDED ANSWER)</div>
                <div style="background-color: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 0 6px 6px 0; padding: 16px 20px; color: #111827; font-size: 14px; line-height: 1.6;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if sources:
                with st.expander("Sources & References"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**{src['doc_name']}** *(Page {src['page']})*")
                        st.markdown(f"<div style='color: #6B7280; font-size: 13px; margin-bottom: 8px;'><em>\"{src['text']}\"</em></div>", unsafe_allow_html=True)
                        
        # Save Assistant Response in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources
        })
        st.rerun()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-text">
    Brain Tumor Assistant &bull; Clinical Knowledge-Grounded Internal Dashboard
</div>
""", unsafe_allow_html=True)