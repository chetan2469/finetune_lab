import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import streamlit as st
import logging

# Configure page settings
st.set_page_config(
    page_title="Fine-Tune Teaching Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def init_session_state():
    """Initialize all session state variables used across the app."""
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'lora_config' not in st.session_state:
        st.session_state.lora_config = {'r': 8, 'alpha': 16, 'dropout': 0.05}
    if 'training_hyperparameters' not in st.session_state:
        st.session_state.training_hyperparameters = {'lr': 2e-4, 'batch_size': 4, 'epochs': 3, 'weight_decay': 0.01}

def main():
    init_session_state()
    
    st.sidebar.title("🎓 Fine-Tune Lab")
    st.sidebar.markdown("Learn how to fine-tune Large Language Models!")
    
    # Navigation
    st.sidebar.header("Navigation")
    pages = {
        "📊 Dashboard": "dashboard",
        "🧠 Architecture": "architecture",
        "📝 1. Dataset Prep": "dataset",
        "🧩 2. LoRA Setup": "lora",
        "🚀 3. Training": "training",
        "💬 4. Evaluation": "evaluation"
    }
    
    selected_page_name = st.sidebar.radio("Go to step:", list(pages.keys()))
    selected_page = pages[selected_page_name]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Legend:**
    - 📝 Data formatting
    - 🧩 Adapter weights
    - 🚀 Model training
    - 💬 Chat interface
    """)
    
    # Route to the appropriate page
    if selected_page == "dashboard":
        from ui.dashboard import render_dashboard
        render_dashboard()
    elif selected_page == "architecture":
        from ui.architecture import render_architecture
        render_architecture()
    elif selected_page == "dataset":
        from ui.dataset import render_dataset
        render_dataset()
    elif selected_page == "lora":
        from ui.lora import render_lora
        render_lora()
    elif selected_page == "training":
        from ui.training import render_training
        render_training()
    elif selected_page == "evaluation":
        from ui.evaluation import render_evaluation
        render_evaluation()

if __name__ == "__main__":
    main()
