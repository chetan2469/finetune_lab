import streamlit as st
from ui.components import render_teaching_explanation, render_page_header

def render_dashboard():
    render_page_header(
        "Fine-Tune Teaching Lab", 
        "Welcome! Learn how Large Language Models are customized through fine-tuning."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Fine-Tuning?
        While pre-training teaches a model general language and world knowledge, **fine-tuning** adapts the model for specific tasks, behaviors, or domains. 
        
        Instead of learning from scratch, we take a pre-trained model and slightly adjust its internal weights using a curated dataset of examples.
        """)
        
        st.info("👈 Use the sidebar to navigate through the interactive fine-tuning pipeline!")
        
        st.markdown("### The Fine-Tuning Pipeline")
        st.markdown("""
        1. **Dataset Prep**: Format examples into standard instruction/response templates.
        2. **LoRA Setup**: Inject small, trainable adapter weights into the frozen model.
        3. **Training**: Update the adapter weights by learning from the dataset.
        4. **Evaluation**: Chat with the model to see how its behavior changed.
        """)
        
    with col2:
        st.markdown("### How does it compare?")
        
        with st.expander("📝 Prompt Engineering", expanded=True):
            st.markdown("""
            **Effort**: Low
            **Impact**: Changes behavior per-request.
            **Cost**: Cheap
            _Best for simple formatting or context steering._
            """)
            
        with st.expander("📚 RAG (Retrieval-Augmented Generation)"):
            st.markdown("""
            **Effort**: Medium
            **Impact**: Injects external knowledge dynamically.
            **Cost**: Medium
            _Best for answering questions based on private or recent data._
            """)
            
        with st.expander("🧠 Fine-Tuning (What we are doing!)"):
            st.markdown("""
            **Effort**: High
            **Impact**: Permanently changes model style, tone, or specific skills.
            **Cost**: High (Training)
            _Best for teaching a model a new format (like JSON output) or a specific persona._
            """)

    render_teaching_explanation(
        title="Fine-Tuning Overview",
        what="Fine-tuning adjusts a small portion of an LLM's weights using a highly curated, specific dataset.",
        why="Pre-trained models are 'jacks of all trades'. Fine-tuning makes them specialists in a particular domain without losing their general knowledge.",
        how="We use techniques like LoRA (Low-Rank Adaptation) to train tiny 'adapter' modules instead of updating all the massive model weights, making training feasible on regular hardware."
    )
