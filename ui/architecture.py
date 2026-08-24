import streamlit as st
from ui.components import render_page_header

def render_architecture():
    render_page_header(
        "Architecture & Theory", 
        "Understand the mechanics of Large Language Models and Fine-Tuning."
    )
    
    st.markdown("### 1. The Language Modeling Objective")
    st.info("Large Language Models (LLMs) like LLaMA, Gemma, and Qwen are at their core just **next-token predictors**.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        During **Pre-training**, the model reads terabytes of text from the internet. 
        It learns to predict the next word in a sequence:
        
        `The quick brown fox jumps over the lazy [dog]`
        
        This massive pre-training phase gives the model a broad understanding of language, facts, reasoning, and even coding!
        However, a pre-trained model doesn't know it's supposed to be an "assistant". If you prompt it with a question, it might just write more questions.
        """)
    with col2:
        st.markdown("""
        **Pre-training Costs**:
        - Millions of dollars
        - Thousands of GPUs
        - Months of time
        """)
        
    st.markdown("---")
    
    st.markdown("### 2. Supervised Fine-Tuning (SFT)")
    st.markdown("""
    To make a model act like a helpful assistant, we perform **Supervised Fine-Tuning**.
    We provide the model with high-quality pairs of `(Instruction, Response)`.
    
    The model is still just predicting the next token, but now the context always follows a specific pattern (like ChatML).
    It learns: *"Ah, when I see `<|user|> question <|assistant|>`, I should output a helpful answer, not just continue the text."*
    """)
    
    st.markdown("---")
    
    st.markdown("### 3. Parameter-Efficient Fine-Tuning: LoRA")
    st.warning("Full fine-tuning updates all billions of parameters in the model. This requires massive VRAM. Enter **LoRA (Low-Rank Adaptation)**.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        **How LoRA works:**
        1. **Freeze** the massive pre-trained weights (W0). No gradients are calculated for them.
        2. **Inject** two tiny matrices, **A** and **B**, into the layers.
        3. During training, we only update A and B.
        4. During inference, the output is `(W0 * input) + (A * B * input)`.
        
        Because A and B are very small (determined by the **Rank 'r'**), we reduce trainable parameters by 99%, allowing us to train on a single consumer GPU or Mac!
        """)
    with col4:
        st.code("""
# Mathematical representation of LoRA
h = W0 * x + ∆W * x

# We decompose the large ∆W into two small matrices
∆W = B * A 

# Where:
# W0 is (d, d)  -> e.g. 4096 x 4096 (16M parameters)
# B  is (d, r)  -> e.g. 4096 x 8    (32K parameters)
# A  is (r, d)  -> e.g. 8 x 4096    (32K parameters)

# Total trainable: 64K instead of 16M!
        """, language="python")

    st.markdown("---")
    st.success("Now that you understand the theory, head over to the **Dataset Prep** page to start formatting your data for SFT!")
