import streamlit as st
from ui.components import render_teaching_explanation, render_page_header

def calculate_trainable_params(r: int):
    # Dummy calculation for a hypothetical 7B model (e.g. Llama-2-7b)
    # Llama-2-7b has ~6.7B parameters.
    # The number of trainable parameters injected by LoRA scales roughly linearly with r
    base_params = 6_738_415_616
    trainable = r * 1_048_576 # Just a mock scaling factor for demonstration
    percent = (trainable / (base_params + trainable)) * 100
    return trainable, base_params + trainable, percent

def render_lora():
    render_page_header(
        "LoRA Setup (Low-Rank Adaptation)", 
        "Configure the adapter weights that we will inject into the model."
    )
    
    st.markdown("""
    Instead of updating all billions of parameters in a Large Language Model (Full Fine-Tuning), we use **LoRA**. 
    LoRA freezes the original model and injects small, trainable "adapter" modules into the layers.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("LoRA Hyperparameters")
        
        r = st.slider(
            "Rank (r)", 
            min_value=4, max_value=128, value=st.session_state.lora_config.get('r', 32), step=4,
            help="The 'rank' of the adapter matrices. Higher means more expressive power and accuracy, but more parameters to train."
        )
        
        alpha = st.slider(
            "Alpha (α)", 
            min_value=8, max_value=256, value=st.session_state.lora_config.get('alpha', 32), step=8,
            help="A scaling factor. Usually set to 1x or 2x the Rank (r)."
        )
        
        dropout = st.slider(
            "Dropout", 
            min_value=0.0, max_value=0.2, value=st.session_state.lora_config.get('dropout', 0.05), step=0.01,
            help="Probability of dropping out adapter neurons to prevent overfitting."
        )
        
        # Save to session state
        st.session_state.lora_config = {'r': r, 'alpha': alpha, 'dropout': dropout}
        
    with col2:
        st.subheader("Parameter Efficiency Calculator")
        st.markdown("For a hypothetical **7B parameter** model (like Llama-2):")
        
        trainable, total, percent = calculate_trainable_params(r)
        
        st.metric("Trainable Parameters", f"{trainable:,}")
        st.metric("Total Parameters", f"{total:,}")
        st.metric("% Trainable", f"{percent:.4f}%")
        
        st.success(f"By using LoRA with r={r}, you are only training **{percent:.4f}%** of the model! This is why it can run on a single consumer GPU.")

    st.subheader("Code Snippet (peft)")
    st.code(f"""
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r={r},
    lora_alpha={alpha},
    lora_dropout={dropout},
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to your loaded base model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
    """, language="python")

    render_teaching_explanation(
        title="Understanding Rank (r)",
        what="The Rank 'r' determines the size of the LoRA matrices.",
        why="Large neural network layers have a 'low intrinsic rank'—meaning they don't need all their dimensions to learn new things. We can represent the updates as the product of two much smaller matrices (A and B).",
        how="If a weight matrix is 4096x4096 (16M parameters), a LoRA update with r=8 replaces it with learning two matrices: 4096x8 and 8x4096 (total 65K parameters). That's a 99.6% reduction!"
    )
    
    st.info("👉 Next step: Let's train these parameters!")
