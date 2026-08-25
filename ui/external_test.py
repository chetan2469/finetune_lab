import streamlit as st
import os
import glob
from ui.components import render_teaching_explanation, render_page_header

@st.cache_resource
def load_gguf_model(model_path: str):
    from llama_cpp import Llama
    # Initialize Llama model
    # n_ctx is context window
    # n_gpu_layers=-1 attempts to offload everything to GPU (Metal on Mac)
    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False
    )

def render_external_test():
    render_page_header(
        "Test External GGUF Model", 
        "Load and chat with a locally downloaded .gguf model."
    )
    
    render_teaching_explanation(
        title="Testing External Models",
        what="We are using llama.cpp to load highly optimized, quantized (.gguf) models on your local machine.",
        why="Quantized models use less RAM and run much faster, making them ideal for running locally. You can download fine-tuned models from platforms like Hugging Face (e.g. models ending in .gguf) and drop them into the models/ folder to test them here.",
        how="We scan the models/ directory for .gguf files, load them into memory using llama-cpp-python, and pass your chat messages to generate responses."
    )
    
    st.markdown("---")
    
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    model_basenames = [os.path.basename(f) for f in gguf_files]
    
    model_source = st.radio("Model Source:", ["From models/ directory", "Custom absolute path"], horizontal=True)
    
    if model_source == "From models/ directory":
        if not gguf_files:
            st.warning(f"⚠️ No `.gguf` files found in the `{models_dir}` directory. Please download a model and place it in that folder, or select 'Custom absolute path'.")
            return
        selected_model_name = st.selectbox("Select a local model:", model_basenames)
        selected_model_path = os.path.join(models_dir, selected_model_name)
    else:
        selected_model_path = st.text_input("Enter absolute path to .gguf file:", placeholder="/Users/username/Downloads/model.gguf")
        selected_model_name = os.path.basename(selected_model_path) if selected_model_path else "custom_model"
        
        if selected_model_path:
            if not os.path.exists(selected_model_path):
                st.error("File not found at the specified path.")
                return
            if not selected_model_path.endswith('.gguf'):
                st.warning("Path doesn't end with .gguf. It might not work correctly.")
        else:
            return
    
    if st.button("Load Model"):
        with st.spinner(f"Loading {selected_model_name} into memory..."):
            # Load it into cache
            load_gguf_model(selected_model_path)
            st.session_state.external_model_loaded = selected_model_name
        st.success(f"Model {selected_model_name} successfully loaded into memory!")
        
    if "external_chat_history" not in st.session_state:
        st.session_state.external_chat_history = []
        
    with st.expander("⚙️ Chat Settings & Memory", expanded=False):
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.external_chat_history = []
            st.rerun()
            
    st.subheader("Chat Interface")
    
    # Display chat history
    for message in st.session_state.external_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask the model something..."):
        if not st.session_state.get("external_model_loaded") == selected_model_name:
            st.error("Please click 'Load Model' first!")
            return
            
        st.session_state.external_chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("*(Generating...)*")
            
            try:
                model = load_gguf_model(selected_model_path)
                
                # Format for typical instruction models
                formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
                
                output = model(
                    formatted_prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stop=["### Instruction:", "User:"],
                    echo=False
                )
                
                response_text = output['choices'][0]['text'].strip()
                
                if not response_text:
                    response_text = "(Empty response)"
                    
                response_placeholder.markdown(response_text)
                st.session_state.external_chat_history.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                response_placeholder.error(f"Error generating response: {str(e)}")
