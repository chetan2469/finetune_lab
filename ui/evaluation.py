import streamlit as st
import time
import torch
import gc
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from ui.components import render_teaching_explanation, render_page_header

@st.cache_resource
def load_inference_model(model_id: str, adapter_path: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.bfloat16 if torch.backends.mps.is_available() or torch.cuda.is_bfloat16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map="auto"
    )
    
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
    return model, tokenizer

def generate_response(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.5, # Slightly higher temperature
            repetition_penalty=1.1, # Less aggressive penalty to prevent destroying grammar
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response

def render_evaluation():
    render_page_header(
        "Side-by-Side Evaluation Chat", 
        "Ask a question and see how the Base Model compares to your Fine-Tuned Model simultaneously."
    )
    
    with st.expander("⚙️ Chat Settings & Memory Management", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🧹 Clear Memory (RAM/VRAM)", use_container_width=True):
                st.cache_resource.clear()
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.success("Memory cleared!")

    # Out of the expander block to avoid Streamlit API Exception
    render_teaching_explanation(
        title="Evaluating Fine-Tunes Side-by-Side",
        what="We are generating responses from both the untampered Base model and the Fine-Tuned model sequentially.",
        why="To truly understand if your fine-tuning was successful, you need an A/B test. If the fine-tuned model outputs gibberish while the base model answers correctly, your training hyperparameters (like Learning Rate) were likely too aggressive, causing 'Catastrophic Overfitting'.",
        how="We load the base model, generate an answer, then apply the LoRA adapter, generate the second answer, and then clean up to save RAM."
    )

    adapter_exists = os.path.exists("./results/final_adapter")
    has_trained = st.session_state.get('model_trained', False) or adapter_exists
    
    if not has_trained:
        st.warning("⚠️ You haven't run the training step yet! The right side will remain empty until you train a model.")

    st.markdown("---")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Two columns for the chat histories
    chat_col1, chat_col2 = st.columns(2)
    
    with chat_col1:
        st.subheader("Base Model")
        for message in st.session_state.chat_history:
            if message["role"] == "user" or message["role"] == "base_assistant":
                display_role = "assistant" if message["role"] == "base_assistant" else "user"
                with st.chat_message(display_role):
                    st.markdown(message["content"])
                    
    with chat_col2:
        st.subheader("Fine-Tuned Model")
        for message in st.session_state.chat_history:
            if message["role"] == "user" or message["role"] == "tuned_assistant":
                display_role = "assistant" if message["role"] == "tuned_assistant" else "user"
                with st.chat_message(display_role):
                    st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask the model something..."):
        # Add user prompt to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with chat_col1:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                base_placeholder = st.empty()
                base_placeholder.markdown("*(Generating...)*")
                
        with chat_col2:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                tuned_placeholder = st.empty()
                if has_trained:
                    tuned_placeholder.markdown("*(Waiting...)*")
                else:
                    tuned_placeholder.markdown("*(No fine-tuned model available)*")
                    
        model_id = st.session_state.get("model_id", "Qwen/Qwen2.5-0.5B")
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # 1. GENERATE BASE MODEL
        try:
            model, tokenizer = load_inference_model(model_id, adapter_path=None)
            base_response = generate_response(formatted_prompt, model, tokenizer)
            
            if "### Instruction:" in base_response:
                base_response = base_response.split("### Instruction:")[0].strip()
            if not base_response.strip():
                base_response = "(Empty response)"
                
            base_placeholder.markdown(base_response)
            st.session_state.chat_history.append({"role": "base_assistant", "content": base_response})
            
            # Clear base model instance to free memory before loading adapter
            del model
            st.cache_resource.clear()
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
        except Exception as e:
            base_placeholder.error(f"Error: {str(e)}")

        # 2. GENERATE FINE-TUNED MODEL
        if has_trained:
            tuned_placeholder.markdown("*(Generating...)*")
            try:
                # Reload model but with the adapter this time
                model, tokenizer = load_inference_model(model_id, adapter_path="./results/final_adapter")
                tuned_response = generate_response(formatted_prompt, model, tokenizer)
                
                if "### Instruction:" in tuned_response:
                    tuned_response = tuned_response.split("### Instruction:")[0].strip()
                if not tuned_response.strip():
                    tuned_response = "(Empty response)"
                    
                # Highlight if it looks like catastrophic overfitting (gibberish/punctuation loops)
                if len(set(tuned_response)) < 5 and len(tuned_response) > 10:
                    tuned_response = f"⚠️ **Catastrophic Overfitting Detected:**\n\n{tuned_response}\n\n_Tip: Your Learning Rate was likely too high. Go back to Training and lower it!_"
                
                tuned_placeholder.markdown(tuned_response)
                st.session_state.chat_history.append({"role": "tuned_assistant", "content": tuned_response})
                
            except Exception as e:
                tuned_placeholder.error(f"Error: {str(e)}")
