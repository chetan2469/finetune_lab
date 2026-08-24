import streamlit as st
import pandas as pd
from ui.components import render_teaching_explanation, render_page_header

def render_dataset():
    render_page_header(
        "Dataset Preparation", 
        "Format your raw data into a structure the model can learn from."
    )
    
    st.markdown("""
    To fine-tune a model for chat or instruction following, we need to show it examples of how to respond. 
    A common format is the **ChatML** or **Instruction** format, which clearly separates the user's prompt from the assistant's response.
    """)
    
    # Dataset Selection
    st.subheader("1. Load Dataset")
    dataset_source = st.radio("Choose a dataset source:", ["Sample: Python QA", "Sample: Support Tickets", "Upload Custom (JSONL)"], horizontal=True)
    
    # Mock datasets
    if dataset_source == "Sample: Python QA":
        data = [
            {"instruction": "How do I reverse a string in Python?", "response": "You can reverse a string using slicing: `my_string[::-1]`."},
            {"instruction": "What is a dictionary?", "response": "A dictionary is a built-in Python data structure that stores key-value pairs."},
            {"instruction": "How to append to a list?", "response": "Use the `.append()` method: `my_list.append(item)`."}
        ]
        df = pd.DataFrame(data)
    elif dataset_source == "Sample: Support Tickets":
        data = [
            {"instruction": "My laptop won't turn on.", "response": "Please check if it's plugged in and the battery is charged. Hold the power button for 10 seconds."},
            {"instruction": "How do I reset my password?", "response": "Click on 'Forgot Password' on the login screen and follow the email instructions."},
            {"instruction": "The screen is flickering.", "response": "Try updating your display drivers or booting in safe mode to isolate the issue."}
        ]
        df = pd.DataFrame(data)
    else:
        uploaded_file = st.file_uploader("Upload a JSONL file with 'instruction' and 'response' columns", type=['jsonl'])
        if uploaded_file is not None:
            df = pd.read_json(uploaded_file, lines=True)
        else:
            df = pd.DataFrame(columns=["instruction", "response"])
            st.info("Please upload a file to proceed.")
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        st.subheader("2. Apply Chat Template")
        st.markdown("We need to combine the instruction and response into a single text block with special tokens that the model understands.")
        
        template_style = st.selectbox("Select a Template Style:", ["ChatML (<|im_start|>, <|im_end|>)", "Alpaca (### Instruction:, ### Response:)"])
        
        if st.button("Format Dataset"):
            formatted_data = []
            for _, row in df.iterrows():
                if template_style == "ChatML (<|im_start|>, <|im_end|>)":
                    text = f"<|im_start|>user\n{row['instruction']}<|im_end|>\n<|im_start|>assistant\n{row['response']}<|im_end|>"
                else:
                    text = f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['response']}"
                formatted_data.append({"formatted_text": text})
            
            st.session_state.current_dataset = formatted_data
            st.success("Dataset formatted successfully!")
            
            st.markdown("**Preview of Formatted Data:**")
            st.code(formatted_data[0]["formatted_text"], language="text")
            
            st.info("👉 Next step: Configure LoRA adapters.")

    render_teaching_explanation(
        title="Why format the data?",
        what="We combine the input and expected output into a single string with special markers.",
        why="Language models only predict the 'next token'. By formatting it this way, we teach the model to see the user's prompt, understand the markers, and then start predicting the assistant's response.",
        how="We use standard templates like ChatML. It's critical that the template matches exactly what the base model was pre-trained or instruction-tuned on if we are doing further fine-tuning."
    )
