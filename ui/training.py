import streamlit as st
import pandas as pd
from ui.components import render_teaching_explanation, render_page_header
import torch
import gc

def render_training():
    render_page_header(
        "Training Lab", 
        "Train the LoRA adapters on your formatted dataset using a real Hugging Face model."
    )
    
    st.markdown("""
    We will now take your formatted dataset and use it to update the LoRA adapter weights.
    The base model weights are frozen; only the LoRA matrices will be updated using backpropagation.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        model_id = st.text_input(
            "Hugging Face Model ID", 
            value=st.session_state.get("model_id", "Qwen/Qwen2.5-0.5B"),
            help="E.g., Qwen/Qwen2.5-0.5B or google/gemma-2b. Smaller models are strongly recommended for local training."
        )
        
        lr = st.selectbox(
            "Learning Rate", 
            options=[2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6],
            index=1, # Default to 1e-4 which is standard for r=32
            format_func=lambda x: f"{x:.0e}",
            help="Step size for weight updates. If your model outputs '!!!!!!!', change this to 1e-5 or 5e-5!"
        )
        
        epochs = st.slider(
            "Epochs", 
            min_value=1, max_value=30, value=st.session_state.training_hyperparameters.get('epochs', 10),
            help="How many times to pass over the entire dataset. For small datasets (100 rows), you need 10-15 epochs to accurately memorize facts!"
        )
        
        batch_size = st.selectbox(
            "Batch Size", 
            options=[1, 2, 4, 8], index=0,
            help="Number of examples processed before updating the model. Keep this low (1 or 2) to save memory!"
        )
        
        st.session_state.training_hyperparameters = {'lr': lr, 'epochs': epochs, 'batch_size': batch_size}
        st.session_state.model_id = model_id
        
        start_training = st.button("🚀 Start Real Training", type="primary")
    
    with col2:
        st.subheader("Training Progress")
        
        if start_training:
            if 'current_dataset' not in st.session_state or not st.session_state.current_dataset:
                st.error("⚠️ No dataset found! Please go to 'Dataset Prep' and format a dataset first.")
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
                from peft import LoraConfig, get_peft_model
                from trl import SFTTrainer
                from datasets import Dataset

                st.info("Loading model and tokenizer... This might take a minute.")
                
                # Setup UI elements to update during training
                progress_bar = st.progress(0)
                status_text = st.empty()
                chart_placeholder = st.empty()
                
                loss_data = pd.DataFrame(columns=["loss"])
                
                # Define custom Streamlit Callback
                class StreamlitCallback(TrainerCallback):
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if "loss" in logs:
                            step = state.global_step
                            loss = logs["loss"]
                            
                            # Append to dataframe and update chart
                            new_row = pd.DataFrame({"loss": [loss]}, index=[step])
                            nonlocal loss_data
                            loss_data = pd.concat([loss_data, new_row])
                            chart_placeholder.line_chart(loss_data, y="loss")
                            
                            # Update progress
                            if state.max_steps > 0:
                                progress = min(step / state.max_steps, 1.0)
                                progress_bar.progress(progress)
                                
                            status_text.text(f"Step {step}/{state.max_steps} | Loss: {loss:.4f}")

                try:
                    # 1. Load Tokenizer & Model
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        
                    # Using float32 for MPS (Mac) to prevent NaN gradient explosions (bfloat16/float16 bugs on Apple Silicon)
                    if torch.backends.mps.is_available():
                        dtype = torch.float32
                    elif torch.cuda.is_bfloat16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16
                    
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_id, 
                        torch_dtype=dtype,
                        device_map="auto"
                    )
                    
                    # 2. Apply LoRA
                    lora_cfg = st.session_state.lora_config
                    peft_config = LoraConfig(
                        r=lora_cfg.get('r', 8),
                        lora_alpha=lora_cfg.get('alpha', 16),
                        lora_dropout=lora_cfg.get('dropout', 0.05),
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers to distribute learning
                    )
                    
                    model = get_peft_model(base_model, peft_config)
                    trainable, total = model.get_nb_trainable_parameters()
                    status_text.text(f"LoRA adapters applied. Trainable parameters: {trainable:,} / {total:,}")
                    
                    # 3. Prepare Dataset
                    hf_dataset = Dataset.from_list(st.session_state.current_dataset)
                    
                    # Critical fix: Append the EOS token to teach the model to STOP generating at the end!
                    # Without this, the model hallucinates infinite exclamation marks or new instructions.
                    if tokenizer.eos_token:
                        def add_eos(example):
                            example["formatted_text"] = example["formatted_text"] + tokenizer.eos_token
                            return example
                        hf_dataset = hf_dataset.map(add_eos)
                        
                    # 4. Training Arguments
                    training_args = TrainingArguments(
                        output_dir="./results",
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=4,
                        learning_rate=lr,
                        logging_steps=1, # Log every step for the UI
                        num_train_epochs=epochs,
                        report_to="none", # Disable wandb/tensorboard
                        save_strategy="no", # Don't save checkpoints during run to save space
                        weight_decay=0.01, # Regularization to prevent weights from exploding
                        max_grad_norm=0.3, # Gradient clipping to stop catastrophic overfitting
                    )
                    
                    # 5. Trainer
                    trainer = SFTTrainer(
                        model=model,
                        train_dataset=hf_dataset,
                        peft_config=peft_config,
                        dataset_text_field="formatted_text",
                        max_seq_length=256,
                        tokenizer=tokenizer,
                        args=training_args,
                        callbacks=[StreamlitCallback()]
                    )
                    
                    status_text.text("Starting training loop...")
                    trainer.train()
                    
                    # 6. Save final adapter
                    trainer.model.save_pretrained("./results/final_adapter")
                    tokenizer.save_pretrained("./results/final_adapter")
                    
                    progress_bar.progress(1.0)
                    st.session_state.model_trained = True
                    st.success("🎉 Training Complete! Adapters saved to `./results/final_adapter`. Head to Evaluation.")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                finally:
                    # Cleanup memory
                    if 'model' in locals():
                        del model
                    if 'base_model' in locals():
                        del base_model
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()

        elif st.session_state.get('model_trained', False):
            st.success("Model has been trained! Head to the Evaluation tab.")
        else:
            st.info("Configure parameters and click Start Real Training. Warning: This will consume RAM/VRAM.")

    render_teaching_explanation(
        title="TrainerCallback (Streamlit Magic)",
        what="We use a custom `TrainerCallback` to pipe the logs back into Streamlit.",
        why="Normally, `trainer.train()` runs a blocking loop that prints to the console. By hooking into `on_log`, we can dynamically update the Streamlit UI chart at the exact moment a loss value is calculated.",
        how="We pass `callbacks=[StreamlitCallback()]` to the SFTTrainer initialization."
    )
