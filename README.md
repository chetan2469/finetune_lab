# LLM Fine-Tuning Teaching Lab

Welcome to the **LLM Fine-Tuning Teaching Lab**, an interactive educational platform designed for students at Chedo Tech to learn how to practically fine-tune Large Language Models (LLMs) locally!

Built with **Streamlit**, **Hugging Face Transformers**, and **PEFT (LoRA)**, this application takes you step-by-step through the entire lifecycle of training an AI model.

## Features

1. **Architecture & Theory**: Understand the underlying math of Neural Networks and why we use Low-Rank Adaptation (LoRA) instead of Full Fine-Tuning.
2. **Dataset Preparation**: Learn how to format raw JSON data into structural templates (like ChatML or Alpaca) with special EOS (End Of String) tokens so the model can learn how to stop talking.
3. **LoRA Configuration**: Experiment with Rank (`r`), Alpha (`α`), and Dropout to see mathematically how many parameters you save.
4. **Real Training Engine**: Train an actual local model (e.g. `Qwen2.5-0.5B`) natively on your machine using `SFTTrainer`. Features anti-overfitting mechanisms like distributed layer targeting and robust gradient clipping.
5. **A/B Side-by-Side Evaluation**: Chat with both the untouched Base Model and your newly Fine-Tuned model simultaneously to evaluate the exact impact of your training. Also includes an automated "Catastrophic Overfitting" detector.

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- (For Mac) Apple Silicon M1/M2/M3 is fully supported (using MPS fallback for unsupported ops).
- (For Windows/Linux) NVIDIA GPU recommended but can run on CPU/CUDA.

### 2. Installation

Clone this repository and install the dependencies:
```bash
git clone https://github.com/chetan2469/finetune_lab.git
cd finetune_lab
pip install -r requirements.txt
```

### 3. Running the Lab
Start the interactive UI:
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

---

## 📚 How to Use the Lab

### Phase 1: Format the Data
Go to the **Dataset Prep** tab. Upload the provided `chedo_tech_finetune_dataset.jsonl` (or use the sample datasets). Select the Alpaca template and click **Format Dataset**. This applies the critical structural tags and EOS tokens.

### Phase 2: Configure LoRA
Go to the **LoRA Setup** tab. Set **Rank (r)** and **Alpha** to `32`. This gives the model a larger "notebook" to store factual data. Note how the percentage of trainable parameters changes!

### Phase 3: Train the Model
Go to the **Training** tab. We recommend the following robust settings for a small 100-row dataset:
- **Model**: `Qwen/Qwen2.5-0.5B` (Lightweight and fast).
- **Learning Rate**: `1e-4` or `5e-5` (Safe step size).
- **Epochs**: `10` (Allows the model to read the data enough times to memorize it).
Click **Start Real Training**. Wait for the loss curve to finish charting!

### Phase 4: Evaluate
Go to the **Evaluation** tab. Ask the model a question like: *"What is Chedo Tech?"*
You will see the base model hallucinate or give a generic answer, while your fine-tuned model will give you the exact details you trained it on!

---

## 🛠️ Tech Stack
- **UI Framework**: Streamlit
- **ML Framework**: PyTorch
- **Transformers library**: Hugging Face `transformers`, `peft`, `trl`, `datasets`
- **Data handling**: Pandas

## 📝 License
This project is for educational purposes. Feel free to use and modify the lab for teaching students about AI!
