# LLM-Finetuning


# Llama-3.2-3B Reasoning Fine-Tune (CoT)

This repository demonstrates a parameter-efficient fine-tuning (PEFT) pipeline for the Llama-3.2-3B-Instruct model. The goal was to enhance the model's reasoning capabilities using Chain-of-Thought (CoT) data while maintaining a low memory footprint suitable for consumer GPUs (Tesla T4 / RTX 3060).

## 🔧 Tech Stack
- **Frameworks:** PyTorch, Unsloth, Hugging Face Transformers, TRL (Transformer Reinforcement Learning).
- **Technique:** QLoRA (Quantized Low-Rank Adaptation).
- **Hardware:** Trained on NVIDIA Tesla T4 (Google Colab).
- **Dataset:** [ServiceNow-AI/R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT).

## 🚀 Key Features
1. **4-bit Quantization:** Used `load_in_4bit=True` to fit the 3B model into limited VRAM.
2. **LoRA Adapters:** Targeted specific modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.) with Rank 16 and Alpha 16.
3. **Data Formatting:** Custom formatting applied to injection of "Thought" traces into the training prompt to mimic DeepSeek-R1 style reasoning.
4. **GGUF Export:** The final model is fused and converted to GGUF format for CPU/Edge deployment via `llama.cpp`.

## 📊 Training Configuration

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Llama-3.2-3B-Instruct | Optimized by Unsloth |
| **LoRA Rank (r)** | 16 | Controls the number of trainable parameters |
| **LoRA Alpha** | 16 | Scaling factor for LoRA weights |
| **Batch Size** | 2 | Per device |
| **Gradient Accumulation** | 4 | Effective batch size = 8 |
| **Learning Rate** | 2e-4 | Linear scheduler |
| **Optimizer** | AdamW (8-bit) | Memory optimized optimizer |

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install unsloth
   pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
