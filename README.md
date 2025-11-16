**# Fine-Tuning `google/mt5-small` for English-to-Hindi Translation

This project demonstrates various techniques for fine-tuning the `google/mt5-small` model for machine translation, specifically from English to Hindi.

It explores and compares three different fine-tuning strategies:
1.  **Full Fine-Tuning:** (Attempted) Training all model parameters.
2.  **LoRA (Low-Rank Adaptation):** A Parameter-Efficient Fine-Tuning (PEFT) method that trains only a small subset of new parameters.
3.  **QLoRA (Quantized LoRA):** An even more efficient method that combines 4-bit quantization with LoRA to drastically reduce memory usage.

## 🚀 Techniques & Technologies

* **Model:** `google/mt5-small` (a multilingual T5 model)
* **Dataset:** `cfilt/iitb-english-hindi` from the Hugging Face Hub
* **Libraries:**
    * `transformers` (for models, tokenizers, and pipelines)
    * `datasets` (for loading and processing data)
    * `peft` (for LoRA and QLoRA implementation)
    * `bitsandbytes` (for 4-bit quantization)
    * `trl` (for the `SFTTrainer`)

## 📋 Prerequisites

Before running the notebook, you'll need to install the required libraries and provide a Hugging Face API token.

### 1. Dependencies

You can install all necessary Python packages using the following command:

```bash
pip install transformers peft datasets accelerate bitsandbytes trl**
