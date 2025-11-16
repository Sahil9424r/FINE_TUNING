# English-to-Hindi Translation with `google/mt5-small`

This project details the process of fine-tuning the `google/mt5-small` model for English-to-Hindi translation. It explores and compares three different fine-tuning methodologies:

1.  **Full Fine-Tuning:** Training all 1.2 billion parameters of the model (resource-intensive).
2.  **LoRA (Low-Rank Adaptation):** A Parameter-Efficient Fine-Tuning (PEFT) method that freezes the base model and trains only a small set of adapter weights.
3.  **QLoRA (Quantized LoRA):** A highly-efficient method that combines 4-bit quantization of the base model with LoRA, drastically reducing memory requirements.

The code is contained within the `FINE_TUNIN_BYME.ipynb` notebook.

## 📖 Table of Contents

* Key Concepts Explained
    * Model: `google/mt5-small`
    * Dataset: `cfilt/iitb-english-hindi`
    * Technique 1: Full Fine-Tuning
    * Technique 2: LoRA
    * Technique 3: QLoRA
* Technology Stack
* Setup & Installation
    * 1. Install Dependencies
    * 2. Hugging Face Authentication
* Data Preprocessing (Common Steps)
    * 1. Load Dataset
    * 2. Format for Seq2Seq
    * 3. Tokenize Data
* Experiment 1: Full Fine-Tuning
    * Code
    * Results
* Experiment 2: LoRA (PEFT)
    * Code
    * Trainable Parameters
* Experiment 3: QLoRA (Quantization + PEFT)
    * Code
    * Optimizer Note
* How to Train on a Custom CSV
* Inference: How to Use the Model

---

## 🔑 Key Concepts Explained

### Model: `google/mt5-small`
**mT5 (multilingual T5)** is a version of the T5 (Text-to-Text Transfer Transformer) model trained on a massive multilingual dataset (mC4). It's an encoder-decoder model, making it perfect for sequence-to-sequence tasks like translation. The `small` version has ~1.2 billion parameters.

### Dataset: `cfilt/iitb-english-hindi`
This is a parallel corpus containing **1.6 million** English-Hindi sentence pairs, ideal for training a translation model.

### Technique 1: Full Fine-Tuning
This is the traditional method where **all** parameters in the model are updated during training.
* **Pros:** Can achieve the highest potential accuracy.
* **Cons:** Extremely resource-intensive. Requires a very large amount of VRAM (GPU memory) and takes a long time to train. This notebook's attempt was stopped due to this limitation.

### Technique 2: LoRA
**LoRA (Low-Rank Adaptation)** is a Parameter-Efficient Fine-Tuning (PEFT) technique.
* **How it works:** It freezes the entire pre-trained model and injects small, trainable "adapter" layers (matrices) into specific components (like the query `q` and value `v` attention modules).
* **Pros:**
    * Trains **<1%** of the total parameters.
    * Drastically reduces VRAM requirements.
    * Much faster training.
    * The original model weights are untouched, and the trained adapters are small, making it easy to have multiple "versions" (e.g., different tasks) of a single base model.

### Technique 3: QLoRA
**QLoRA (Quantized LoRA)** is an optimization of LoRA.
* **How it works:** It combines LoRA with **4-bit quantization**.
    1.  **Quantization:** The large, frozen base model is loaded into GPU memory in 4-bit precision (using `bitsandbytes`) instead of the usual 32-bit or 16-bit. This cuts the memory for the base model by 75-88%.
    2.  **LoRA:** The small LoRA adapters are then trained on top of this quantized model (in 16-bit or 32-bit for stability).
* **Pros:** The most memory-efficient method. Allows fine-tuning large models on consumer-grade GPUs.

---

## 🛠️ Technology Stack

* `transformers`: For loading models and tokenizers.
* `datasets`: For loading and processing the `iitb-english-hindi` dataset.
* `peft`: For implementing LoRA and QLoRA.
* `bitsandbytes`: For 4-bit model quantization.
* `trl`: For the `SFTTrainer`, a simplified trainer class.
* `accelerate`: For optimizing PyTorch training on any hardware.

---

## 🚀 Setup & Installation

### 1. Install Dependencies
Run this cell in your notebook to install all required libraries.

```python
!pip install transformers peft datasets accelerate bitsandbytes trl
