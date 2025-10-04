# AI Agent Instructions - Tech Challenge Fine-Tuning Project

## Project Overview

This is a machine learning project focused on fine-tuning the **Phi-4-mini-instruct** (3.8B parameters) foundation model using **LoRA (Low-Rank Adaptation)** on the AmazonTitles-1.3MM dataset. The goal is to generate detailed product descriptions from product titles in Amazon's e-commerce style.

**Key Result**: Given a product title, the model generates relevant product descriptions trained on 3K samples (2.7K training / 300 validation) from Amazon product data.

## Architecture & Tech Stack

### Core Components

- **Foundation Model**: microsoft/Phi-4-mini-instruct (3.8B parameters, 128K context window)
- **Fine-tuning Technique**: LoRA via PEFT - trains only 0.08% of parameters (3.1M/3.8B)
- **Dataset**: AmazonTitles-1.3MM - processed from 2.2M → 1.3M → 3K samples
- **Training Infrastructure**: NVIDIA RTX 2000 Ada (16GB GPU)
- **Training Duration**: 2 epochs, ~60 minutes, loss reduction 2.89→2.77

### Key Libraries

```python
torch>=2.0.0              # Deep learning framework
transformers>=4.30.0      # Hugging Face transformers
peft>=0.4.0               # Parameter-Efficient Fine-Tuning (LoRA)
accelerate>=0.20.0        # Training optimization
datasets>=2.12.0          # Dataset handling
pandas>=2.0.0             # Data manipulation
```

## Project Structure

```
tech_challenge_finetune.ipynb    # Main notebook (32+ cells, sequential workflow)
dataset/trn.json                 # Training data (title + content columns)
modelo_finetuned/                # Final LoRA adapters + tokenizer
checkpoints/                     # Training checkpoints (every 50 steps)
outputs/                         # All intermediate checkpoints
docs/                           # Implementation plan + challenge requirements
requirements.txt                 # Python dependencies
```

## Critical Workflows

### 1. Dataset Preparation Pipeline

The notebook implements a 3-stage data processing pipeline:

1. **Load Raw Data** (Cell #9): Read `dataset/trn.json` with `title` and `content` columns
2. **Clean Data** (Cell #11): Remove nulls, duplicates, filter by length (title>10 chars, content>20 chars)
3. **Format Prompts** (Cell #14): Convert to training format:
   ```python
   f"Descreva o produto com o seguinte título: {title}\n\nDescrição: {content}"
   ```
4. **Split Data** (Cell #15): 90/10 train/validation split with `SEED=42` for reproducibility

**Important**: Dataset reduction from 2.2M → 1.3M → 3K is intentional for training feasibility on 16GB GPU.

### 2. Model Loading Pattern

**Critical Setup** (Cells #17-18):

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Required: Phi-4 has no pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for training
    trust_remote_code=True
)

if torch.cuda.is_available():
    model = model.cuda()
```

### 3. LoRA Configuration

**Attention Module Targeting** (Cell #21):

```python
lora_config = LoraConfig(
    r=16,                  # LoRA rank
    lora_alpha=32,         # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Phi-4 attention layers
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model.enable_input_require_grads()  # Critical: Enable gradients before LoRA
model = get_peft_model(model, lora_config)
```

**Why these modules?** Phi-4 uses standard transformer attention with query, key, value, and output projections. Training only these reduces parameters from 3.8B to 3.1M.

### 4. Training Configuration

**Optimizations for 16GB GPU** (Cell #25):

```python
TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,     # Effective batch size = 8
    learning_rate=5e-5,
    fp16=True,                         # Mixed precision
    save_steps=50,                     # Frequent checkpointing
    eval_steps=50,
    dataloader_num_workers=0,          # Critical: Avoids fork issues
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)
```

**Memory Optimization Strategy**:

- FP16 mixed precision reduces memory usage
- Gradient accumulation achieves larger effective batch size without OOM
- `dataloader_num_workers=0` prevents multiprocessing issues in notebooks

### 5. Inference Pattern

**Generation Function** (Cell #18):

```python
def generate_response(model, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Project-Specific Conventions

### Naming and Structure

- **Constants**: UPPER_SNAKE_CASE for hyperparameters (`MODEL_NAME`, `BATCH_SIZE`, `LORA_R`)
- **Paths**: Relative paths from project root (`./modelo_finetuned`, `./outputs`)
- **Seed Management**: `SEED=42` used consistently across train/val split and sampling

### Prompt Engineering Format

The project uses a specific prompt template that must be maintained:

```python
# Training format
f"Descreva o produto com o seguinte título: {title}\n\nDescrição: {content}"

# Inference format
f"Descreva o produto com o seguinte título: {title}\n\nDescrição:"
```

**Why this format?** Trained on Portuguese instructions to match the academic requirement and dataset context.

### Checkpoint Management

- **Training checkpoints**: `outputs/checkpoint-{step}` (every 50 steps)
- **Best checkpoint**: Automatically selected based on validation loss
- **Final model**: `modelo_finetuned/` contains LoRA adapters only (not full model)

**Loading fine-tuned model**:

```python
# Adapters only - requires base model reference
model = AutoModelForCausalLM.from_pretrained("./modelo_finetuned")
```

## Common Pitfalls & Solutions

### GPU Memory Issues

**Problem**: CUDA out of memory during training
**Solution**:

- Reduce `BATCH_SIZE=1` or `gradient_accumulation_steps=8`
- Decrease `MAX_LENGTH=64` (currently 128)
- Use 8-bit quantization: `load_in_8bit=True`

### Tokenizer Warnings

**Problem**: "The tokenizer has no padding token" warning
**Solution**: Already handled in Cell #17:

```python
tokenizer.pad_token = tokenizer.eos_token
```

### Model Comparison Memory Limitations

**Design Decision**: The notebook (Cell #32) compares baseline vs fine-tuned sequentially, not simultaneously, due to 16GB GPU limit. Baseline results are saved in `base_results` DataFrame and compared against fine-tuned model later.

## Development Guidelines

### When Modifying Hyperparameters

1. **Always update constants at top** (Cell #6): `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, etc.
2. **Adjust MAX_LENGTH carefully**: Longer sequences = more memory, better context
3. **LoRA rank tradeoff**: Higher `LORA_R` = more parameters = better quality but slower training

### When Changing the Model

1. **Update target_modules**: Different models have different attention layer names
   - Phi-4: `["q_proj", "k_proj", "v_proj", "o_proj"]`
   - Llama: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
2. **Check tokenizer compatibility**: Some models need special token handling
3. **Adjust MAX_LENGTH**: Different models have different optimal context lengths

### When Extending the Dataset

1. **Maintain column names**: Code expects `title` and `content` columns
2. **Preserve prompt format**: Critical for consistent model behavior
3. **Monitor sample size**: Training time scales linearly with data size
4. **Consider stratified sampling**: Current implementation uses uniform random sampling

## Testing & Validation

### No Automated Tests

This project follows academic requirements and uses manual validation:

1. **Baseline comparison** (Cell #19): 3 random samples tested pre-training
2. **Post-training comparison** (Cell #32): Same samples tested with fine-tuned model
3. **Interactive demo** (Cell #36): Manual testing interface

### Evaluation Criteria (Qualitative)

- **Relevância**: Description matches product title
- **Estrutura**: Proper formatting and organization
- **Especificidade**: Technical details and features
- **Coerência**: Text fluency and consistency
- **Domínio**: Amazon e-commerce style adherence

## External Dependencies

### Dataset Download

**Required manual step**: Download `trn.json` from [Google Drive](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view) and place in `dataset/` directory.

**Why manual?** Large file (1.3M records) not suitable for git. Academic project requirement.

### Model Download

**Automatic on first run**: Hugging Face transformers downloads `microsoft/Phi-4-mini-instruct` (~7.6GB) on first execution. Cached in `~/.cache/huggingface/`.

## Academic Context

This project is **Tech Challenge Phase 3** for a Postgraduate AI program. Requirements:

- ✅ Fine-tune a foundation model (Phi-4-mini-instruct)
- ✅ Use AmazonTitles-1.3MM dataset
- ✅ Generate product descriptions from titles
- ✅ Demonstrate before/after comparison
- ✅ Interactive demo
- ✅ Video presentation (≤10 min)
- ✅ GitHub repository with code

**Deliverables**: YouTube video + PDF with links + GitHub repo

## Quick Reference

### Run Training (Full Pipeline)

```bash
# 1. Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset to dataset/trn.json (manual)

# 3. Open notebook and run cells 1-32 sequentially
jupyter notebook tech_challenge_finetune.ipynb
```

### Use Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./modelo_finetuned")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

prompt = "Descreva o produto com o seguinte título: Wireless Headphones\n\nDescrição:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Check Training Progress

```python
# In notebook after training
trainer.state.log_history  # View loss progression
trainer.state.best_model_checkpoint  # Get best checkpoint path
```

## Key Files to Reference

- **Main workflow**: `tech_challenge_finetune.ipynb` (cells 1-37)
- **Requirements**: `docs/tech-challenge.md` - original problem statement
- **Implementation plan**: `docs/plano-implementacao.md` - architectural decisions
- **Model config**: `modelo_finetuned/adapter_config.json` - LoRA parameters
- **Training state**: `checkpoints/checkpoint-*/trainer_state.json` - training metrics
