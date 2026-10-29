"""
train_qlora.py - Baymax Phase 2, step 4: QLoRA fine-tune of Qwen2.5-1.5B-Instruct.

Reads:   train.jsonl, eval.jsonl     (from prepare_dataset.py)
Writes:  outputs/baymax-lora/        (the LoRA adapter, tens of MB, not a full model)

Run:     python train_qlora.py
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT_DIR = "outputs/baymax-lora-v7"
MAX_LEN = 512                                # lower to 384 if you hit out-of-memory

# bf16 needs an Ampere-or-newer GPU (RTX 30xx+); older cards fall back to fp16.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

# ---------------------------------------------------------------
# 1. The "Q" in QLoRA: load the frozen base model in 4-bit NF4.
#    Weights are stored in 4 bits and decompressed on the fly into
#    compute_dtype for each forward pass.
# ---------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,      # quantizes the quantization constants too
    bnb_4bit_compute_dtype=compute_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},                  # put everything on GPU 0
    dtype=compute_dtype,
)
model.config.use_cache = False           # incompatible with gradient checkpointing

# ---------------------------------------------------------------
# 2. The "LoRA": small trainable matrices attached to the frozen layers.
#    r = rank (capacity of the adapter), alpha = scaling of its output.
# ---------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------
# 3. Data. We split each example into prompt (system + user) and
#    completion (assistant), so the loss is computed ONLY on Baymax's
#    reply, not on the question we fed in.
# ---------------------------------------------------------------
def to_prompt_completion(ex):
    msgs = ex["messages"]
    return {"prompt": msgs[:-1], "completion": [msgs[-1]]}

data = load_dataset("json", data_files={"train": "train.jsonl", "eval": "eval.jsonl"})
data = data.map(to_prompt_completion, remove_columns=["messages"])

# ---------------------------------------------------------------
# 4. Training settings (tuned for a 6GB card).
#    Effective batch = 2 x 8 = 16 examples per optimizer step.
# ---------------------------------------------------------------
args = SFTConfig(
    output_dir="outputs/baymax-lora-v7",
    num_train_epochs=2,
    per_device_train_batch_size=2,       # if out-of-memory: 1 (and set accumulation to 16)
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_steps=5,
    eval_strategy="epoch",               # measure eval loss after each epoch
    save_strategy="epoch",
    save_total_limit=2,
    bf16=use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=True,         # recompute activations to save VRAM
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",            # memory-friendly optimizer
    max_length=MAX_LEN,
    completion_only_loss=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["eval"],
    processing_class=tokenizer,
    peft_config=lora_config,
)
trainer.model.print_trainable_parameters()   # expect well under 5% trainable

trainer.train()

trainer.model.save_pretrained(OUT_DIR)       # saves only the adapter
tokenizer.save_pretrained(OUT_DIR)
print(f"Adapter saved to {OUT_DIR}")