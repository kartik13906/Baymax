"""
prepare_dataset.py - Baymax Phase 2, step 3: format data for fine-tuning.
 
Reads:   baymax_data.jsonl
Writes:  train.jsonl, eval.jsonl   (each row: {"messages": [system, user, assistant]})
 
Run:     python prepare_dataset.py
"""
 
import json
import random
 
import numpy as np
from transformers import AutoTokenizer
 
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # must match the model you will fine-tune
IN_PATH = "baymax_final.jsonl"
MAX_TOKENS = 512        # sequence limit for a 6GB GPU; longer rows get dropped
EVAL_FRACTION = 0.05
BEHAVIOUR_COPIES = 3    # train on each hand-written behaviour example this many times
 
# The system prompt the FINAL model will always see. Keep it short and stable:
# the model learns "when I see this, I act as Baymax".
SYSTEM = (
    "You are Baymax, a gentle and caring personal healthcare companion. "
    "You give safe, simple guidance and recommend seeing a doctor when something may be serious."
)
 
 
def main():
    # 1. Load the tokenizer that belongs to the base model.
    #    A different tokenizer/template would quietly damage the fine-tune.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
 
    with open(IN_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} rows")
 
    # Rows added by finalize_data.py (dose / diagnosis / off-topic / crisis /
    # emergency examples) carry a "kind" field. We remember their questions so
    # we can repeat them in the training set below.
    behaviour_q = {r["question"] for r in rows if "kind" in r}
 
    examples, lengths = [], []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["baymax_answer"]},
        ]
        # 2. apply_chat_template turns messages into the exact ChatML text
        #    Qwen was trained on (<|im_start|>role ... <|im_end|>).
        text = tok.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tok(text)["input_ids"])
        lengths.append(n_tokens)
        if n_tokens <= MAX_TOKENS:
            examples.append({"messages": messages})
 
    # 3. Token-length stats: this is how you pick MAX_TOKENS from real numbers.
    arr = np.array(lengths)
    print(f"Token lengths -> mean {arr.mean():.0f}, 95th pct {np.percentile(arr, 95):.0f}, max {arr.max()}")
    print(f"Kept {len(examples)}/{len(rows)} rows under {MAX_TOKENS} tokens")
 
    # 4. Shuffle and split.
    random.seed(42)
    random.shuffle(examples)
    n_eval = max(1, int(len(examples) * EVAL_FRACTION))
    eval_set, train_set = examples[:n_eval], examples[n_eval:]
 
    # Upsample behaviour examples in TRAIN only (the eval set stays clean).
    # Only ~6% of the data teaches Baymax when to refuse, which was too weak a signal.
    extra = [ex for ex in train_set if ex["messages"][1]["content"] in behaviour_q]
    train_set = train_set + extra * (BEHAVIOUR_COPIES - 1)
    random.shuffle(train_set)
    print(f"Behaviour examples in train: {len(extra)} x {BEHAVIOUR_COPIES} copies")
 
    for path, data in [("train.jsonl", train_set), ("eval.jsonl", eval_set)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(train_set)} train rows, {len(eval_set)} eval rows")
 
    # 5. Look at what the model will actually see.
    print("\n--- One formatted example ---")
    print(tok.apply_chat_template(train_set[0]["messages"], tokenize=False))
 
 
if __name__ == "__main__":
    main()
 