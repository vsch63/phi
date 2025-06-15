import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Adjust MODEL_PATH to your phi2_model folder
MODEL_PATH = os.path.join(os.getcwd(), "phi2_model")
print("Loading tokenizer and model from:", MODEL_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model with disk offload (similar to app.py)
config = AutoConfig.from_pretrained(MODEL_PATH)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model = load_checkpoint_and_dispatch(
    model,
    MODEL_PATH,
    device_map="auto",
    no_split_module_classes=["GPTNeoXLayer"],
    offload_folder="offload_test",   # separate offload folder
    offload_state_dict=True,
    dtype=torch.float32
)

# Sanity-check prompt
test_prompt = (
    "You are a COBOL expert. Read the following COBOL snippet:\n"
    "IDENTIFICATION DIVISION.\n"
    "PROGRAM-ID. TEST.\n"
    "PROCEDURE DIVISION.\n"
    "    DISPLAY \"HELLO\".\n"
    "    STOP RUN.\n\n"
    "Question: What does this program do?\n"
    "Answer:"
)

# Tokenize and generate
inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
print(f"Sanity check: prompt tokens = {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Full output:\n", output_text)
suffix = output_text[len(test_prompt):].strip() if output_text.startswith(test_prompt) else output_text.strip()
print("Extracted answer:", suffix)
