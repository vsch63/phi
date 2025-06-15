from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = r"C:\Users\vscha\Downloads\phi2_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
