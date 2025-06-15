import os
import uuid
import torch
import chardet
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\vscha\Downloads\phi2_cobol_chatbot\cobol_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model setup
MODEL_ID = os.path.join(os.getcwd(), "phi2_model")  # Assuming model is in ./phi2
print("Loading tokenizer and model from:", MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

model = load_checkpoint_and_dispatch(
    model,
    MODEL_ID,
    device_map="auto",
    no_split_module_classes=["GPTNeoXLayer"],
    offload_folder="offload",
    offload_state_dict=True,
    dtype=torch.float32  # CPU-safe
)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(4096)
        result = chardet.detect(rawdata)
        return result['encoding'] or 'utf-8'

def read_cobol_file(filepath):
    encoding = detect_encoding(filepath)
    with open(filepath, "r", encoding=encoding, errors="ignore") as f:
        return f.read()

def clean_cobol_code(cobol_code):
    return cobol_code.encode("ascii", "ignore").decode(errors="ignore")

def ask_phi2_about_cobol(cobol_code, question):
    cobol_code = clean_cobol_code(cobol_code)

    prompt = f"""You are a COBOL expert. Read the following COBOL program and answer the user's question.

COBOL Program:
{cobol_code}

User's Question:
{question}

Answer:"""

    # Truncate input so we don’t exceed model limit
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1848)
    prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}

    print(f"Prompt token length: {prompt_inputs['input_ids'].shape[1]} tokens")
    print("Generating answer...")

    with torch.no_grad():
        try:
            output = model.generate(
                **prompt_inputs,
                max_new_tokens=200,
                do_sample=False,  # Greedy decoding is safer
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.unk_token_id]] if tokenizer.unk_token_id is not None else None
            )
        except Exception as e:
            return f"Error during generation: {str(e)}"

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()

@app.route("/")
def index():
    cobol_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".COB")]
    return render_template("index.html", cobol_files=cobol_files)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({"message": "Uploaded successfully", "filename": filename})
    return jsonify({"error": "No file uploaded"}), 400

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    cobol_file = data.get("filename")
    question = data.get("question")
    if not cobol_file or not question:
        return jsonify({"error": "Filename and question required"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, cobol_file)
    if not os.path.exists(filepath):
        return jsonify({"error": "COBOL file not found"}), 404

    try:
        cobol_code = read_cobol_file(filepath)
        answer = ask_phi2_about_cobol(cobol_code, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
