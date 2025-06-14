from flask import Flask, render_template, request
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

app = Flask(__name__)

# Load model and tokenizer
model_path = r"notebooks\t5_grammar_corrector"

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prediction function
def correct_grammar(text):
    input_text = "correct: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="tf", padding=True)
    output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return corrected_text

@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    output_text = ""

    if request.method == "POST":
        input_text = request.form["input_text"]
        output_text = correct_grammar(input_text)

    return render_template("index.html", input_text=input_text, output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)
