
# Install necessary packages
!pip install -q gradio
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install --upgrade gradio

import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Initialize memory
memory = {}

def update_memory(key, value):
    """Update memory with a key-value pair."""
    memory[key] = value
    return f"Memory updated: {key} is now {value}"

def generate_text(inp):
    """Generate text based on input using GPT-2."""
    input_ids = tokenizer.encode(inp, return_tensors='tf')
    beam_output = model.generate(
        input_ids, 
        max_length=100, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

# Gradio interface
input_textbox = gr.Textbox(lines=2, placeholder="Enter a sentence...")
memory_key_textbox = gr.Textbox(lines=1, placeholder="Memory key...")
memory_value_textbox = gr.Textbox(lines=1, placeholder="Memory value...")

generate_btn = gr.Button("Generate Text")
update_memory_btn = gr.Button("Update Memory")

generate_text_interface = gr.Interface(
    fn=generate_text,
    inputs=input_textbox,
    outputs="text",
    title="GPT-2 Text Generation"
)

# Launch the interface
generate_text_interface.launch()
