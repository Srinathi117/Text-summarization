pip install transformers gradio rouge-score

from transformers import pipeline 
import gradio as gr
from rouge_score import rouge_scorer

# Load the PEGASUS model
summarizer = pipeline("summarization", model="google/pegasus-large")

# Define ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Enhanced summarization function with adjustable settings
def summarize_text_advanced(article):
    summary = summarizer(
        article, 
        max_length=150,  # Adjusted max length for more concise summaries
        min_length=40,   # Adjusted min length for relevance
        do_sample=True,   # Enable sampling for varied output
        top_k=40,        # Limit sampling to top 40 words
        top_p=0.85,      # Use nucleus sampling with p=0.85
        temperature=0.6,  # Control randomness for coherent output
        num_beams=4,     # Use beam search for quality summaries
        early_stopping=True
    )
    return summary[0]['summary_text']

# Function to calculate ROUGE accuracy
def calculate_rouge_accuracy(reference_summary, generated_summary):
    scores = scorer.score(reference_summary, generated_summary)
    rouge_1 = scores['rouge1'].fmeasure
    rouge_2 = scores['rouge2'].fmeasure
    rouge_L = scores['rougeL'].fmeasure
    return f"ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}, ROUGE-L: {rouge_L:.4f}"

# Gradio interface for advanced summarization
with gr.Blocks() as demo_abstractive:
    gr.Markdown("# Enhanced Summarization with PEGASUS and ROUGE Score")
    input_text = gr.Textbox(label="Input Article", lines=10, placeholder="Enter your article here...")
    reference_summary = gr.Textbox(label="Reference Summary", lines=5, placeholder="Enter reference summary here...")
    output_text = gr.Textbox(label="Generated Summary", lines=5)
    rouge_score_output = gr.Textbox(label="ROUGE Score", lines=2)

    summarize_button = gr.Button("Summarize and Calculate ROUGE")
    
    def process_abstractive(article, reference):
        generated_summary = summarize_text_advanced(article)
        rouge_score = calculate_rouge_accuracy(reference, generated_summary)
        return generated_summary, rouge_score
    
    summarize_button.click(fn=process_abstractive, inputs=[input_text, reference_summary], outputs=[output_text, rouge_score_output])

demo_abstractive.launch()

