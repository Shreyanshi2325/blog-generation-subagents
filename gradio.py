import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/generate-blog"

def generate_blog(topic):
    if not topic.strip():
        return "❌ Please enter a topic"

    response = requests.post(
        API_URL,
        json={"topic": topic},
        timeout=120
    )

    return response.json()["output"]

with gr.Blocks(title="AI Blog Generator") as demo:
    gr.Markdown("## ✍️ AI Blog Generator (Agentic AI)")

    topic = gr.Textbox(
        label="Blog Topic",
        placeholder="Enter blog topic (e.g. Global Warming)"
    )

    generate_btn = gr.Button("Generate Blog 🚀")
    output = gr.Textbox(
        label="Generated Blog",
        lines=15
    )

    generate_btn.click(
        fn=generate_blog,
        inputs=topic,
        outputs=output
    )

demo.launch()
