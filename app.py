import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "HuggingFaceTB/SmolVLM-Instruct"
local_mode_ckpt = "/data/students/gerry/repos/jetson-vlm/local_model"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load model
model = AutoModelForVision2Seq.from_pretrained(local_mode_ckpt,torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2")
model = model.to(DEVICE)

current_image = None
def generate_text(img, text):
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[current_image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return generated_texts[0]

def capture_image(img):
    global current_image
    current_image = img.copy()
    print(type(current_image))


# Create a Gradio interface with text and image as input and text as output
import gradio as gr
with gr.Blocks() as demo:
    with gr.Group():
        gr.Markdown("## Image and Text Processing App")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", streaming=True)
            with gr.Column():
                text_input = gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text")
    
        image_input.stream(capture_image, [image_input], [], time_limit=30, stream_every=0.075)     

        output_text = gr.Textbox(label="Output Text")        
        submit_button = gr.Button("Submit")
        
        submit_button.click(
            fn=generate_text,
            inputs=[image_input, text_input],
            outputs=output_text
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3123)