import gradio as gr
from transformers import pipeline
from langdetect import detect
from huggingface_hub import login
import os

login(token=os.getenv("HF_Token"))
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
bg_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-bg-en")
en_to_bg = pipeline("translation", model="Helsinki-NLP/opus-mt-en-bg")

def generate_response(user_input, top_p, temperature, chat_counter, chatbot, history, request: gr.Request):
    lang = detect(user_input)
    print(f"Detected language: {lang}")

    # Translate if needed
    if lang == "bg":
        user_input_translated = bg_to_en(user_input)[0]["translation_text"]
    else:
        user_input_translated = user_input

    # Create prompt
    prompt = f"""
Comprehensive E-commerce Bot Prompt:

Role and Persona:
You are an exceptionally helpful, friendly, and clear e-commerce business assistant, specifically designed for beginners and aspiring entrepreneurs. Your primary goal is to guide users through the process of establishing, developing, and scaling their online businesses, drawing upon the proven strategies and real-world experience embodied by the Masterbrand e-commerce training program. You should embody the spirit of a seasoned mentor who simplifies complex e-commerce concepts into actionable, easy-to-understand advice.

Core Functionality and Expertise:
Your expertise is rooted in the principles taught by Masterbrand, which focuses on building successful e-commerce brands from the ground up. You are proficient in:
1. Foundational E-commerce Setup: Guiding users on how to start an e-commerce brand from scratch, including initial ideation, niche selection, and basic platform setup.
2. Business Development and Growth: Providing insights into developing online businesses, implementing effective growth strategies, and structuring a business for long-term success.
3. Strategic Implementation: Offering practical advice on applying working strategies and business structures that lead to real sales and tangible results.
4. Brand Building: Assisting users in transforming their initial ideas into established, profitable brands.
5. Market Context (Bulgaria): While your advice is universally applicable, you understand the nuances of the e-commerce landscape, particularly in regions like Bulgaria, given Masterbrand's origin as the #1 ECOM training in Bulgaria.
6. Learning Resource Guidance: You can refer to the types of resources found in Masterbrand (e.g., video lessons, modules, practical resources) to explain concepts or suggest learning paths.

Tone and Communication Style:
Your communication should always be:
- Simple and Clear: Avoid jargon where possible, and explain complex terms in an accessible manner.
- Friendly and Encouraging: Maintain a supportive and motivating tone, especially for beginners who may feel overwhelmed.
- Action-Oriented: Provide advice that is practical and can be immediately applied by the user.
- Empathetic: Acknowledge the challenges beginners face and offer reassurance, similar to how Atanas Peltekov, the founder of Masterbrand, emphasizes overcoming initial hurdles.

Constraints and Limitations:
You are an assistant, not a decision-maker. Your role is to provide information and guidance, not to make business decisions for the user.
You should not provide financial or legal advice. Always recommend consulting with professionals for such matters.
Your knowledge is based on the principles of successful e-commerce as observed and taught by Masterbrand. While comprehensive, it is not exhaustive of all possible e-commerce strategies.

Example Interaction Flow:
User: "I want to start an online store, but I have no idea where to begin."
Bot: "That's a fantastic goal! Many successful e-commerce brands started exactly where you are now. To begin, let's think about what you're passionate about or what problem you want to solve for customers. This will help us choose a niche. Once we have a niche, we can explore simple ways to set up your first online presence. What kind of products or services are you considering?"

Integration with User Input:
Always consider the user's specific input and tailor your advice accordingly. If the user provides a specific question or scenario, address it directly while integrating the core principles of e-commerce success.

User said:
{user_input_translated}

Give simple, friendly, clear advice:
"""

    # Generate response
    response_text = generator(prompt, max_length=200, top_p=top_p, temperature=temperature, do_sample=True)[0]["generated_text"]

    if lang == "bg":
        response_text = en_to_bg(response_text)[0]["translation_text"]

    # Format for type="messages"
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response_text})

    chat_counter += 1
    return history, history, chat_counter, "✅ Success", gr.update(value=''), gr.update(interactive=True)

def reset_textbox():
    return gr.update(value='', interactive=False), gr.update(interactive=False)

# ==== Custom CSS ====
custom_css = """ 
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* {
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
    min-height: 100vh !important;
}
.main-header {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
    padding: 2rem 0 !important;
    text-align: center !important;
    border-bottom: 2px solid #ff4d00 !important;
    margin-bottom: 2rem !important;
}
.main-header h1 {
    color: #ffffff !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 20px rgba(255, 77, 0, 0.3) !important;
}
.main-header p {
    color: #cccccc !important;
    font-size: 1.2rem !important;
}
.chatbot-container {
    background: rgba(42, 42, 42, 0.8) !important;
    border: 2px solid #ff4d00 !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(255, 77, 0, 0.2) !important;
    margin-bottom: 2rem !important;
}
.input-container textarea {
    background: #2a2a2a !important;
    border: 1px solid #ff4d00 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    padding: 1rem !important;
    resize: none !important;
}
.input-container textarea:focus {
    border-color: #ff6d20 !important;
    box-shadow: 0 0 15px rgba(255, 77, 0, 0.3) !important;
    outline: none !important;
}
.submit-button {
    background: linear-gradient(135deg, #ff4d00 0%, #ff6d20 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(255, 77, 0, 0.3) !important;
}
.submit-button:hover {
    background: linear-gradient(135deg, #ff6d20 0%, #ff8d40 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 77, 0, 0.4) !important;
}
.slider-container label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.status-container textarea {
    background: transparent !important;
    border: none !important;
    color: #4CAF50 !important;
    font-weight: 600 !important;
}
.footer {
    display: none !important;
}
"""

# ==== Theme ====
masterbrand_theme = gr.themes.Base(primary_hue="orange").set(
    body_background_fill="#1a1a1a",
    body_text_color="#ffffff",
    border_color_accent="#ff4d00",
    button_primary_background_fill="#ff4d00",
    button_primary_background_fill_hover="#ff6d20",
    button_primary_text_color="#ffffff"
)

# ==== UI ====
with gr.Blocks(theme=masterbrand_theme, css=custom_css, title="MasterBrand AI Assistant") as demo:
    gr.HTML("""
    <div class="main-header">
        <h1>🛍️ MASTERBRAND AI ASSISTANT</h1>
        <p>Your Personal E-commerce Business Expert - Available in English & Bulgarian</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="💬 Chat with MasterBrand AI",
                height=400,
                elem_classes=["chatbot-container"],
                type="messages"
            )

            inputs = gr.Textbox(
                placeholder="Ask me anything about your e-commerce business...",
                label="Your Question",
                lines=2,
                elem_classes=["input-container"]
            )

            with gr.Row():
                submit_btn = gr.Button(
                    "🚀 Get Expert Advice",
                    variant="primary",
                    size="lg",
                    elem_classes=["submit-button"]
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-container"]
                )

    with gr.Accordion("⚙️ Advanced Settings", open=False):
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="🎯 Creativity (Top-p)", elem_classes=["slider-container"])
        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="🌡️ Temperature", elem_classes=["slider-container"])

    # State
    state = gr.State([])
    chat_counter = gr.Number(value=0, visible=False)

    # Event Handlers
    inputs.submit(reset_textbox, [], [inputs, submit_btn], queue=False)
    inputs.submit(generate_response, [inputs, top_p, temperature, chat_counter, chatbot, state], [chatbot, state, chat_counter, status_box, inputs, submit_btn])
    submit_btn.click(reset_textbox, [], [inputs, submit_btn], queue=False)
    submit_btn.click(generate_response, [inputs, top_p, temperature, chat_counter, chatbot, state], [chatbot, state, chat_counter, status_box, inputs, submit_btn])

# ==== Launch ====
if __name__ == "__main__":
    # Check if running on Render
    if os.environ.get("RENDER") == "true":
        demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=10000)
    else:
        demo.queue(max_size=10).launch()
