from typing import List, Dict, Generator
import gradio as gr
from groq import Groq
import os
import json
from datetime import datetime

# ----------------------------- CONFIG -----------------------------
# API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "...")
client = Groq(api_key=GROQ_API_KEY)

# Sessions folder
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Models
models = {
    "gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192},
    "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000},
    "llama-3.1-8b-instant": {"name": "LLaMA3.1-8b-instant", "tokens": 128000},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768},
}

# ----------------------------- UTILS -----------------------------

def list_sessions():
    files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    return sorted(files, reverse=True)

def save_session(session_name: str, history: List[Dict]):
    with open(os.path.join(SESSIONS_DIR, session_name), "w") as f:
        json.dump(history, f)

def load_session(session_name: str):
    with open(os.path.join(SESSIONS_DIR, session_name), "r") as f:
        return json.load(f)

def generate_stream(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content

def chat_with_groq(message: str, history: List[Dict], model_choice: str, max_tokens: int):
    messages = history + [{"role": "user", "content": message}]
    try:
        chat_completion = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )
        full_response = ""
        for token in generate_stream(chat_completion):
            full_response += token
            yield full_response
    except Exception as e:
        yield f"Error: {e}"

# ----------------------------- GRADIO APP -----------------------------

with gr.Blocks(title="Groq + Gradio Chatbot") as demo:
    gr.Markdown("## Groq Chatbot")

    with gr.Row():
        with gr.Column(scale=1):
            session_list = gr.Radio(
                label="Previous Sessions",
                choices=list_sessions(),
                value=None,
                interactive=True,
            )
            new_chat_button = gr.Button("➕ New Chat")
        
        with gr.Column(scale=4):
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=list(models.keys()),
                value="llama3-8b-8192",
                interactive=True,
            )
            token_slider = gr.Slider(
                label="Max Tokens",
                minimum=512,
                maximum=32768,
                step=512,
                value=4096,
                interactive=True,
            )
            chatbot = gr.Chatbot(label="Chat", height=500, render_markdown=True)
            user_input = gr.Textbox(placeholder="Type a message and press Enter", show_label=False)

    # States
    current_session = gr.State("")
    history_state = gr.State([])

    # ------------------ Logic Functions ------------------

    def respond(message, history, model_choice, max_tokens, session_file):
        if not message.strip():
            return history, "", session_file, history

        history = history or []

        # Auto-create session if not exists
        if not session_file:
            session_file = datetime.now().strftime("session_%Y_%m_%d_%H%M%S.json")
            save_session(session_file, history)

        stream_output = chat_with_groq(message, history, model_choice, max_tokens)

        full_response = ""
        for partial in stream_output:
            full_response = partial
            updated_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response}
            ]

            # Save updated history
            save_session(session_file, updated_history)

            # Format for Gradio Chatbot
            formatted_chat = [
                [updated_history[i]["content"], updated_history[i+1]["content"]]
                for i in range(0, len(updated_history), 2)
            ]

            yield formatted_chat, "", session_file, updated_history

    def new_chat():
        session_name = datetime.now().strftime("session_%Y_%m_%d_%H%M%S.json")
        save_session(session_name, [])
        return [], "", session_name, list_sessions(), session_name, []

    def load_chat(session_name):
        if session_name is None:
            return [], "", "", []
        history = load_session(session_name)

        # Format for Gradio Chatbot
        formatted_chat = [
            [history[i]["content"], history[i+1]["content"]]
            for i in range(0, len(history), 2)
        ]
        return formatted_chat, "", session_name, history

    # ------------------ Event Handlers ------------------

    user_input.submit(
        fn=respond,
        inputs=[user_input, history_state, model_dropdown, token_slider, current_session],
        outputs=[chatbot, user_input, current_session, history_state],
        concurrency_limit=1
    )

    new_chat_button.click(
        fn=new_chat,
        inputs=[],
        outputs=[chatbot, user_input, current_session, session_list, session_list, history_state]
    )

    session_list.change(
        fn=load_chat,
        inputs=[session_list],
        outputs=[chatbot, user_input, current_session, history_state]
    )

# ----------------------------- LAUNCH -----------------------------
demo.launch(share=True)

