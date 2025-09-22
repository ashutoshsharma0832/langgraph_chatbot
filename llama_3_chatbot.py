from typing import Annotated, List, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
import gradio as gr

#LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]

#LangGraph builder
graph_builder = StateGraph(State)

# model
llm = ChatOllama(model="llama3")

# Chatbot function for LangGraph
def chatbot(state: State) -> State:
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error: {str(e)}"}]}

#LangGraph nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

#Convert Gradio messages LangChain message 
def dicts_to_langchain_messages(history: Optional[List[Dict]]) -> List:
    messages = []
    if not history:
        return messages
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages

#Convert LangChain message objects to Gradio messages format
def langchain_messages_to_dicts(messages: List) -> List[Dict]:
    result = []
    for msg in messages:
        # msg is HumanMessage or AIMessage
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        result.append({"role": role, "content": msg.content})
    return result

# 8. Gradio chat handler function
def gradio_chat(user_input: str, history: Optional[List[Dict]]):
    if history is None:
        history = []

    # Convert Gradio history to LangChain messages
    past_messages = dicts_to_langchain_messages(history)
    # Append current user input as a HumanMessage
    past_messages.append(HumanMessage(content=user_input))

    # messages to LangGraph state
    state = {"messages": past_messages}
    final_response = ""

    try:
        for event in graph.stream(state, stream_mode="values"):
            # Debug print, remove if verbose
            print("Event:", event)
            if "messages" in event:
                msgs = event["messages"]
                if msgs and hasattr(msgs[-1], "content"):
                    # The last message should be AIMessage from assistant
                    final_response = msgs[-1].content
    except Exception as e:
        final_response = f"Error: {e}"
        print(final_response)

    # adding history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final_response})


    return history, ""

# Gradio UI
with gr.Blocks() as demo:
    chatbot_ui = gr.Chatbot(elem_id="chatbot", label="LangGraph + Ollama Chatbot", height=400, type="messages")
    user_input = gr.Textbox(show_label=False, placeholder="Type your message here and press Enter")

    def user_submit(text, history):
        if not text.strip():
            return history, ""
        return gradio_chat(text, history)

    user_input.submit(user_submit, inputs=[user_input, chatbot_ui], outputs=[chatbot_ui, user_input])

demo.launch(share=True)
