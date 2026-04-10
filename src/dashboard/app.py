"""
Warp-Claw Dashboard
Optional Streamlit UI for monitoring and interaction.
"""

import streamlit as st
import requests
import json
from datetime import datetime


st.set_page_config(
    page_title="Warp-Claw Dashboard",
    page_icon="🤖",
    layout="wide"
)


def get_api_url() -> str:
    """Get API URL from environment or default."""
    import os
    return os.environ.get("WARP_CLAW_API_URL", "http://localhost:8000")


def fetch_models():
    """Fetch available models."""
    try:
        r = requests.get(f"{get_api_url()}/v1/models", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fetch_agents():
    """Fetch active agents."""
    try:
        r = requests.get(f"{get_api_url()}/v1/agents", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def send_chat(prompt: str, model: str = "qwen-0.5b", stream: bool = False):
    """Send a chat message."""
    try:
        r = requests.post(
            f"{get_api_url()}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream
            },
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def spawn_council(prompt: str, council_types: list):
    """Spawn a council."""
    try:
        r = requests.post(
            f"{get_api_url()}/v1/agents/spawn",
            json={
                "prompt": prompt,
                "council_types": council_types
            },
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# === UI ===

st.title("🤖 Warp-Claw Dashboard")
st.markdown("Multi-agent council system for Apple Silicon")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    selected_model = st.selectbox(
        "Model",
        ["qwen-0.5b", "qwen-1.5b", "llama-3.2-1b"]
    )
    
    st.divider()
    
    st.subheader("Active Agents")
    agents = fetch_agents()
    if "error" not in agents:
        st.write(f"Active councils: {agents.get('council_count', 0)}")
    else:
        st.write("API not connected")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Council", "Models"])

with tab1:
    st.header("Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input
    prompt = st.chat_input("Send a message...")
    
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.spinner("Generating..."):
            response = send_chat(prompt, selected_model)
        
        if "error" not in response:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": content})
            
            with st.chat_message("assistant"):
                st.write(content)
        else:
            st.error(f"Error: {response.get('error')}")

with tab2:
    st.header("Council Spawner")
    
    with st.form("council_form"):
        council_prompt = st.text_area("Prompt", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            use_research = st.checkbox("Research Council", value=True)
            use_code = st.checkbox("Code Council")
        with col2:
            use_creative = st.checkbox("Creative Council")
            use_meta = st.checkbox("Meta Council")
        
        submit = st.form_submit_button("Spawn Council")
        
        if submit and council_prompt:
            council_types = []
            if use_research:
                council_types.append("research")
            if use_code:
                council_types.append("code")
            if use_creative:
                council_types.append("creative")
            if use_meta:
                council_types.append("meta")
            
            with st.spinner("Spawning..."):
                result = spawn_council(council_prompt, council_types)
            
            if "error" not in result:
                st.success(f"Council spawned: {result.get('council_id')}")
                
                # Show status
                council_id = result.get("council_id")
                status = requests.get(
                    f"{get_api_url()}/v1/agents/status/{council_id}"
                ).json()
                
                st.json(status)
            else:
                st.error(f"Error: {result.get('error')}")

with tab3:
    st.header("Models")
    
    models = fetch_models()
    
    if "error" not in models:
        for model in models.get("data", []):
            with st.expander(model.get("id", "unknown")):
                st.write(f"ID: {model.get('id')}")
                st.write(f"Created: {model.get('created')}")
                st.write(f"Owner: {model.get('owned_by')}")
    else:
        st.error(f"Error loading models: {models.get('error')}")
        st.info("Is the API server running?")


# Footer
st.divider()
st.caption(f"Warp-Claw v0.1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")