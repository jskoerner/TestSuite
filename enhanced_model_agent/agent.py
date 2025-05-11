"""
Model Callbacks Example with Before and After Processing
"""

import copy
from datetime import datetime
from typing import Optional
import time

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from .rag.retriever import Retriever

# Initialize the Retriever (do this once, globally)
retriever = Retriever(
    csv_path="enhanced_model_agent/data/embeddings.csv",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def print_state_debug(label, state):
    print(f"[DEBUG] {label}:")
    try:
        for k, v in state._value.items():
            try:
                print(f"  {k}: {v!r}")
            except Exception as e:
                print(f"  {k}: <unprintable: {e}>")
    except Exception as e:
        print(f"  <Could not print state: {e}>")

def before_agent_callback(callback_context):
    # Store both UNIX timestamp and readable string
    now = time.time()
    # Initialize timing state if it doesn't exist
    if "timing" not in callback_context.state:
        callback_context.state["timing"] = {}
    # Store agent start time
    callback_context.state["timing"]["agent_start_time"] = now
    callback_context.state["timing"]["agent_start_time_str"] = datetime.fromtimestamp(now).isoformat()
    # Ensure timing is persisted in the root state
    callback_context.state["agent_start_time"] = now
    callback_context.state["agent_start_time_str"] = datetime.fromtimestamp(now).isoformat()
    print("Agent processing started.")
    print_state_debug("State at end of before_agent_callback", callback_context.state)
    return None

def after_agent_callback(callback_context):
    # Store both UNIX timestamp and readable string
    now = time.time()
    # Ensure timing state exists
    if "timing" not in callback_context.state:
        callback_context.state["timing"] = {}
    # Store agent end time
    callback_context.state["timing"]["agent_end_time"] = now
    callback_context.state["timing"]["agent_end_time_str"] = datetime.fromtimestamp(now).isoformat()
    # Ensure timing is persisted in the root state
    callback_context.state["agent_end_time"] = now
    callback_context.state["agent_end_time_str"] = datetime.fromtimestamp(now).isoformat()
    
    # Calculate and store elapsed time
    agent_start = callback_context.state["timing"].get("agent_start_time")
    if agent_start:
        elapsed = now - agent_start
        callback_context.state["timing"]["agent_elapsed_time"] = elapsed
        callback_context.state["agent_elapsed_time"] = elapsed
        print(f"Agent processing took {elapsed:.2f} seconds.")
    
    # Print RAG timing if available
    rag_elapsed = callback_context.state.get("rag_elapsed_time")
    if rag_elapsed:
        print(f"RAG retrieval took {rag_elapsed:.2f} seconds.")
    
    # Ensure the timing state is preserved
    callback_context.state["agent_timing"] = callback_context.state["timing"]
    print_state_debug("State at end of after_agent_callback", callback_context.state)
    return None

def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Before model callback that:
    1. Logs request information
    2. Rag 
    3. Tracks request timing
    """
    # Get the state and agent name
    state = callback_context.state
    agent_name = callback_context.agent_name

    # Extract the last user message
    last_user_message = ""
    if llm_request.contents and len(llm_request.contents) > 0:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts and len(content.parts) > 0:
                if hasattr(content.parts[0], "text") and content.parts[0].text:
                    last_user_message = content.parts[0].text
                    break

    # Time the RAG retrieval
    rag_start = time.time()
    retrieved_chunks = retriever.retrieve(last_user_message, top_k=1)
    rag_elapsed = time.time() - rag_start
    callback_context.state["rag_elapsed_time"] = rag_elapsed
    if retrieved_chunks:
        context, score = retrieved_chunks[0]
        # Prepend the context to the user's message for the LLM
        last_user_message = f"Context from nutrition handbook: {context}\n\nUser question: {last_user_message}"
        state["retriever_confidence"] = float(score)
    else:
        state["retriever_confidence"] = None

    # Log the request
    print(f"\n=== REQUEST STARTED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Agent: {agent_name}")
    print(f"Message: {last_user_message[:100]}...")

    # Store message for after callback
    state["last_user_message"] = last_user_message

    # Store model start time
    now = time.time()
    if "timing" not in callback_context.state:
        callback_context.state["timing"] = {}
    callback_context.state["timing"]["model_start_time"] = now
    callback_context.state["timing"]["model_start_time_str"] = datetime.fromtimestamp(now).isoformat()
    # Ensure timing is persisted in the root state
    callback_context.state["model_start_time"] = now
    callback_context.state["model_start_time_str"] = datetime.fromtimestamp(now).isoformat()
    
    # Start RAG timer just before retrieval
    callback_context.state["timing"]["rag_start_time"] = time.time()
    callback_context.state["rag_start_time"] = time.time()
    print_state_debug("State at end of before_model_callback", callback_context.state)
    print("✓ Request approved for processing")

    return None

def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """
    After model callback that:
    1. Transforms negative words to positive ones
    2. Adds a signature with response time
    3. Logs completion
    """
    # Skip if no response
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return None

    # Get the state
    state = callback_context.state
    
    # Store model end time
    now = time.time()
    if "timing" not in callback_context.state:
        callback_context.state["timing"] = {}
    callback_context.state["timing"]["model_end_time"] = now
    callback_context.state["timing"]["model_end_time_str"] = datetime.fromtimestamp(now).isoformat()
    # Ensure timing is persisted in the root state
    callback_context.state["model_end_time"] = now
    callback_context.state["model_end_time_str"] = datetime.fromtimestamp(now).isoformat()
    
    # Calculate model elapsed time
    model_start = callback_context.state["timing"].get("model_start_time")
    if model_start:
        elapsed = now - model_start
        callback_context.state["timing"]["model_elapsed_time"] = elapsed
        callback_context.state["model_elapsed_time"] = elapsed
        print(f"Model processing took {elapsed:.2f} seconds.")
    
    # Calculate RAG elapsed time
    rag_start = callback_context.state["timing"].get("rag_start_time")
    if rag_start:
        rag_elapsed = now - rag_start
        callback_context.state["timing"]["rag_elapsed_time"] = rag_elapsed
        callback_context.state["rag_elapsed_time"] = rag_elapsed
    
    # Ensure the timing state is preserved
    callback_context.state["model_timing"] = callback_context.state["timing"]
    # --- Ensure agent timing is also present in the root state ---
    for key in [
        "agent_start_time", "agent_start_time_str",
        "agent_end_time", "agent_end_time_str",
        "agent_elapsed_time"
    ]:
        if key in callback_context.state["timing"]:
            callback_context.state[key] = callback_context.state["timing"][key]
    print_state_debug("State at end of after_model_callback", callback_context.state)
    
    # Extract and modify the response text
    response_text = ""
    for part in llm_response.content.parts:
        if hasattr(part, "text") and part.text:
            response_text += part.text

    if not response_text:
        return None

    # Word replacements for more positive language
    replacements = {
        "problem": "challenge",
        "difficult": "complex",
        "hard": "challenging",
        "issue": "situation",
        "bad": "suboptimal"
    }

    # Perform replacements
    modified_text = response_text
    modified = False
    for original, replacement in replacements.items():
        if original in modified_text.lower():
            modified_text = modified_text.replace(original, replacement)
            modified_text = modified_text.replace(
                original.capitalize(), replacement.capitalize()
            )
            modified = True

    # Add signature with response time
    response_time = elapsed if "model_elapsed_time" in callback_context.state["timing"] else None
    signature = f"\n\n[Response time: {response_time:.2f}s]"
    modified_text += signature

    # Create modified response
    modified_parts = [copy.deepcopy(part) for part in llm_response.content.parts]
    for i, part in enumerate(modified_parts):
        if hasattr(part, "text") and part.text:
            modified_parts[i].text = modified_text

    print(f"=== REQUEST COMPLETED in {response_time:.2f}s ===")
    
    return LlmResponse(content=types.Content(role="model", parts=modified_parts))

# Create the Agent
root_agent = LlmAgent(
    name="enhanced_model_agent",
    model="gemini-2.0-flash",
    description="An agent that demonstrates both before and after model callbacks",
    instruction="""
    You are a helpful and positive assistant.
    
    Your job is to:
    - Answer user questions concisely
    - Provide factual information
    - Be friendly and respectful
    """,
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback
)
