import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if Gemini API key is configured
HAS_GEMINI_KEY = os.environ.get("GEMINI_API_KEY") is not None
if not HAS_GEMINI_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not set.")
    print("The dashboard will run, but the AI agent chat interface will be disabled.")
    print("Please set GEMINI_API_KEY in your .env file or environment to enable chat.")


# Import required libraries
try:
    import gradio as gr
except ImportError:
    print("❌ ERROR: gradio is not installed. Run 'uv pip install gradio'")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ ERROR: google-genai is not installed. Run 'uv pip install google-genai'")
    sys.exit(1)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils import clean_schema, get_gemini_model

# Globals for MCP connection and Gemini configurations
mcp_session = None
gemini_client = None
gemini_tools = []
system_instruction = ""

async def refresh_registry():
    """Fetches the list of active variables registered in the MCP session."""
    global mcp_session
    if mcp_session is None:
        return gr.Dropdown(choices=[], value=None)
    
    try:
        vars_res = await mcp_session.read_resource("registry://variables")
        vars_info = json.loads(vars_res.contents[0].text)
        choices = list(vars_info.keys())
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        print(f"Error refreshing registry: {e}")
        return gr.Dropdown(choices=[], value=None)

async def get_var_details(name):
    """Fetches the detailed coefficient table for a selected variable."""
    global mcp_session
    if not name or mcp_session is None:
        return "Select a registered variable to view its terms."
    
    try:
        var_detail_res = await mcp_session.read_resource(f"registry://variable/{name}")
        return var_detail_res.contents[0].text
    except Exception as e:
        return f"Error retrieving details for '{name}': {e}"

async def agent_chat_stream(message, history, history_state):
    """Streams the agent reasoning steps and tool execution outputs in real-time."""
    global mcp_session, gemini_client, gemini_tools, system_instruction
    
    if not message.strip():
        yield history, history_state
        return

    # Append user message to history
    history = history + [{"role": "user", "content": message}]
    yield history, history_state

    if not HAS_GEMINI_KEY:
        history = history + [
            {
                "role": "assistant",
                "content": "❌ **Error:** `GEMINI_API_KEY` is not set. Please set it in your `.env` file or environment to chat with the agent."
            }
        ]
        yield history, history_state
        return

    # Initialize Gemini Content Config with system instructions and tools
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[types.Tool(function_declarations=gemini_tools)],
        temperature=0.0
    )
    
    # Recreate the Gemini chat session loaded with the current history state
    chat = gemini_client.chats.create(model=get_gemini_model(), config=config, history=history_state)
    
    # Send message to Gemini
    response = chat.send_message(message)
    
    current_bot_message = ""
    history = history + [{"role": "assistant", "content": "Thinking..."}]
    yield history, history_state
    
    max_steps = 10
    step = 1
    
    while step <= max_steps:
        # If Gemini yields a text explanation
        if response.text:
            current_bot_message += response.text
            history[-1]["content"] = current_bot_message
            yield history, history_state
            
        # Check if Gemini requested any tool calls
        if not response.function_calls:
            break
            
        # Execute each requested function call on the Sandalwood MCP server
        tool_responses = []
        for function_call in response.function_calls:
            tool_name = function_call.name
            tool_args = dict(function_call.args) if function_call.args else {}
            
            # Log the tool invocation in the chatbot area
            status_text = f"\n\n⚙️ *Calling Tool:* `{tool_name}`\n*Arguments:* `{json.dumps(tool_args)}`...\n"
            current_bot_message += status_text
            history[-1]["content"] = current_bot_message
            yield history, history_state
            
            # Execute actual call on background MCP server
            try:
                mcp_result = await mcp_session.call_tool(tool_name, arguments=tool_args)
                result_text = mcp_result.content[0].text
                
                # Check if it was an error or success
                try:
                    res_json = json.loads(result_text)
                    if res_json.get("status") == "error":
                        current_bot_message += f"❌ *Tool Error:* {res_json.get('message')}\n"
                    else:
                        current_bot_message += f"✅ *Tool Output:* Completed successfully.\n"
                except json.JSONDecodeError:
                    current_bot_message += f"✅ *Tool Output:* Completed successfully.\n"
            except Exception as e:
                result_text = f"Error executing tool: {e}"
                current_bot_message += f"❌ *Execution Error:* {e}\n"
                
            history[-1]["content"] = current_bot_message
            yield history, history_state
            
            # Prepare result payload for Gemini
            tool_responses.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_text}
                )
            )
            
        current_bot_message += "\n🤔 *Gemini is analyzing the results...*\n"
        history[-1]["content"] = current_bot_message
        yield history, history_state
        
        # Send function response back to Gemini to continue the loop
        response = chat.send_message(tool_responses)
        step += 1
        
    # Finalize the updated conversation history
    new_history_state = chat.get_history()
    history[-1]["content"] = current_bot_message
    yield history, new_history_state

def clear_history():
    """Resets chat window and state."""
    return [], []

def build_ui():
    """Constructs the Gradio Block layout."""
    with gr.Blocks(
        title="Sandalwood TPSA Agent Dashboard"
    ) as demo:
        # State variables to hold Gemini history
        history_state = gr.State([])

        gr.Markdown("# 🌀 Sandalwood TPSA & Differential Algebra Agent Dashboard")
        gr.Markdown(
            "Talk to a Google Gemini agent that can natively execute Sandalwood Differential Algebra "
            "and Truncated Power Series Algebra operations using the Model Context Protocol (MCP)."
        )
        
        with gr.Row():
            # Left Column: Chat Window
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Agent Chat Window", height=500)
                msg_input = gr.Textbox(
                    label="Ask the agent to perform operations...",
                    placeholder="e.g., 'Initialize sandalwood with max_order 4, parse x1**2 + sin(x2), and take derivative wrt x1'",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear Chat History")
                    
            # Right Column: Stateful Registry Inspector
            with gr.Column(scale=2):
                gr.Markdown("### 🗂️ Stateful Session Registry")
                gr.Markdown("Registered Multivariate Taylor Functions and Taylor Maps in this session:")
                registry_list = gr.Dropdown(
                    label="Select Active Variable",
                    choices=[],
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Refresh Registry List", variant="secondary")
                
                gr.Markdown("### 🔍 Variable Exponent & Coefficient Inspector")
                var_details_display = gr.Code(
                    label="Active Terms (Tabular Format)",
                    language="markdown",
                    interactive=False,
                    lines=15
                )

        # Event Bindings
        # 1. Chat submit: run agent stream, then immediately update variables dropdown
        submit_btn.click(
            agent_chat_stream,
            inputs=[msg_input, chatbot, history_state],
            outputs=[chatbot, history_state]
        ).then(
            refresh_registry,
            outputs=[registry_list]
        )
        
        # 2. Clear Chat Window
        clear_btn.click(
            clear_history,
            outputs=[chatbot, history_state]
        )
        
        # 3. Manual Registry Refresh
        refresh_btn.click(
            refresh_registry,
            outputs=[registry_list]
        )
        
        # 4. Selection change on Registry list updates the variable terms view
        registry_list.change(
            get_var_details,
            inputs=[registry_list],
            outputs=[var_details_display]
        )
        
    return demo

async def main():
    global mcp_session, gemini_client, gemini_tools, system_instruction
    
    # Initialize Gemini client if API key is present
    if HAS_GEMINI_KEY:
        gemini_client = genai.Client()
    else:
        gemini_client = None
    
    # Define MCP subprocess launching parameters
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "sandalwood.agent_tools.mcp_server"]
    )
    
    print("Connecting to Sandalwood MCP Server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize the MCP session
            await session.initialize()
            mcp_session = session
            print("MCP Session initialized.")
            
            # 2. Fetch system instructions prompt from the MCP server
            expert_prompt = await mcp_session.get_prompt("sandalwood-da-expert")
            system_instruction = expert_prompt.messages[0].content.text
            
            # 3. Fetch tool schemas from the MCP server and map them to Gemini declarations if key is present
            mcp_tools_res = await mcp_session.list_tools()
            if HAS_GEMINI_KEY:
                for tool in mcp_tools_res.tools:
                    gemini_tools.append(
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=clean_schema(tool.inputSchema)
                        )
                    )
                print(f"Mapped {len(gemini_tools)} tool schemas for Gemini.")
            else:
                print("⚠️ Skipping Gemini tool mapping because GEMINI_API_KEY is not set.")
            
            # 4. Build and start Gradio UI in a non-blocking thread lock manner
            demo = build_ui()
            demo.launch(
                server_name="127.0.0.1", 
                server_port=7860, 
                prevent_thread_lock=True,
                theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
                css="footer {visibility: hidden}"
            )
            print("\n🚀 Sandalwood TPSA Agent Dashboard is running at http://127.0.0.1:7860")
            print("Press Ctrl+C in terminal to stop the application.")
            
            # Keep the async connection open until interrupted
            # Exit cleanly if running in test environment to avoid hangs in pytest/CI
            if os.environ.get("PYTEST_CURRENT_TEST"):
                print("Running under pytest. Exiting cleanly after successful startup.")
                return

            try:
                while True:
                    await asyncio.sleep(1)
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\nShutting down Gradio Dashboard...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
