import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utils import clean_schema, get_gemini_model

# Try to import modern Google GenAI SDK. If not available, we run in simulated mode.
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# We check if GEMINI_API_KEY environment variable is set
LIVE_MODE = HAS_GEMINI and os.environ.get("GEMINI_API_KEY") is not None

async def run_mcp_llm_demo():
    print("=================================================================")
    print("===            Sandalwood MCP + Gemini Agent Demo             ===")
    print("=================================================================\n")

    if LIVE_MODE:
        print("🟢 Running in LIVE MODE using the Google Gemini API.")
    else:
        print("🟡 Running in SIMULATED MODE (Mocking LLM decisions, calling real MCP tools).")
        print("   To run in LIVE MODE:")
        print("   1. Install google-genai: pip install google-genai")
        print("   2. Set environment variable: export GEMINI_API_KEY='your-key-here'\n")

    # 1. Define how to start the Sandalwood MCP Server
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "sandalwood.agent_tools.mcp_server"]
    )
    
    print("Step 1: Connecting to Sandalwood MCP Server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            # Initialize the MCP connection
            await mcp_session.initialize()
            print("MCP Session initialized successfully.\n")

            # 2. Fetch the tools from the Sandalwood MCP Server
            print("Step 2: Listing tools from MCP Server...")
            mcp_tools_res = await mcp_session.list_tools()
            mcp_tools = mcp_tools_res.tools
            print(f"Discovered {len(mcp_tools)} tools from MCP server.\n")

            # 3. Format the tools to the schema expected by Gemini (FunctionDeclarations)
            print("Step 3: Mapping MCP tools to Gemini Tool Definitions...")
            gemini_tools = []
            if LIVE_MODE:
                for tool in mcp_tools:
                    gemini_tools.append(
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=clean_schema(tool.inputSchema)
                        )
                    )
            print(f"Mapped {len(mcp_tools)} tool schemas for Gemini.\n")

            # 4. User Prompt
            user_prompt = (
                "Differentiate the expression 'x1**3 + x2' with respect to x1, "
                "then evaluate that derivative at the point [2.0, 1.0]. "
                "Initialize sandalwood with max_order=4 and max_dimension=2."
            )
            print(f"User Prompt: '{user_prompt}'\n")

            # 5. Agent loop
            if LIVE_MODE:
                await run_live_agent(mcp_session, gemini_tools, user_prompt)
            else:
                await run_simulated_agent(mcp_session, user_prompt)

async def run_live_agent(mcp_session, gemini_tools, user_prompt):
    """Executes the agent loop making real calls to the Google Gemini API."""
    client = genai.Client()
    
    # We fetch the system instructions prompt from MCP server
    expert_prompt = await mcp_session.get_prompt("sandalwood-da-expert")
    system_instruction = expert_prompt.messages[0].content.text
    
    # Configure the chat session with instructions and tools
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[types.Tool(function_declarations=gemini_tools)],
        temperature=0.0
    )
    
    print("Initializing Gemini Chat Session...")
    chat = client.chats.create(model=get_gemini_model(), config=config)
    
    print("Sending conversation to Gemini...")
    response = chat.send_message(user_prompt)
    
    max_steps = 10
    step = 1
    
    while step <= max_steps:
        print(f"--- Agent Step {step} ---")
        
        # Print Gemini's textual response if any
        if response.text:
            print(f"\nGemini: {response.text}\n")
            
        # If Gemini didn't ask to run any function calls, we are finished!
        if not response.function_calls:
            print("Gemini has completed the task.")
            break
            
        # Execute each function call requested by Gemini
        tool_responses = []
        for function_call in response.function_calls:
            tool_name = function_call.name
            tool_args = dict(function_call.args) if function_call.args else {}
            
            print(f"Tool Call Requested: {tool_name} with args: {tool_args}")
            
            # Execute the tool on the Sandalwood MCP server
            mcp_result = await mcp_session.call_tool(tool_name, arguments=tool_args)
            result_text = mcp_result.content[0].text
            print(f"MCP Server Output: {result_text}\n")
            
            # Pack the tool result as expected by Gemini Chat history
            tool_responses.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_text}
                )
            )
            
        # Send the tool outputs back to Gemini to continue the reasoning loop
        print("Sending tool outputs back to Gemini...")
        response = chat.send_message(tool_responses)
        step += 1

async def run_simulated_agent(mcp_session, user_prompt):
    """Simulates the LLM reasoning steps, invoking actual MCP server tools."""
    print("--- Simulating Agent Reasoning Steps ---")
    
    # Step A: Initialize Sandalwood
    print("Simulating Step A: LLM decides to initialize sandalwood")
    print("Tool Call: 'initialize_sandalwood' with max_order=4, max_dimension=2")
    init_res = await mcp_session.call_tool(
        "initialize_sandalwood",
        arguments={"max_order": 4, "max_dimension": 2, "implementation": "python"}
    )
    print(f"MCP Output: {init_res.content[0].text}\n")

    # Step B: Parse Expression
    print("Simulating Step B: LLM decides to parse the mathematical formula")
    print("Tool Call: 'parse_expression_to_mtf' with expression='x1**3 + x2', name='f1'")
    parse_res = await mcp_session.call_tool(
        "parse_expression_to_mtf",
        arguments={
            "expression": "x1**3 + x2",
            "dimension": 2,
            "max_order": 4,
            "name": "f1"
        }
    )
    print(f"MCP Output: {parse_res.content[0].text}\n")

    # Step C: Compute Derivative
    print("Simulating Step C: LLM decides to compute the partial derivative df1/dx1")
    print("Tool Call: 'compute_partial_derivative' with mtf_ref='f1', var_index=1, name='df1'")
    deriv_res = await mcp_session.call_tool(
        "compute_partial_derivative",
        arguments={
            "mtf_ref": "f1",
            "var_index": 1,
            "name": "df1"
        }
    )
    print(f"MCP Output: {deriv_res.content[0].text}\n")

    # Step D: Evaluate derivative
    print("Simulating Step D: LLM decides to evaluate df1/dx1 at [2.0, 1.0]")
    print("Tool Call: 'evaluate_mtf' with mtf_ref='df1', point=[2.0, 1.0]")
    eval_res = await mcp_session.call_tool(
        "evaluate_mtf",
        arguments={
            "mtf_ref": "df1",
            "point": [2.0, 1.0]
        }
    )
    print(f"MCP Output: {eval_res.content[0].text}\n")

    # Step E: LLM final answer synthesis
    print("Simulating Step E: LLM synthesizes final answer based on tool outputs:")
    res_data = json.loads(eval_res.content[0].text)
    val = res_data['data']['result']
    print(f"\n[Simulated Gemini]:\n"
          f"To differentiate the expression 'x1**3 + x2' with respect to x1 and evaluate it at [2.0, 1.0]:\n"
          f"1. First, we initialize the Sandalwood environment.\n"
          f"2. We compile the expression to the registry as 'f1'.\n"
          f"3. Differentiating 'f1' analytically gives 3 * x1**2, registered as 'df1'.\n"
          f"4. Evaluating 'df1' at the point [2.0, 1.0] gives: 3 * (2.0)**2 = 12.0.\n\n"
          f"The final calculated value is {val}.\n")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(run_mcp_llm_demo())
