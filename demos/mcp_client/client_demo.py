import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_mcp_client_demo():
    print("=================================================================")
    # Set up server parameters to launch the Sandalwood MCP server in a subprocess
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "sandalwood.agent_tools.mcp_server"]
    )
    
    print("1. Connecting to Sandalwood MCP Server via Stdio...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize the MCP session
            await session.initialize()
            print("Session initialized successfully.\n")

            # 2. List prompts and fetch the expert prompt instructions
            print("2. Fetching 'sandalwood-da-expert' Prompt...")
            prompts = await session.list_prompts()
            print(f"Available Prompts: {[p.name for p in prompts.prompts]}")
            
            expert_prompt = await session.get_prompt("sandalwood-da-expert")
            print("Expert Prompt Content Snippet:")
            # Print the first few lines of the system prompt instructions
            snippet = "\n".join(expert_prompt.messages[0].content.text.split("\n")[:6])
            print(f"{snippet}\n...\n")

            # 3. List tools
            print("3. Fetching Available Tools...")
            tools = await session.list_tools()
            print(f"Total Tools Registered: {len(tools.tools)}")
            print(f"Some available tools: {[t.name for t in tools.tools[:6]]}...\n")

            # 4. Call Tool: initialize_sandalwood
            print("4. Invoking 'initialize_sandalwood' tool...")
            init_res = await session.call_tool(
                "initialize_sandalwood",
                arguments={
                    "max_order": 3,
                    "max_dimension": 2,
                    "implementation": "python"
                }
            )
            # The result content is a text block containing JSON string
            res_dict = json.loads(init_res.content[0].text)
            print(f"Response: {res_dict['data']['result']}\n")

            # 5. Call Tool: parse_expression_to_mtf
            print("5. Invoking 'parse_expression_to_mtf' tool...")
            parse_res = await session.call_tool(
                "parse_expression_to_mtf",
                arguments={
                    "expression": "x1**2 + sin(x2)",
                    "dimension": 2,
                    "max_order": 3,
                    "name": "my_function"
                }
            )
            parse_dict = json.loads(parse_res.content[0].text)
            ref_name = parse_dict['data']['result']['ref']
            print(f"Registered ref: '{ref_name}'")
            print(f"MTF Terms:\n{parse_dict['data']['result']['info']}\n")

            # 6. Read Resource: registry://variables
            print("6. Reading Resource 'registry://variables'...")
            vars_res = await session.read_resource("registry://variables")
            vars_info = json.loads(vars_res.contents[0].text)
            print(f"Variables registry content: {json.dumps(vars_info, indent=2)}\n")

            # 7. Read Resource: registry://variable/{name}
            print(f"7. Reading Resource 'registry://variable/{ref_name}'...")
            var_detail_res = await session.read_resource(f"registry://variable/{ref_name}")
            print(f"Tabular Representation of '{ref_name}':")
            print(var_detail_res.contents[0].text)
            print()

            # 8. Call Tool: evaluate_mtf
            print("8. Evaluating 'my_function' at [2.0, 0.0]...")
            eval_res = await session.call_tool(
                "evaluate_mtf",
                arguments={
                    "mtf_ref": "my_function",
                    "point": [2.0, 0.0]
                }
            )
            eval_dict = json.loads(eval_res.content[0].text)
            val = eval_dict['data']['result']
            print(f"Result value: {val}  (expected: 2.0**2 + sin(0.0) = 4.0)")
            print("\nMCP Sandalwood Demo Completed Successfully!")
            print("=================================================================")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(run_mcp_client_demo())
