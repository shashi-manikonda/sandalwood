# sandalwood Demonstrations

Welcome to the `sandalwood` demos! This directory contains a collection of Jupyter notebooks and Python scripts to help you learn and explore the capabilities of the library.

## Demo Structure

The demos are organized into the following categories:

- **`1_beginner/`**: Start here if you are new to `sandalwood`. These notebooks cover the basic concepts and functionalities.
- **`2_advanced_topics/`**: Once you are comfortable with the basics, explore these notebooks to learn about more advanced features.
- **`3_performance/`**: This section contains performance benchmarks.
- **`agent_tools_demo.py`**: A standalone Python script demonstrating the full Sandalwood Agent Tools API via the LangChain tool interface — parsing expressions, calculus, arithmetic, complex analysis, TaylorMap inversion, and validation in one self-contained script.
- **`mcp_client/`**: Programmatic demonstrations of the Sandalwood Model Context Protocol (MCP) server and client integrations:
  - `client_demo.py` — Low-level MCP protocol walkthrough (list tools, read resources, call tools directly).
  - `llm_agent_demo.py` — Full LLM agent loop with Google Gemini (or simulated mode without an API key).
- **`mcp_dashboard/`**: A Gradio web application (`app.py`) that connects a Gemini agent to the Sandalwood MCP server in an interactive side-by-side chat and registry inspector. Run with `uv run python demos/mcp_dashboard/app.py`.


## Recommended Order

We recommend going through the notebooks and scripts in the following order:

1.  **`1_beginner/0_Quick_Start.ipynb`**: A quick introduction to get you up and running.
2.  **`1_beginner/1_Basic_Functionality.ipynb`**: A more detailed look at the core features.
3.  **`2_advanced_topics/2_Advanced_Functionality.ipynb`**: Learn about substitution, composition, and more.
4.  **`2_advanced_topics/3_Taylor_Maps.ipynb`**: A guide to using the `TaylorMap` object.
5.  **`2_advanced_topics/4_Convergence_and_Accuracy.ipynb`**: A discussion on the convergence and accuracy of Taylor series.
6.  **`agent_tools_demo.py`**: Try the AI Agent Tools API directly from Python — no API key needed.
7.  **`mcp_client/client_demo.py`**: Connect to the Sandalwood MCP server and call tools programmatically.
8.  **`mcp_client/llm_agent_demo.py`**: Run a full LLM agent loop (set `GEMINI_API_KEY` for live mode, or use simulated mode).
9.  **`mcp_dashboard/app.py`**: Launch the interactive Gradio dashboard for a chat-based DA computing experience.

The script in `3_performance/` can be run at any time to see the performance benefits of `sandalwood`.
