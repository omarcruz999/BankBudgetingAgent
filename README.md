# Budgeting Agent with LangChain and Gemini

This project implements a command-line AI agent that reads a CSV bank statement,
interprets the data using a Large Language Model (LLM), and produces a detailed
budget analysis report. The agent mirrors a classroom example built with
LangChain, but swaps in finance-focused tools that summarize transactions,
highlight spending patterns, and generate actionable recommendations.

## Features

- **LLM integration** with Google Gemini (via LangChain) for natural-language
  reasoning about financial data.
- **Structured tools** that load statements, compute spending summaries, and
  export human-readable budgeting reports.
- **Safety checks** on file paths, CSV schema validation, and logging of every
  agent interaction for traceability.
- **CLI workflow** that accepts a user-provided statement path (or falls back to
  a sample) and writes Markdown reports under `outputs/`.
- **Sample datasets** in `samples/` including a compact statement
  (`sample_statement.csv`) and an extended 50-transaction statement
  (`sample_statement_large.csv`).

## Prerequisites

- Python 3.10+
- A Google Gemini API key with access to the free tier
- pip (or another Python package manager)

## Setup

1. **Clone or open the repo** in your workspace.
2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**:
   - Create a `.env` file in the project root (same folder as `bank_agent.py`).
   - Add your Gemini API key:
     ```env
     GOOGLE_API_KEY=your_api_key_here
     ```

If `requirements.txt` is not yet present, you can install packages manually:
```bash
pip install langchain langchain-google-genai python-dotenv
```

## Running the Agent

1. Ensure your virtual environment is active and `.env` is configured.
2. Run the CLI script:
   ```bash
   python bank_agent.py
   ```
3. When prompted, enter the path to your CSV bank statement. Press Enter to use
the default sample at `samples/sample_statement_large.csv`.
4. The agent will display a "Working..." message, then write a Markdown report
   to the `outputs/` folder (e.g., `outputs/budget_agent_result_<timestamp>.md`).
5. Open the generated report in your editor or viewer of choice to read the
   narrative summary and LLM interaction trace.

## Project Structure

```
.
├── bank_agent.py              # Main agent implementation and CLI entry point
├── samples/
│   ├── sample_statement.csv
│   └── sample_statement_large.csv
├── outputs/                   # Generated Markdown reports (created at runtime)
└── README.md
```

## Notes & Next Steps

- The agent relies on LangChain's experimental Gemini integration; ensure the
  package versions in your environment support `ChatGoogleGenerativeAI`.
- Logs for each run are stored in `agent.log` for auditability.
- You can extend the toolset with additional actions (e.g., budget target
  comparisons, anomaly detection, automatic exports) by adding more `@tool`
  functions and wiring them into `build_budget_agent()`.
- Consider adding automated tests around CSV parsing and summary math to catch
  regressions as the project grows.
