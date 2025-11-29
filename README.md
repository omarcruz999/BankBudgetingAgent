# Bank Budgeting Agent

This repository contains a LangChain-powered budgeting assistant that reviews
CSV bank statements with Google Gemini, summarizes spending behavior, and
produces polished HTML reports. The workflow focuses on transparency: every run
shows the prompt, the tool calls LangChain executed on your behalf, and the
final narrative that you can share with stakeholders or keep for your records.

## Key Capabilities

- **Command-line flow:** Guided prompts collect the bank statement path and let
  you choose between a default budgeting brief or custom instructions.
- **Finance-focused tools:** The agent only calls two safe tools—`load_statement`
  (structured CSV parsing) and `spending_overview` (income/expense summaries)—so
  every insight is grounded in the exact transactions you provided.
- **Styled HTML deliverable:** Each run writes a standalone report to
  `outputs/` with an executive summary, category insights, and an LLM trace card
  that documents how the answer was produced.
- **Traceability and safety:** File-path validation, CSV schema checks, logging
  via `agent.log`, and path shortening inside the trace keep sensitive details
  protected while still offering reproducibility.
- **Sample datasets:** Two ready-made statements (`sample_statement.csv` and
  `sample_statement_large.csv`) help you experiment without touching real money
  data.

## LangChain Tools

- `load_statement(path: str)` – Parses and validates the CSV, returning each
  transaction with date, description, category, and amount so Gemini can reason
  over structured data instead of raw text.
- `spending_overview(path: str)` – Builds high-level metrics (total income,
  total expenses, net savings, average daily cash flow, and per-category totals)
  for the selected statement to accelerate the narrative summary.
- `compare_statements(primary_path: str, comparison_path: str)` – Surfaces deltas
  between two CSV statements, making it easy to highlight category swings and
  net changes in comparison reports.

Each tool is registered inside `build_budget_agent()` and is the only way the
LLM can touch your files, which keeps the analysis deterministic and auditable.

## Application Flow

1. **Prompting:** The CLI gathers the target CSV path (or falls back to the
  bundled sample) and asks whether you want a standard overview, a custom
  briefing, a transaction search, subscription detection, or statement
  comparison.
2. **Tool execution:** Based on that choice, the agent calls the appropriate
  LangChain tools—always starting with `load_statement`, then optionally
  `spending_overview` or `compare_statements` to obtain structured metrics.
3. **LLM reasoning:** Gemini synthesizes a narrative tailored to your prompt,
  referencing the structured tool outputs rather than the raw CSV text.
4. **Report generation:** Helper functions transform the LLM reply and trace
  into HTML, apply branded styling, and save the document inside `outputs/`
  with a timestamped filename.
5. **Review & share:** You can open the HTML directly or use `web_app.py` to
  serve the latest report via Flask for demos and feedback sessions.

## Tech Stack

- Python 3.10+
- LangChain + `ChatGoogleGenerativeAI` (Gemini 2.5 Flash)
- Flask-based web preview (optional via `web_app.py`)
- Rich HTML/CSS templates for report output

## Setup

1. **Clone/open the repo** and create a virtual environment:
   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # macOS/Linux
   source myenv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables** by adding a `.env` file next to
   `bank_agent.py`:
   ```env
   GOOGLE_API_KEY=your_gemini_key
   ```

## Running the CLI Agent

```bash
python bank_agent.py
```

1. Enter the path to your CSV statement or press Enter to use the large sample
   (`samples/sample_statement_large.csv`).
2. Choose from the interactive menu:
   - **1** – Run the curated budgeting summary prompt.
   - **2** – Provide your own instructions (e.g., "focus on cash-flow risks").
   - **3** – Switch to another statement path.
   - **4** – Exit.
3. When the analysis finishes, open the newest file inside `outputs/` (example:
   `outputs/budget_agent_result_20251122T013230Z.html`) to view:
   - A metadata table with timestamps, instructions, and selected statement.
   - A markdown-free executive summary with headings, lists, and highlights in
     blue for readability.
   - An "LLM Interaction Highlights" section showing user prompts, tool calls,
     and a reference back to the executive summary for the assistant reply.

## Optional Web Preview

Run `python web_app.py` to launch a lightweight Flask front end that serves the
latest generated report for quick sharing or demos.

## Repository Layout

```
.
├── bank_agent.py          # Agent logic, HTML builder, CLI entry point
├── web_app.py             # Optional Flask viewer for generated reports
├── samples/               # Example statements for testing
├── outputs/               # HTML reports (created at runtime)
├── templates/             # HTML template for the web app
├── requirements.txt
└── README.md
```

## Extending the Project

- Add new LangChain tools (e.g., anomaly detection, month-over-month deltas) and
  register them in `build_budget_agent()` to expand the assistant's skill set.
- Wire additional sections into `_build_html_document()` if you want charts or
  KPI tables.
- Introduce automated tests for `_parse_statement`, `_summarize_by_category`,
  and other helpers to protect against regression as reports evolve.

With these building blocks you can quickly tailor the budgeting agent for
personal finance coaching, internal audit support, or executive briefings.
