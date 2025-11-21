"""LangChain-powered budgeting agent for bank statements.

This module wires a minimal set of CSV-centric tools into a LangChain agent so
an LLM can inspect bank statements, compute spending summaries, and export
budgeting reports. It mirrors the classroom examples while fulfilling the
project requirements: LLM integration, action execution, CLI entrypoint, and
basic safety checks/logging.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
from dotenv import load_dotenv  # type: ignore[import]


ROOT_DIR = Path(__file__).parent.resolve()
LOG_PATH = ROOT_DIR / "agent.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class Transaction:
    """Typed structure representing a single bank transaction."""

    date: str
    description: str
    category: str
    amount: float

    @property
    def is_expense(self) -> bool:
        return self.amount < 0

    @property
    def is_income(self) -> bool:
        return self.amount >= 0


def _resolve_path(path_str: str) -> Path:
    """Expand, resolve, and validate a user-provided path."""

    path = Path(path_str).expanduser().resolve()
    if not path.exists():  # Avoid propagating invalid file operations.
        raise FileNotFoundError(f"Path not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, received directory: {path}")
    if path.suffix.lower() != ".csv":  # Restrict to CSV statements for safety.
        raise ValueError(f"Only CSV statements are supported, received: {path.suffix}")
    return path


def _parse_statement(path: Path) -> List[Transaction]:
    """Read a CSV bank statement into structured transactions."""

    transactions: List[Transaction] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"Date", "Description", "Category", "Amount"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Statement missing columns: {sorted(missing)}")
        for row in reader:
            try:
                amount = float(row["Amount"])  # Negative for expenses, positive for income.
            except (TypeError, ValueError) as exc:  # Surface malformed amounts.
                raise ValueError(f"Invalid amount value: {row.get('Amount')}") from exc
            transactions.append(
                Transaction(
                    date=row["Date"],
                    description=row["Description"],
                    category=row["Category"],
                    amount=amount,
                )
            )
    logger.info("Loaded %d transactions from %s", len(transactions), path)
    return transactions


def _summarize_by_category(transactions: Iterable[Transaction]) -> Dict[str, Dict[str, float]]:
    """Compute income/expense totals for each category."""

    summary: Dict[str, Dict[str, float]] = {}
    for tx in transactions:
        bucket = summary.setdefault(tx.category, {"income": 0.0, "expenses": 0.0})
        if tx.is_income:
            bucket["income"] += tx.amount
        else:
            bucket["expenses"] += abs(tx.amount)
    return summary


def _build_spending_overview(statement_path: Path) -> Dict[str, Any]:
    """Return structured spending summary metrics for the statement path."""

    transactions = _parse_statement(statement_path)
    total_income = sum(tx.amount for tx in transactions if tx.is_income)
    total_expense = sum(-tx.amount for tx in transactions if tx.is_expense)
    net = total_income - total_expense
    category_summary = _summarize_by_category(transactions)
    daily_totals: Dict[str, float] = {}
    for tx in transactions:
        daily_totals.setdefault(tx.date, 0.0)
        daily_totals[tx.date] += tx.amount
    average_daily_cash_flow = sum(daily_totals.values()) / max(len(daily_totals), 1)
    return {
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "net_savings": round(net, 2),
        "average_daily_cash_flow": round(average_daily_cash_flow, 2),
        "category_breakdown": {
            cat: {
                "income": round(values["income"], 2),
                "expenses": round(values["expenses"], 2),
            }
            for cat, values in category_summary.items()
        },
    }


@tool
def load_statement(path: str) -> str:
    """Return the parsed transactions for the CSV statement at the given path."""

    try:
        statement_path = _resolve_path(path)
        transactions = _parse_statement(statement_path)
        payload = [asdict(tx) for tx in transactions]
        logger.info("Tool load_statement returned %d transactions", len(payload))
        return json.dumps({"transactions": payload})
    except Exception as exc:  # Convert to string so the LLM sees errors clearly.
        logger.exception("load_statement failed")
        return json.dumps({"error": str(exc)})


@tool
def spending_overview(path: str) -> str:
    """Compute high-level spending metrics (totals, averages, category breakdown)."""

    try:
        statement_path = _resolve_path(path)
        summary = _build_spending_overview(statement_path)
        logger.info("Computed spending overview for %s", statement_path)
        return json.dumps(summary)
    except Exception as exc:
        logger.exception("spending_overview failed")
        return json.dumps({"error": str(exc)})


@tool
def export_budget_report(path: str, output_path: str, confirm: str = "no") -> str:
    """Export a budgeting report to disk after explicit confirmation ('yes')."""

    if confirm.lower() != "yes":  # Safety check for potentially destructive writes.
        return "Confirmation required: set 'confirm' argument to 'yes' to write the report."
    try:
        statement_path = _resolve_path(path)
        transactions = _parse_statement(statement_path)
        summary = _build_spending_overview(statement_path)
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            handle.write("Budget Report\n")
            handle.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
            handle.write(f"Source statement: {statement_path}\n\n")
            handle.write(json.dumps(summary, indent=2))
            handle.write("\n\nTop transactions:\n")
            for tx in sorted(transactions, key=lambda t: abs(t.amount), reverse=True)[:5]:
                handle.write(
                    f"- {tx.date} | {tx.description} | {tx.category} | {tx.amount:.2f}\n"
                )
        logger.info("export_budget_report wrote report to %s", output)
        return f"Budget report written to {output}"
    except Exception as exc:
        logger.exception("export_budget_report failed")
        return str(exc)


def build_budget_agent() -> Any:
    """Instantiate the LangChain agent wired with budgeting tools."""

    api_key = os.environ.get("GOOGLE_API_KEY", "USE_YOUR_API_KEY")
    if not api_key or api_key == "USE_YOUR_API_KEY":
        logger.warning("GOOGLE_API_KEY environment variable not set; agent may fail at runtime.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    system_prompt = (
        "You are a budgeting analyst. Read the user's bank statement using the "
        "available tools, explain spending habits, highlight categories exceeding the "
        "user's thresholds, and suggest actionable strategies to save more. Always "
        "deliver a detailed, narrative report without asking for additional "
        "confirmation steps. If exporting is required, call export_budget_report "
        "directly with a suitable default path."
    )
    return create_agent(
        model=llm,
        tools=[load_statement, spending_overview, export_budget_report],
        system_prompt=system_prompt,
    )


def invoke_agent(agent: Any, user_input: str) -> Dict[str, Any]:
    """Run the LangChain agent with a plain-text user input."""

    return agent.invoke({"messages": [{"role": "user", "content": user_input}]})


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def extract_final_reply(result: Dict[str, Any]) -> str:
    """Return the assistant's final reply string from the agent invocation result."""

    messages = result.get("messages", [])
    if messages and isinstance(messages[-1], BaseMessage):
        return _coerce_message_text(messages[-1].content)
    return ""


def _format_args(args: Dict[str, Any]) -> str:
    if not args:
        return "(no arguments)"
    try:
        return json.dumps(args, indent=2)
    except (TypeError, ValueError):
        return str(args)


def _summarize_args(args: Dict[str, Any]) -> str:
    if not args:
        return "(no arguments)"
    pairs: List[str] = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value, ensure_ascii=False)
        else:
            serialized = str(value)
        pairs.append(f"{key}={serialized}")
    return ", ".join(pairs)


def _summarize_tool_response(name: str, content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    if isinstance(content, dict):
        parsed = content
    else:
        raw_text = str(content)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
    if name == "load_statement":
        transactions = parsed.get("transactions", [])
        count = len(transactions)
        preview_parts: List[str] = []
        for tx in transactions[:3]:
            date = tx.get("date", "?")
            desc = tx.get("description", "Unknown")
            amount = tx.get("amount", 0)
            preview_parts.append(f"{date} – {desc} ({amount})")
        preview = ", ".join(preview_parts)
        if count > 3:
            preview += f", … {count - 3} more"
        return f"Loaded {count} transactions" + (f": {preview}" if preview else ".")
    if name == "spending_overview":
        income = parsed.get("total_income")
        expenses = parsed.get("total_expense")
        net = parsed.get("net_savings")
        avg_cash = parsed.get("average_daily_cash_flow")
        return (
            "Spending summary → "
            f"Income ${income}, Expenses ${expenses}, Net ${net}, Average daily cash flow ${avg_cash}"
        )
    if isinstance(parsed, dict) and parsed.get("error"):
        return f"Error: {parsed['error']}"
    return json.dumps(parsed, indent=2)


def format_tool_trace(messages: List[BaseMessage]) -> str:
    """Produce a concise, human-readable trace of tool usage."""

    events: List[Dict[str, Any]] = []
    tool_events_by_id: Dict[str, Dict[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _coerce_message_text(msg.content).strip()
            event = {"label": "User instructions", "details": []}
            _add_trace_sentence(event, text, 260)
            if event["details"]:
                events.append(event)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for call in msg.tool_calls:
                    args = call.get("args", {})
                    label = f"Tool call: {call['name']}"
                    event = {"label": label, "details": []}
                    _add_trace_sentence(event, f"Args -> {_summarize_args(args)}")
                    events.append(event)
                    call_id = call.get("id")
                    if call_id:
                        tool_events_by_id[str(call_id)] = event
            elif msg.content:
                event = {"label": "Assistant response", "details": []}
                _add_trace_sentence(event, _coerce_message_text(msg.content).strip())
                if event["details"]:
                    events.append(event)
        elif isinstance(msg, ToolMessage):
            summary = _summarize_tool_response(msg.name, msg.content)
            if msg.tool_call_id and msg.tool_call_id in tool_events_by_id:
                _add_trace_sentence(tool_events_by_id[msg.tool_call_id], f"Result -> {summary}")
            else:
                event = {"label": f"Tool result: {msg.name}", "details": []}
                _add_trace_sentence(event, summary)
                if event["details"]:
                    events.append(event)

    lines: List[str] = []
    for event in events:
        details = [detail for detail in event.get("details", []) if detail]
        if not details:
            lines.append(f"- **{event['label']}**")
            continue
        first, *rest = details
        lines.append(f"- **{event['label']}:** {first}")
        for detail in rest:
            lines.append(f"  {detail}")
    return "\n".join(lines)


def run_budget_review(statement_path: str, additional_prompt: str) -> Dict[str, Any]:
    """Convenience wrapper that builds the agent, runs it, and logs the session."""

    agent = build_budget_agent()
    combined_prompt = (
        "You are auditing the bank statement located at: "
        f"{statement_path}. "
        "Follow best budgeting practices and present a thorough report covering "
        "high-level metrics, category-level insights, notable transactions, and "
        "tailored budgeting actions. "
        f"Additional instructions: {additional_prompt}"
    )
    result = invoke_agent(agent, combined_prompt)
    logger.info("Agent invocation completed for %s", statement_path)
    return result


def _prompt_statement_path(default_statement: Path) -> str:
    """Interactively gather and validate the statement path from the user."""

    while True:
        user_input = input(
            "Enter the path to your CSV bank statement "
            f"(press Enter for sample: {default_statement}): "
        ).strip()
        candidate = user_input or str(default_statement)
        try:
            resolved = _resolve_path(candidate)
            return str(resolved)
        except Exception as exc:
            print(f"Invalid statement path: {exc}\nPlease try again.\n")


_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_PATTERN = re.compile(r"_(.+?)_")
_CODE_PATTERN = re.compile(r"`([^`]+)`")
_WORKSPACE_REPLACEMENTS = [
    str(ROOT_DIR),
    str(ROOT_DIR).replace("\\", "\\\\"),
    ROOT_DIR.as_posix(),
]


def _to_workspace_relative(path_str: str) -> str:
    if not path_str:
        return path_str
    candidate = Path(path_str.strip().strip('"'))
    try:
        resolved = candidate.expanduser().resolve()
    except (OSError, RuntimeError):
        return path_str
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path_str


def _shorten_paths_in_text(text: str) -> str:
    if not text:
        return text
    result = text
    for marker in _WORKSPACE_REPLACEMENTS:
        result = result.replace(f"{marker}\\", "")
        result = result.replace(f"{marker}/", "")
        result = result.replace(marker, "")
    result = result.replace(".\\", "")
    result = result.replace("./", "")
    result = result.replace("\\\\", "\\")
    result = result.replace("\\", "/")
    while "//" in result:
        result = result.replace("//", "/")
    return result


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_trace_text(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    cleaned = _collapse_whitespace(_shorten_paths_in_text(text))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _add_trace_sentence(event: Dict[str, Any], text: str, limit: int = 200) -> None:
    if not text:
        return
    cleaned = _clean_trace_text(text, limit)
    if not cleaned:
        return
    details = event.setdefault("details", [])
    if len(details) >= 2:
        return
    details.append(cleaned)


def _apply_basic_formatting(segment: str) -> str:
    escaped = escape(segment)
    escaped = _BOLD_PATTERN.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC_PATTERN.sub(r"<em>\1</em>", escaped)
    return escaped.replace("\n", "<br />")


def _format_inline_html(text: str) -> str:
    """Escape HTML, support inline formatting, and highlight code spans."""

    if not text:
        return ""

    html_parts: List[str] = []
    last = 0
    for match in _CODE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last:
            html_parts.append(_apply_basic_formatting(text[last:start]))
        html_parts.append(f"<code>{escape(match.group(1))}</code>")
        last = end
    if last < len(text):
        html_parts.append(_apply_basic_formatting(text[last:]))
    return "".join(html_parts)


def _format_summary_html(text: str) -> str:
    """Convert the agent narrative into structured, markdown-free HTML."""

    cleaned = text.strip()
    if not cleaned:
        return "<p><em>No response returned.</em></p>"

    html_parts: List[str] = []
    list_type: str | None = None

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            html_parts.append(f"</{list_type}>")
            list_type = None

    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            close_list()
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            close_list()
            level = len(heading_match.group(1))
            content = heading_match.group(2)
            tag = "h3" if level <= 3 else "h4"
            html_parts.append(f"<{tag}>{_format_inline_html(content)}</{tag}>")
            continue
        if stripped.startswith(('- ', '* ')):
            if list_type != "ul":
                close_list()
                html_parts.append("<ul class=\"summary-list\">")
                list_type = "ul"
            html_parts.append(f"<li>{_format_inline_html(stripped[2:])}</li>")
            continue
        if re.match(r"^\d+\.\s+", stripped):
            if list_type != "ol":
                close_list()
                html_parts.append("<ol class=\"summary-list\">")
                list_type = "ol"
            content = re.sub(r"^\d+\.\s+", "", stripped)
            html_parts.append(f"<li>{_format_inline_html(content)}</li>")
            continue
        close_list()
        html_parts.append(f"<p>{_format_inline_html(stripped)}</p>")

    close_list()
    return "".join(html_parts)


def _format_trace_html(trace: str) -> str:
    """Render the textual trace as card-like elements."""

    trimmed = trace.strip()
    if not trimmed:
        return "<p>No tool interactions were recorded.</p>"

    events: List[Dict[str, List[str]]] = []
    current: Dict[str, List[str]] | None = None
    for line in trimmed.splitlines():
        if line.startswith("- **"):
            if current:
                events.append(current)
            body = line[4:]
            if "**" in body:
                label, remainder = body.split("**", 1)
            else:
                label, remainder = body, ""
            detail = remainder.lstrip(": ").strip()
            current = {"title": label.strip(), "details": []}
            if detail:
                current["details"].append(detail)
        elif line.startswith("  ") and current:
            current["details"].append(line.strip())
    if current:
        events.append(current)
    if not events:
        return f"<pre class=\"trace-block\">{escape(trimmed)}</pre>"

    html_parts = ["<div class=\"trace-grid\">"]
    for event in events:
        html_parts.append("<article class=\"trace-card\">")
        html_parts.append(
            f"<div class=\"trace-card__title\">{_format_inline_html(event['title'])}</div>"
        )
        if event["details"]:
            html_parts.append("<ul class=\"trace-card__details\">")
            for detail in event["details"]:
                html_parts.append(f"<li>{_format_inline_html(detail)}</li>")
            html_parts.append("</ul>")
        html_parts.append("</article>")
    html_parts.append("</div>")
    return "".join(html_parts)


def _format_human_timestamp(moment: datetime) -> str:
    aware = moment.astimezone(timezone.utc)
    return aware.strftime("%B %d, %Y at %I:%M %p %Z")


def _extract_export_path(messages: List[BaseMessage]) -> Optional[str]:
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "export_budget_report":
            text = _coerce_message_text(msg.content).strip()
            match = re.search(r"Budget report written to (.+)", text)
            if match:
                return _to_workspace_relative(match.group(1).strip())
            if text:
                return _to_workspace_relative(text)
    return None


def _build_html_document(
    generated_at: datetime,
    statement_path: str,
    instructions: str,
    summary_html: str,
    trace_html: str,
    export_path: Optional[str] = None,
) -> str:
    """Construct a polished standalone HTML document for the report."""

    rows = [
        f"<tr><th>Generated</th><td>{escape(_format_human_timestamp(generated_at))}</td></tr>",
        f"<tr><th>Statement</th><td>{escape(Path(statement_path).name)}</td></tr>",
        f"<tr><th>Instructions</th><td>{escape(instructions)}</td></tr>",
    ]
    if export_path:
        relative = _to_workspace_relative(export_path)
        rows.append(
            f"<tr><th>Exported PDF</th><td><code>{escape(relative)}</code></td></tr>"
        )
    metadata_rows = "".join(rows)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Budget Analysis Report</title>
  <style>
    :root {{
      font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      --accent: #0f62fe;
      --accent-soft: #e0ecff;
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475467;
      --border: #d7dce4;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      background: linear-gradient(135deg, var(--accent), #54b7ff);
      color: white;
      padding: 2.5rem 3rem 3.5rem;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
    }}
    header h1 {{
      margin: 0 0 0.4rem;
      font-size: 2.3rem;
    }}
    header p {{
      margin: 0;
      font-size: 1.05rem;
      color: rgba(255,255,255,0.9);
    }}
    main {{
      max-width: 960px;
      margin: -2rem auto 3rem;
      padding: 0 1.5rem;
    }}
    section {{
      background: var(--card);
      border-radius: 20px;
      padding: 2rem;
      margin-bottom: 1.75rem;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      border: 1px solid var(--border);
    }}
    section h2 {{
      margin-top: 0;
      letter-spacing: 0.01em;
    }}
    table.meta {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.98rem;
    }}
    table.meta th {{
      text-align: left;
      width: 28%;
      color: var(--muted);
      font-weight: 600;
      padding: 0.35rem 0;
    }}
    table.meta td {{
      padding: 0.35rem 0;
    }}
    .summary h3 {{
      margin-bottom: 0.5rem;
      color: var(--accent);
      font-size: 1.15rem;
    }}
    .summary p {{
      line-height: 1.7;
      margin-bottom: 1rem;
    }}
    .summary-list {{
      margin: 0 0 1.1rem 1.5rem;
      padding: 0;
    }}
        .trace-grid {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}
    .trace-card {{
            display: flex;
            flex-direction: row;
            gap: 1rem;
            align-items: flex-start;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.25rem;
      background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    .trace-card__title {{
      font-weight: 600;
            margin-bottom: 0;
            color: var(--accent);
            min-width: 210px;
    }}
    .trace-card__details {{
            flex: 1;
            margin: 0;
            padding-left: 0;
      color: var(--muted);
            list-style: none;
    }}
        .trace-card__details li {{
                        margin-bottom: 0.35rem;
            word-break: break-word;
            overflow-wrap: anywhere;
        }}
        code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
            background: var(--accent-soft);
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.95rem;
        }}
    @media (max-width: 640px) {{
      header {{ padding: 2rem 1.5rem 3rem; }}
      main {{ padding: 0 1rem; }}
      section {{ padding: 1.5rem; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Budget Analysis Report</h1>
    <p>Prepared by the Bank Statement Budgeting Agent</p>
  </header>
  <main>
    <section>
      <h2>Session Overview</h2>
      <table class=\"meta\">
        {metadata_rows}
      </table>
    </section>
    <section class=\"summary\">
      <h2>Executive Summary</h2>
      {summary_html}
    </section>
    <section>
      <h2>LLM Interaction Highlights</h2>
      {trace_html}
    </section>
  </main>
</body>
</html>
"""


def _generate_html_report(
    statement_path: str, instructions: str, outputs_dir: Path
) -> Path:
    """Run the agent and write its response/trace to a styled HTML report."""

    result = run_budget_review(statement_path, instructions)
    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_file = outputs_dir / f"budget_agent_result_{timestamp}.html"
    final_reply = extract_final_reply(result).strip()
    trace_block = format_tool_trace(result.get("messages", []))
    summary_html = _format_summary_html(_shorten_paths_in_text(final_reply))
    trace_html = _format_trace_html(trace_block)
    export_path = _extract_export_path(result.get("messages", []))
    document = _build_html_document(
        generated_at=generated_at,
        statement_path=statement_path,
        instructions=instructions,
        summary_html=summary_html,
        trace_html=trace_html,
        export_path=export_path,
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(document)
    return output_file


def _run_analysis_flow(statement_path: str, instructions: str, outputs_dir: Path) -> None:
    """Execute an agent run with helpful CLI messaging."""

    print("\nWorking on your budgeting analysis...\n")
    try:
        report_path = _generate_html_report(statement_path, instructions, outputs_dir)
        print(f"Analysis complete! Results saved to: {report_path}\n")
    except Exception as exc:
        logger.exception("Budget analysis run failed")
        print(f"An error occurred while running the agent: {exc}\n")


def main() -> None:
    """Interactive CLI entry point with multiple analysis options."""

    workspace = Path(__file__).parent
    default_statement = workspace / "samples" / "sample_statement_large.csv"
    outputs_dir = workspace / "outputs"
    print("Bank Statement Budgeting Agent")
    print("--------------------------------------------------")
    print(f"Sample statements available under: {default_statement.parent}")
    print(f" - Default sample: {default_statement}")
    statement_path = _prompt_statement_path(default_statement)
    example_prompt = (
        "Summarize overall income vs expenses, highlight any category exceeding $500, "
        "call out the five largest expenses, describe spending trends you notice, "
        "and provide three actionable budgeting recommendations. Format amounts in USD "
        "and deliver a detailed narrative without asking follow-up questions."
    )
    while True:
        print("What would you like to do next?")
        print("  1) Run quick summary (default prompt)")
        print("  2) Run custom analysis (you supply instructions)")
        print("  3) Change statement path")
        print("  4) Exit")
        choice = input("Select an option (1-4): ").strip().lower()
        if choice in {"1", "a"}:
            _run_analysis_flow(statement_path, example_prompt, outputs_dir)
        elif choice in {"2", "b"}:
            custom_prompt = input(
                "Enter the instructions you want the agent to follow: "
            ).strip()
            if not custom_prompt:
                print("Custom instructions cannot be empty. Please try again.\n")
                continue
            _run_analysis_flow(statement_path, custom_prompt, outputs_dir)
        elif choice in {"3", "c"}:
            statement_path = _prompt_statement_path(default_statement)
        elif choice in {"4", "q", "quit", "exit"}:
            print("Goodbye!")
            break
        else:
            print("Unrecognized option. Please choose 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()
