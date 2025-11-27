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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

CLI_DIVIDER = "-" * 50
SUBSCRIPTION_FOOTNOTE = (
    "Estimated single instance because only one transaction appears in the "
    "statement period, but the description or category suggests a recurring charge."
)
SUBSCRIPTION_DEFAULT_INSTRUCTIONS = (
    "Review the entire statement for repeating charges or services. Group likely "
    "subscriptions by merchant, note frequency (weekly/biweekly/monthly), estimate "
    "total monthly cost, and highlight any unusual spikes or cancellations."
)

COMPARISON_DEFAULT_INSTRUCTIONS = (
    "Focus on the five largest expense swings, note any new or discontinued "
    "subscriptions, and summarize the total change in income, expenses, and net "
    "savings between the two periods."
)

DEFAULT_SUMMARY_PROMPT = (
    "Summarize overall income vs expenses, highlight any category exceeding $500, "
    "call out the five largest expenses, describe spending trends you notice, "
    "and provide three actionable budgeting recommendations. Format amounts in USD "
    "and deliver a detailed narrative without asking follow-up questions."
)


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
def compare_statements(primary_path: str, comparison_path: str) -> str:
    """Compare two CSV statements and return category-level deltas."""

    try:
        primary_statement = _resolve_path(primary_path)
        comparison_statement = _resolve_path(comparison_path)
        primary_transactions = _parse_statement(primary_statement)
        comparison_transactions = _parse_statement(comparison_statement)
        primary_summary = _summarize_by_category(primary_transactions)
        comparison_summary = _summarize_by_category(comparison_transactions)

        def _totals(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
            income_total = sum(values["income"] for values in summary.values())
            expense_total = sum(values["expenses"] for values in summary.values())
            return {
                "income": round(income_total, 2),
                "expenses": round(expense_total, 2),
                "net": round(income_total - expense_total, 2),
            }

        primary_totals = _totals(primary_summary)
        comparison_totals = _totals(comparison_summary)

        category_deltas: List[Dict[str, float | str]] = []
        for category in sorted(set(primary_summary) | set(comparison_summary)):
            baseline = primary_summary.get(category, {"income": 0.0, "expenses": 0.0})
            updated = comparison_summary.get(category, {"income": 0.0, "expenses": 0.0})
            category_deltas.append(
                {
                    "category": category,
                    "baseline_income": round(baseline["income"], 2),
                    "baseline_expenses": round(baseline["expenses"], 2),
                    "comparison_income": round(updated["income"], 2),
                    "comparison_expenses": round(updated["expenses"], 2),
                    "income_change": round(updated["income"] - baseline["income"], 2),
                    "expense_change": round(updated["expenses"] - baseline["expenses"], 2),
                }
            )

        response = {
            "primary": {
                "path": str(primary_statement),
                "totals": primary_totals,
            },
            "comparison": {
                "path": str(comparison_statement),
                "totals": comparison_totals,
            },
            "difference": {
                "income_change": round(
                    comparison_totals["income"] - primary_totals["income"], 2
                ),
                "expense_change": round(
                    comparison_totals["expenses"] - primary_totals["expenses"], 2
                ),
                "net_change": round(
                    comparison_totals["net"] - primary_totals["net"], 2
                ),
            },
            "category_deltas": category_deltas,
        }

        logger.info(
            "compare_statements analyzed %s vs %s", primary_statement, comparison_statement
        )
        return json.dumps(response)
    except Exception as exc:
        logger.exception("compare_statements failed")
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


def build_budget_agent(system_prompt: Optional[str] = None) -> Any:
    """Instantiate the LangChain agent wired with budgeting tools."""

    api_key = os.environ.get("GOOGLE_API_KEY", "USE_YOUR_API_KEY")
    if not api_key or api_key == "USE_YOUR_API_KEY":
        logger.warning("GOOGLE_API_KEY environment variable not set; agent may fail at runtime.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    default_prompt = (
        "You are a budgeting analyst. Read the user's bank statement using the "
        "available tools, explain spending habits, highlight categories exceeding the "
        "user's thresholds, and suggest actionable strategies to save more. Always "
        "deliver a detailed, narrative report without asking for additional "
        "confirmation steps."
    )
    return create_agent(
        model=llm,
        tools=[load_statement, spending_overview, compare_statements],
        system_prompt=system_prompt or default_prompt,
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
        if transactions:
            first = transactions[0]
            date = first.get("date", "?")
            desc = first.get("description", "Unknown")
            amount = first.get("amount", 0)
            return f"Loaded {count} transactions (first: {date} {desc} {amount})"
        return f"Loaded {count} transactions."
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
            text = _first_sentences(_shorten_paths_in_text(_coerce_message_text(msg.content)), 2)
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
                _add_trace_sentence(event, "See above.", 260)
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


def run_transaction_query(statement_path: str, query: str) -> Dict[str, Any]:
    """Run a targeted transaction search using the budgeting tools."""

    search_system_prompt = (
        "You specialize in searching bank transactions. Use the available tools to "
        "load the user's statement and answer focused questions about specific "
        "transactions, amounts, and categories. Respond concisely with the relevant "
        "transactions formatted as bullet points including date, description, "
        "category, and amount in USD. If nothing matches, state that no transactions "
        "were found."
    )
    agent = build_budget_agent(system_prompt=search_system_prompt)
    combined_prompt = (
        "The bank statement to inspect is located at: "
        f"{statement_path}. "
        "Use the load_statement tool with this path to fetch transactions before "
        "answering the user's request. Focus on the specific filters or keywords the "
        "user mentions and do not provide unrelated commentary. "
        f"User request: {query}"
    )
    result = invoke_agent(agent, combined_prompt)
    logger.info("Transaction query completed for %s", statement_path)
    return result


def run_subscription_detection(statement_path: str, instructions: Optional[str] = None) -> Dict[str, Any]:
    """Identify recurring or subscription-like transactions."""

    subscription_system_prompt = (
        "You specialize in spotting recurring subscription payments. Use the "
        "available tools to load the statement, detect merchants with regular "
        "charges, estimate cadence and monthly cost, and flag items that look "
        "uncertain. Return concise bullet points including vendor, frequency, and "
        "average monthly spend."
    )
    agent = build_budget_agent(system_prompt=subscription_system_prompt)
    combined_prompt = (
        "The bank statement to inspect is located at: "
        f"{statement_path}. "
        "Use the load_statement tool with this path to gather transactions before "
        "summarizing. Focus on recurring charges such as streaming services, "
        "utilities, software, memberships, or other subscription-like payments. "
        f"Additional focus: {instructions or SUBSCRIPTION_DEFAULT_INSTRUCTIONS}"
    )
    result = invoke_agent(agent, combined_prompt)
    logger.info("Subscription detection completed for %s", statement_path)
    return result


def run_statement_comparison(
    primary_statement_path: str, comparison_statement_path: str, instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Compare two statements and summarize key differences."""

    comparison_system_prompt = (
        "You are a financial analyst comparing two bank statements. Use the "
        "available tools to load each statement and call compare_statements to "
        "obtain structured deltas. Produce a clear narrative highlighting where "
        "spending and income increased or decreased, quantify the top category "
        "changes, and explain the overall impact on net savings. Suggest one or two "
        "practical follow-up actions."
    )
    agent = build_budget_agent(system_prompt=comparison_system_prompt)
    combined_prompt = (
        "The primary bank statement is located at: "
        f"{primary_statement_path}. "
        "The comparison bank statement is located at: "
        f"{comparison_statement_path}. "
        "Use load_statement on each path and call compare_statements with both "
        "paths before drafting your answer. Report amounts in USD with appropriate "
        "signs (prefix expenses with a minus if needed). "
        f"Additional instructions: {instructions or COMPARISON_DEFAULT_INSTRUCTIONS}"
    )
    result = invoke_agent(agent, combined_prompt)
    logger.info(
        "Statement comparison completed for %s vs %s",
        primary_statement_path,
        comparison_statement_path,
    )
    return result


def _prompt_statement_path(default_statement: Path) -> str:
    """Interactively gather and validate the statement path from the user."""

    while True:
        user_input = input(
            "Enter the path to your CSV bank statement "
            f"(press Enter for sample: {_to_workspace_relative(str(default_statement))}): "
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


def _first_sentences(text: str, max_sentences: int = 1) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(parts[:max_sentences]).strip()


def _extract_intro_sentence(text: str) -> str:
    if not text:
        return ""
    shortened = _shorten_paths_in_text(text).replace("`", "")
    lowered = shortened.lower()
    variants = [
        "here's a detailed audit of your bank statement from ",
        "here is a detailed audit of your bank statement from ",
        "here's a detailed budgeting report based on your bank statement from ",
        "here is a detailed budgeting report based on your bank statement from ",
    ]
    start_index = -1
    for variant in variants:
        idx = lowered.find(variant)
        if idx != -1:
            start_index = idx
            break
    if start_index == -1:
        return ""
    end_index = shortened.find(".", start_index)
    if end_index == -1:
        sentence = shortened[start_index:].strip()
    else:
        sentence = shortened[start_index : end_index + 1].strip()
    return sentence


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
    header_title: str = "Budget Analysis Report",
    header_subtitle: str = "Prepared by Spend Scout",
    secondary_statement_path: Optional[str] = None,
) -> str:
    """Construct a polished standalone HTML document for the report."""

    try:
        primary_label = Path(statement_path).name
    except Exception:
        primary_label = statement_path
    rows = [
        f"<tr><th>Generated</th><td>{escape(_format_human_timestamp(generated_at))}</td></tr>",
        f"<tr><th>Statement</th><td>{escape(primary_label)}</td></tr>",
        f"<tr><th>Instructions</th><td>{escape(instructions)}</td></tr>",
    ]
    if secondary_statement_path:
        try:
            secondary_label = Path(secondary_statement_path).name
        except Exception:
            secondary_label = secondary_statement_path
        rows.append(
            f"<tr><th>Comparison</th><td>{escape(secondary_label)}</td></tr>"
        )
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
        .summary h4 {{
            margin-bottom: 0.4rem;
            color: var(--accent);
            font-size: 1.05rem;
        }}
    .summary p {{
      line-height: 1.7;
      margin-bottom: 1rem;
    }}
        .summary-list {{
      margin: 0 0 1.1rem 1.5rem;
      padding: 0;
    }}
        .category-sections {{
            display: flex;
            flex-direction: column;
            gap: 1.75rem;
            margin-top: 2rem;
        }}
        .category-block {{
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }}
        .category-block h3 {{
            margin: 0 0 1rem;
            color: var(--accent);
            font-size: 1.15rem;
        }}
        .category-block__meta {{
            color: var(--muted);
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
        }}
        .comparison-grid {{
            display: grid;
            gap: 1.5rem;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            margin-top: 2rem;
        }}
        .comparison-card {{
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            background: linear-gradient(180deg, #ffffff 0%, #f3f7ff 100%);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }}
        .comparison-card h3 {{
            margin: 0 0 0.75rem;
            color: var(--accent);
            font-size: 1.1rem;
        }}
        .comparison-card__subtitle {{
            margin: 0 0 1rem;
            color: var(--muted);
            font-size: 0.9rem;
        }}
        .comparison-stats {{
            list-style: none;
            margin: 0;
            padding: 0;
        }}
        .comparison-stats li {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.35rem 0;
            font-size: 0.95rem;
            border-bottom: 1px solid #eef2fb;
        }}
        .comparison-stats li:last-child {{
            border-bottom: none;
        }}
        .comparison-value {{
            font-variant-numeric: tabular-nums;
            font-weight: 600;
        }}
        .comparison-delta-positive {{
            color: #2e7d32;
        }}
        .comparison-delta-negative {{
            color: #d1394a;
        }}
        .comparison-delta-neutral {{
            color: var(--muted);
        }}
        table.tx-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.96rem;
        }}
        table.tx-table th {{
            text-align: left;
            padding: 0.5rem 0.65rem;
            color: var(--muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        table.tx-table td {{
            padding: 0.55rem 0.65rem;
            border-bottom: 1px solid #eef2fb;
        }}
        table.tx-table tr:last-child td {{
            border-bottom: none;
        }}
        .tx-amount {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .tx-amount.negative {{
            color: #d1394a;
        }}
        .tx-amount.positive {{
            color: #2e7d32;
        }}
        .category-total {{
            margin-top: 1rem;
            text-align: right;
            font-weight: 600;
            color: var(--text);
        }}
        table.comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 2rem;
            font-size: 0.95rem;
        }}
        table.comparison-table thead th {{
            text-align: left;
            padding: 0.6rem 0.5rem;
            color: var(--muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        table.comparison-table tbody td {{
            padding: 0.6rem 0.5rem;
            border-bottom: 1px solid #eef2fb;
            font-variant-numeric: tabular-nums;
        }}
        table.comparison-table tbody td:first-child {{
            font-variant-numeric: normal;
        }}
        table.comparison-table tbody td:not(:first-child) {{
            text-align: right;
        }}
        table.comparison-table tbody tr:last-child td {{
            border-bottom: none;
        }}
                .trace-grid {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}
        .trace-card {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            align-items: flex-start;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }}
        .trace-card__title {{
            font-weight: 600;
            margin: 0;
            color: var(--accent);
        }}
        .trace-card__details {{
            width: 100%;
            margin: 0;
            padding-left: 0;
            color: var(--muted);
            list-style: none;
            font-size: 0.94rem;
            line-height: 1.45;
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
        <h1>{escape(header_title)}</h1>
        <p>{escape(header_subtitle)}</p>
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


def _generate_transaction_report(
    statement_path: str, query: str, outputs_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Run a transaction query and persist the findings to an HTML report."""

    result = run_transaction_query(statement_path, query)
    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_file = outputs_dir / f"transaction_query_result_{timestamp}.html"
    final_reply = extract_final_reply(result).strip()
    transactions, cleaned_text = _extract_transactions(final_reply)
    trace_block = format_tool_trace(result.get("messages", []))
    if cleaned_text:
        narrative_html = _format_summary_html(_shorten_paths_in_text(cleaned_text))
    else:
        narrative_html = "<p>Matching transactions grouped by category are summarized below.</p>"
    categories_html = _render_transaction_categories(transactions)
    summary_html = narrative_html + categories_html
    trace_html = _format_trace_html(trace_block)
    document = _build_html_document(
        generated_at=generated_at,
        statement_path=statement_path,
        instructions=query,
        summary_html=summary_html,
        trace_html=trace_html,
        header_title="Transaction Query Report",
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(document)
    return output_file, result


_HEADING_MARKER = re.compile(r"^#+\s*(.+)$")
_NUMBERED_MARKER = re.compile(r"^\d+\.\s+")
_TRANSACTION_LINE_PATTERN = re.compile(
    r"Date:\s*(?P<date>[^,]+),\s*Description:\s*(?P<description>[^,]+),\s*Category:\s*(?P<category>[^,]+),\s*Amount:\s*(?P<amount>[\-$0-9.,]+)"
)
_SUBSCRIPTION_SPLIT_PATTERN = re.compile(r"\s+[-–—]\s+")
_SUBSCRIPTION_AMOUNT_PATTERN = re.compile(r"\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_SUBSCRIPTION_FREQ_KEYWORDS = [
    ("weekly", "Weekly"),
    ("biweekly", "Biweekly"),
    ("fortnight", "Biweekly"),
    ("semi-month", "Semi-monthly"),
    ("semimonth", "Semi-monthly"),
    ("monthly", "Monthly"),
    ("quarter", "Quarterly"),
    ("annual", "Annual"),
    ("annually", "Annual"),
    ("yearly", "Annual"),
    ("every other month", "Every other month"),
]

_SUBSCRIPTION_METRIC_PHRASES = [
    "average monthly spend",
    "average monthly cost",
    "monthly cost",
    "monthly spend",
    "average spend",
    "amount",
    "estimated monthly",
    "average monthly",
]


def _format_currency(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _format_delta(amount: float) -> str:
    formatted = _format_currency(amount)
    return f"+{formatted}" if amount > 0 else formatted


def _extract_transactions(text: str) -> Tuple[List[Dict[str, Any]], str]:
    transactions: List[Dict[str, Any]] = []
    kept_lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            kept_lines.append(raw_line)
            continue
        normalized = stripped.lstrip("-*• ")
        match = _TRANSACTION_LINE_PATTERN.search(normalized)
        if not match:
            kept_lines.append(raw_line)
            continue
        amount_str = match.group("amount")
        amount_clean = amount_str.upper().replace("USD", "")
        amount_clean = amount_clean.replace("$", "").replace(",", "").strip()
        try:
            amount_value = float(amount_clean)
        except ValueError:
            kept_lines.append(raw_line)
            continue
        transactions.append(
            {
                "date": match.group("date").strip(),
                "description": match.group("description").strip(),
                "category": match.group("category").strip(),
                "amount": amount_value,
            }
        )
    cleaned_text = "\n".join(line for line in kept_lines if line.strip()).strip()
    return transactions, cleaned_text


def _render_transaction_categories(transactions: List[Dict[str, Any]]) -> str:
    if not transactions:
        return ""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for tx in transactions:
        grouped.setdefault(tx["category"], []).append(tx)
    sections: List[str] = ["<div class=\"category-sections\">"]
    for category in sorted(grouped.keys()):
        entries = grouped[category]
        total = sum(item["amount"] for item in entries)
        sections.append("<article class=\"category-block\">")
        sections.append(f"<h3>{escape(category)}</h3>")
        sections.append(
            f"<div class=\"category-block__meta\">{len(entries)} transaction(s) · Total {escape(_format_currency(total))}</div>"
        )
        sections.append("<table class=\"tx-table\"><thead><tr><th>Date</th><th>Description</th><th class=\"tx-amount\">Amount</th></tr></thead><tbody>")
        for tx in entries:
            amount_class = "positive" if tx["amount"] >= 0 else "negative"
            sections.append(
                "<tr>"
                f"<td>{escape(tx['date'])}</td>"
                f"<td>{escape(tx['description'])}</td>"
                f"<td class=\"tx-amount {amount_class}\">{escape(_format_currency(tx['amount']))}</td>"
                "</tr>"
            )
        sections.append("</tbody></table>")
        sections.append(
            f"<div class=\"category-total\">Category total: {escape(_format_currency(total))}</div>"
        )
        sections.append("</article>")
    sections.append("</div>")
    return "".join(sections)


def _extract_comparison_payload(messages: List[BaseMessage]) -> Optional[Dict[str, Any]]:
    payload: Optional[Dict[str, Any]] = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "compare_statements":
            if isinstance(msg.content, dict):
                payload = dict(msg.content)
                continue
            raw_text = _coerce_message_text(msg.content).strip()
            if not raw_text:
                continue
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
    return payload


def _render_comparison_sections(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""

    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _label(path_str: Any, fallback: str) -> str:
        if not isinstance(path_str, str) or not path_str:
            return fallback
        try:
            return Path(path_str).name or fallback
        except Exception:
            return fallback

    def _delta_class(value: float) -> str:
        if value > 0:
            return "comparison-value comparison-delta-positive"
        if value < 0:
            return "comparison-value comparison-delta-negative"
        return "comparison-value comparison-delta-neutral"

    primary = payload.get("primary", {}) if isinstance(payload.get("primary"), dict) else {}
    comparison = payload.get("comparison", {}) if isinstance(payload.get("comparison"), dict) else {}
    difference = payload.get("difference", {}) if isinstance(payload.get("difference"), dict) else {}

    primary_totals = primary.get("totals", {}) if isinstance(primary.get("totals"), dict) else {}
    comparison_totals = (
        comparison.get("totals", {}) if isinstance(comparison.get("totals"), dict) else {}
    )

    cards: List[str] = ["<div class=\"comparison-grid\">"]

    cards.append("<article class=\"comparison-card\">")
    cards.append("<h3>Primary Statement</h3>")
    cards.append(
        f"<p class=\"comparison-card__subtitle\">{escape(_label(primary.get('path'), 'Primary'))}</p>"
    )
    cards.append("<ul class=\"comparison-stats\">")
    cards.append(
        f"<li><span>Income</span><span class=\"comparison-value\">{escape(_format_currency(_to_float(primary_totals.get('income'))))}</span></li>"
    )
    cards.append(
        f"<li><span>Expenses</span><span class=\"comparison-value\">{escape(_format_currency(_to_float(primary_totals.get('expenses'))))}</span></li>"
    )
    cards.append(
        f"<li><span>Net</span><span class=\"comparison-value\">{escape(_format_currency(_to_float(primary_totals.get('net'))))}</span></li>"
    )
    cards.append("</ul>")
    cards.append("</article>")

    cards.append("<article class=\"comparison-card\">")
    cards.append("<h3>Comparison Statement</h3>")
    cards.append(
        f"<p class=\"comparison-card__subtitle\">{escape(_label(comparison.get('path'), 'Comparison'))}</p>"
    )
    cards.append("<ul class=\"comparison-stats\">")
    cards.append(
        f"<li><span>Income</span><span class=\"comparison-value\">{escape(_format_currency(_to_float(comparison_totals.get('income'))))}</span></li>"
    )
    cards.append(
        f"<li><span>Expenses</span><span class=\"comparison-value\">{escape(_format_currency(_to_float(comparison_totals.get('expenses'))))}</span></li>"
    )
    cards.append(
        f"<li><span>Net</span><span class=\"comparison-value\">{escape(_format_currency(_to_float(comparison_totals.get('net'))))}</span></li>"
    )
    cards.append("</ul>")
    cards.append("</article>")

    delta_income = _to_float(difference.get("income_change"))
    delta_expense = _to_float(difference.get("expense_change"))
    delta_net = _to_float(difference.get("net_change"))

    cards.append("<article class=\"comparison-card\">")
    cards.append("<h3>Period Change</h3>")
    cards.append("<ul class=\"comparison-stats\">")
    cards.append(
        f"<li><span>Income</span><span class=\"{_delta_class(delta_income)}\">{escape(_format_delta(delta_income))}</span></li>"
    )
    cards.append(
        f"<li><span>Expenses</span><span class=\"{_delta_class(delta_expense)}\">{escape(_format_delta(delta_expense))}</span></li>"
    )
    cards.append(
        f"<li><span>Net</span><span class=\"{_delta_class(delta_net)}\">{escape(_format_delta(delta_net))}</span></li>"
    )
    cards.append("</ul>")
    cards.append("</article>")

    cards.append("</div>")

    sections: List[str] = cards

    category_rows = payload.get("category_deltas")
    if isinstance(category_rows, list) and category_rows:
        sorted_rows = sorted(
            [row for row in category_rows if isinstance(row, dict)],
            key=lambda row: max(
                abs(_to_float(row.get("expense_change"))),
                abs(_to_float(row.get("income_change"))),
            ),
            reverse=True,
        )
        if sorted_rows:
            sections.append(
                "<table class=\"comparison-table\"><thead><tr>"
                "<th>Category</th><th>Baseline Income</th><th>Baseline Expenses" \
                "</th><th>Comparison Income</th><th>Comparison Expenses</th>" \
                "<th>Income Change</th><th>Expense Change</th>" \
                "</tr></thead><tbody>"
            )
            for entry in sorted_rows:
                category = escape(str(entry.get("category", "Uncategorized")))
                baseline_income = _format_currency(_to_float(entry.get("baseline_income")))
                baseline_expenses = _format_currency(_to_float(entry.get("baseline_expenses")))
                comparison_income = _format_currency(_to_float(entry.get("comparison_income")))
                comparison_expenses = _format_currency(_to_float(entry.get("comparison_expenses")))
                inc_change_val = _to_float(entry.get("income_change"))
                exp_change_val = _to_float(entry.get("expense_change"))
                income_change = _format_delta(inc_change_val)
                expense_change = _format_delta(exp_change_val)
                sections.append(
                    "<tr>"
                    f"<td>{category}</td>"
                    f"<td>{escape(baseline_income)}</td>"
                    f"<td>{escape(baseline_expenses)}</td>"
                    f"<td>{escape(comparison_income)}</td>"
                    f"<td>{escape(comparison_expenses)}</td>"
                    f"<td class=\"{_delta_class(inc_change_val)}\">{escape(income_change)}</td>"
                    f"<td class=\"{_delta_class(exp_change_val)}\">{escape(expense_change)}</td>"
                    "</tr>"
                )
            sections.append("</tbody></table>")

    return "".join(sections)


def _extract_subscriptions(text: str) -> Tuple[List[Dict[str, Any]], str]:
    subscriptions: List[Dict[str, Any]] = []
    kept_lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            kept_lines.append(raw_line)
            continue
        normalized = stripped.lstrip("-*• ")
        lowered = normalized.lower()
        normalized = normalized.replace("**", "")

        attributes: Dict[str, str] = {}

        token_map: Dict[str, str] = {}
        key_value_pairs: List[Tuple[str, str]] = []
        if ":" in normalized and any(
            keyword in lowered for keyword in ("vendor", "merchant", "frequency", "monthly", "amount", "cost", "spend")
        ):
            for token in re.split(r"[|;]", normalized):
                if ":" not in token:
                    continue
                key, value = token.split(":", 1)
                clean_key = key.strip()
                clean_value = value.strip()
                token_map[clean_key.lower()] = clean_value
                key_value_pairs.append((clean_key, clean_value))
            attributes = {clean_key.lower(): clean_value for clean_key, clean_value in key_value_pairs if clean_value}
        amount_value: Optional[float] = None
        amount_token: Optional[str] = None

        if token_map:
            vendor = token_map.get("vendor") or token_map.get("merchant")
            if not vendor and key_value_pairs:
                vendor = key_value_pairs[0][0]
            frequency = token_map.get("frequency")
            if not frequency and key_value_pairs:
                candidate = key_value_pairs[0][1].lower()
                for keyword, label in _SUBSCRIPTION_FREQ_KEYWORDS:
                    if keyword in candidate:
                        frequency = label
                        break
                if not frequency:
                    if "per month" in candidate or "/month" in candidate:
                        frequency = "Monthly"
                    elif "per week" in candidate or "/week" in candidate:
                        frequency = "Weekly"
                    elif "per year" in candidate or "/year" in candidate:
                        frequency = "Annual"
            amount_field = (
                token_map.get("monthly spend")
                or token_map.get("monthly cost")
                or token_map.get("spend")
                or token_map.get("amount")
                or token_map.get("cost")
            )
            if amount_field:
                match = _SUBSCRIPTION_AMOUNT_PATTERN.search(amount_field)
                if match:
                    amount_clean = match.group(1).replace(",", "")
                    try:
                        amount_value = float(amount_clean)
                        amount_token = match.group(0)
                    except ValueError:
                        amount_value = None
            if amount_field is None and key_value_pairs:
                for _, candidate_value in key_value_pairs:
                    match = _SUBSCRIPTION_AMOUNT_PATTERN.search(candidate_value)
                    if match:
                        amount_clean = match.group(1).replace(",", "")
                        try:
                            amount_value = float(amount_clean)
                            amount_token = match.group(0)
                            break
                        except ValueError:
                            amount_value = None
            detail_text = ""
        else:
            split_parts = _SUBSCRIPTION_SPLIT_PATTERN.split(normalized, maxsplit=1)
            if len(split_parts) == 2:
                vendor = split_parts[0].strip()
                remainder = split_parts[1].strip()
            else:
                vendor = normalized
                remainder = ""
            frequency: Optional[str] = None
            for keyword, label in _SUBSCRIPTION_FREQ_KEYWORDS:
                if keyword in lowered:
                    frequency = label
                    break
            if not frequency:
                if "per month" in lowered or "/month" in lowered:
                    frequency = "Monthly"
                elif "per week" in lowered or "/week" in lowered:
                    frequency = "Weekly"
                elif "per year" in lowered or "/year" in lowered:
                    frequency = "Annual"
            amount_search = _SUBSCRIPTION_AMOUNT_PATTERN.search(remainder or normalized)
            if amount_search:
                amount_clean = amount_search.group(1).replace(",", "")
                try:
                    amount_value = float(amount_clean)
                    amount_token = amount_search.group(0)
                except ValueError:
                    amount_value = None
            detail_text = remainder.strip()
            attributes = {}

        vendor = (vendor or "").strip()

        if amount_token and amount_token in vendor:
            vendor = vendor.replace(amount_token, "", 1).strip(" -:;•")

        if amount_value is None:
            kept_lines.append(raw_line)
            continue

        if detail_text and amount_token and amount_token in detail_text:
            detail_text = detail_text.replace(amount_token, "", 1).strip()

        if detail_text and vendor and detail_text.lower() == vendor.lower():
            detail_text = ""

        quote_matches = re.findall(r"\"([^\"]+)\"", normalized)
        if not quote_matches and detail_text:
            quote_matches = re.findall(r"\"([^\"]+)\"", detail_text)

        heading_candidates = [
            vendor,
            quote_matches[0] if quote_matches else "",
            detail_text.strip() if detail_text else "",
            normalized.strip(),
            raw_line.strip(),
        ]
        heading = next((candidate for candidate in heading_candidates if candidate), None)
        if heading is None:
            heading = f"Subscription {len(subscriptions) + 1}"
        if amount_token and heading.startswith(str(amount_token)):
            heading = heading[len(str(amount_token)) :].strip(" -:;•")
        if not heading:
            heading = f"Subscription {len(subscriptions) + 1}"

        subscriptions.append(
            {
                "title": heading,
                "vendor": vendor,
                "frequency": (frequency or "").strip(),
                "monthly_cost": amount_value,
                "details": detail_text.strip() if detail_text else "",
                "source": raw_line.strip(),
                "normalized": normalized.strip(),
                "attributes": attributes,
            }
        )

    cleaned_text = "\n".join(line for line in kept_lines if line.strip()).strip()
    return subscriptions, cleaned_text


def _strip_subscription_amounts(text: str) -> str:
    cleaned = _SUBSCRIPTION_AMOUNT_PATTERN.sub("", text)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" -:;•()")


def _looks_like_metric_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _SUBSCRIPTION_METRIC_PHRASES)


def _render_subscription_sections(subscriptions: List[Dict[str, Any]]) -> str:
    if not subscriptions:
        return ""

    sections: List[str] = ["<div class=\"subscription-grid\">"]
    for entry in subscriptions:
        attributes = entry.get("attributes") or {}
        candidate_fields: List[str] = []
        for candidate in [
            entry.get("title"),
            entry.get("vendor"),
            attributes.get("vendor"),
            attributes.get("merchant"),
            attributes.get("service"),
            attributes.get("subscription"),
            attributes.get("provider"),
            attributes.get("name"),
            attributes.get("description"),
            entry.get("details"),
            entry.get("normalized"),
            entry.get("source"),
        ]:
            if not candidate:
                continue
            candidate_fields.append(str(candidate))

        title_raw = ""
        for candidate in candidate_fields:
            stripped = re.sub(r"^[*•-]\s*", "", candidate).strip()
            stripped = _strip_subscription_amounts(stripped)
            if not stripped:
                continue
            if stripped.replace(".", "", 1).isdigit():
                continue
            if _looks_like_metric_phrase(stripped):
                continue
            title_raw = stripped
            break
        if not title_raw:
            title_raw = "Subscription"

        title_html = escape(title_raw)
        monthly_cost = entry.get("monthly_cost")
        if monthly_cost is not None:
            price_line = f"{_format_currency(monthly_cost)} per month"
        else:
            price_line = "Price unavailable"

        sections.append("<article class=\"subscription-card\">")
        sections.append(f"<h3 class=\"subscription-card__title\">{title_html}</h3>")
        sections.append(
            f"<p class=\"subscription-card__price\">{escape(price_line)}</p>"
        )
        sections.append("</article>")
    sections.append("</div>")
    return "".join(sections)


def _generate_comparison_report(
    primary_path: str,
    comparison_path: str,
    instructions: Optional[str],
    outputs_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    """Run a comparison analysis and persist the narrative to an HTML report."""

    applied_instructions = instructions or COMPARISON_DEFAULT_INSTRUCTIONS
    result = run_statement_comparison(primary_path, comparison_path, instructions)
    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_file = outputs_dir / f"statement_comparison_result_{timestamp}.html"

    final_reply = extract_final_reply(result).strip()
    if final_reply:
        narrative_html = _format_summary_html(_shorten_paths_in_text(final_reply))
    else:
        narrative_html = "<p><em>No comparison summary was returned.</em></p>"

    payload = _extract_comparison_payload(result.get("messages", []))
    sections_html = _render_comparison_sections(payload or {})
    summary_html = narrative_html + sections_html

    trace_block = format_tool_trace(result.get("messages", []))
    trace_html = _format_trace_html(trace_block)

    document = _build_html_document(
        generated_at=generated_at,
        statement_path=primary_path,
        instructions=applied_instructions,
        summary_html=summary_html,
        trace_html=trace_html,
        header_title="Statement Comparison Report",
        header_subtitle="Prepared by Spend Scout",
        secondary_statement_path=comparison_path,
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(document)

    logger.info("Statement comparison report saved to %s", output_file)
    return output_file, result


def _generate_subscription_report(
    statement_path: str, instructions: Optional[str], outputs_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Run subscription detection and persist the findings to an HTML report."""

    result = run_subscription_detection(statement_path, instructions)
    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_file = outputs_dir / f"subscription_detection_result_{timestamp}.html"

    final_reply = extract_final_reply(result).strip()
    subscriptions, cleaned_text = _extract_subscriptions(final_reply)

    if cleaned_text:
        narrative_html = _format_summary_html(_shorten_paths_in_text(cleaned_text))
    elif subscriptions:
        narrative_html = "<p>Recurring services flagged by the agent are summarized below.</p>"
    else:
        narrative_html = "<p><em>No subscription findings were returned.</em></p>"

    sections_html = _render_subscription_sections(subscriptions)
    summary_html = narrative_html + sections_html

    trace_block = format_tool_trace(result.get("messages", []))
    trace_html = _format_trace_html(trace_block)
    instructions_text = instructions or SUBSCRIPTION_DEFAULT_INSTRUCTIONS

    document = _build_html_document(
        generated_at=generated_at,
        statement_path=statement_path,
        instructions=instructions_text,
        summary_html=summary_html,
        trace_html=trace_html,
        header_title="Subscription Detection Report",
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(document)

    logger.info("Subscription detection report saved to %s", output_file)
    return output_file, result


def generate_budget_report(statement_path: str, instructions: str) -> Path:
    """Public helper to generate a budget analysis report and return the file path."""

    return _generate_html_report(statement_path, instructions, ROOT_DIR / "outputs")


def generate_transaction_report(statement_path: str, query: str) -> Path:
    """Public helper to generate a transaction query report and return the file path."""

    report_path, _ = _generate_transaction_report(statement_path, query, ROOT_DIR / "outputs")
    return report_path


def generate_subscription_report(statement_path: str, instructions: Optional[str]) -> Path:
    """Public helper to generate a subscription detection report and return the file path."""

    report_path, _ = _generate_subscription_report(statement_path, instructions, ROOT_DIR / "outputs")
    return report_path


def generate_comparison_report(
    primary_statement_path: str, comparison_statement_path: str, instructions: Optional[str]
) -> Path:
    """Public helper to generate a statement comparison report and return the file path."""

    report_path, _ = _generate_comparison_report(
        primary_statement_path,
        comparison_statement_path,
        instructions,
        ROOT_DIR / "outputs",
    )
    return report_path


def _render_cli_output(title: str, text: str) -> None:
    """Pretty-print agent text blocks for terminal readability."""

    print(f"{title}\n{CLI_DIVIDER}")
    stripped = text.strip()
    if not stripped:
        print("No content returned.\n")
        return

    blank_pending = False
    footnotes: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if not blank_pending:
                print()
                blank_pending = True
            continue
        blank_pending = False

        line = _BOLD_PATTERN.sub(r"\1", line)

        lowered = line.lower()
        note_start = -1
        if "estimated single instance" in lowered:
            note_start = lowered.find("estimated single instance")
        if note_start == -1 and "only one transaction found in" in lowered:
            note_start = lowered.find("only one transaction found in")
        if note_start != -1:
            line = line[:note_start].rstrip(" ,;:-–—")
            if line:
                line = line.strip()
            else:
                line = ""
            footnotes.add(SUBSCRIPTION_FOOTNOTE)
            if not line:
                continue

        heading_match = _HEADING_MARKER.match(line)
        if heading_match:
            print(heading_match.group(1).upper())
            continue
        if line.startswith("- "):
            print(f"  - {line[2:].strip()}")
            continue
        if line.startswith("* "):
            print(f"  - {line[2:].strip()}")
            continue
        if _NUMBERED_MARKER.match(line):
            print(f"  {line}")
            continue
        if line.endswith(":"):
            print(line.rstrip(":").upper() + ":")
            continue
        print(line)
    print()
    if footnotes:
        print(f"Notes\n{CLI_DIVIDER}")
        for note in sorted(footnotes):
            print(f"  - {note}")
        print()


def _run_analysis_flow(statement_path: str, instructions: str, outputs_dir: Path) -> None:
    """Execute an agent run with helpful CLI messaging."""

    print("\nWorking on your budgeting analysis...\n")
    try:
        report_path = _generate_html_report(statement_path, instructions, outputs_dir)
        relative_report = _to_workspace_relative(str(report_path))
        print(f"Analysis complete! Results saved to: {relative_report}\n")
    except Exception as exc:
        logger.exception("Budget analysis run failed")
        print(f"An error occurred while running the agent: {exc}\n")


def _run_transaction_query_flow(statement_path: str, query: str, outputs_dir: Path) -> None:
    """Execute a transaction search and display results to the CLI."""

    try:
        report_path, result = _generate_transaction_report(statement_path, query, outputs_dir)
        relative_report = _to_workspace_relative(str(report_path))
        print(f"\nTransaction report saved to outputs: {relative_report}\n")
        trace = format_tool_trace(result.get("messages", []))
        if trace:
            show_trace = input("Show tool trace? (yes/[no]): ").strip().lower()
            if show_trace in {"y", "yes"}:
                print("\nTool trace:\n--------------------------------------------------")
                print(trace + "\n")
    except Exception as exc:
        logger.exception("Transaction query failed")
        print(f"An error occurred while searching transactions: {exc}\n")


def _run_subscription_detection_flow(
    statement_path: str, instructions: Optional[str], outputs_dir: Path
) -> None:
    """Execute subscription detection and display the agent's findings."""

    try:
        _, result = _generate_subscription_report(statement_path, instructions, outputs_dir)
        trace = format_tool_trace(result.get("messages", []))
        if trace:
            show_trace = input("Show tool trace? (yes/[no]): ").strip().lower()
            if show_trace in {"y", "yes"}:
                print("\nTool trace:\n--------------------------------------------------")
                print(trace + "\n")
    except Exception as exc:
        logger.exception("Subscription detection failed")
        print(f"An error occurred while detecting subscriptions: {exc}\n")


def _run_statement_comparison_flow(
    _: str,
    default_primary: Path,
    default_comparison: Path,
    outputs_dir: Path,
) -> None:
    """Compare two statements and persist the agent's findings to an HTML report."""

    print("\nComparing statements...\n")
    primary_default_display = _to_workspace_relative(str(default_primary))
    comparison_default_display = _to_workspace_relative(str(default_comparison))

    while True:
        primary_prompt = (
            "Enter the path to the primary CSV "
            f"(press Enter for sample: {primary_default_display}): "
        )
        user_input = input(primary_prompt).strip()
        if not user_input:
            candidate = str(default_primary)
        else:
            candidate = user_input
        try:
            primary_path = _resolve_path(candidate)
            break
        except Exception as exc:
            print(f"Invalid primary path: {exc}\nPlease try again.\n")

    while True:
        comparison_prompt = (
            "Enter the path to the comparison CSV "
            f"(press Enter for sample: {comparison_default_display}): "
        )
        user_input = input(comparison_prompt).strip()
        if not user_input:
            candidate = str(default_comparison)
        else:
            candidate = user_input
        try:
            comparison_path = _resolve_path(candidate)
            break
        except Exception as exc:
            print(f"Invalid comparison path: {exc}\nPlease try again.\n")

    custom_instructions = input(
        "Enter additional comparison guidance (press Enter for default focus): "
    ).strip()
    instructions = custom_instructions or None

    try:
        report_path, result = _generate_comparison_report(
            str(primary_path),
            str(comparison_path),
            instructions,
            outputs_dir,
        )
        relative_report = _to_workspace_relative(str(report_path))
        print(f"\nComparison report saved to outputs: {relative_report}\n")
        trace = format_tool_trace(result.get("messages", []))
        if trace:
            show_trace = input("Show tool trace? (yes/[no]): ").strip().lower()
            if show_trace in {"y", "yes"}:
                print("\nTool trace:\n--------------------------------------------------")
                print(trace + "\n")
    except Exception as exc:
        logger.exception("Statement comparison failed")
        print(f"An error occurred while comparing statements: {exc}\n")


def main() -> None:
    """Interactive CLI entry point with multiple analysis options."""

    workspace = Path(__file__).parent
    default_statement = workspace / "samples" / "sample_statement_large.csv"
    comparison_sample_statement = workspace / "samples" / "sample_statement_next_month.csv"
    outputs_dir = workspace / "outputs"
    print("Spend Scout")
    print("--------------------------------------------------")
    sample_dir_display = _to_workspace_relative(str(default_statement.parent))
    default_sample_display = _to_workspace_relative(str(default_statement))
    comparison_sample_display = _to_workspace_relative(str(comparison_sample_statement))
    print(f"Sample statements available under: {sample_dir_display}")
    print(f" - Default sample: {default_sample_display}")
    print(f" - Comparison sample: {comparison_sample_display}")
    statement_path = _prompt_statement_path(default_statement)
    while True:
        print("What would you like to do next?")
        print("  1) Run quick summary (default prompt)")
        print("  2) Run custom analysis (you supply instructions)")
        print("  3) Find transactions (keyword query)")
        print("  4) Detect subscriptions")
        print("  5) Compare two statements")
        print("  6) Change statement path")
        print("  7) Exit")
        choice = input("Select an option (1-7): ").strip().lower()
        if choice in {"1", "a"}:
            _run_analysis_flow(statement_path, DEFAULT_SUMMARY_PROMPT, outputs_dir)
        elif choice in {"2", "b"}:
            custom_prompt = input(
                "Enter the instructions you want the agent to follow: "
            ).strip()
            if not custom_prompt:
                print("Custom instructions cannot be empty. Please try again.\n")
                continue
            _run_analysis_flow(statement_path, custom_prompt, outputs_dir)
        elif choice in {"3", "f", "find", "search"}:
            search_query = input("Describe the transactions you want to find: ").strip()
            if not search_query:
                print("Search instructions cannot be empty. Please try again.\n")
                continue
            _run_transaction_query_flow(statement_path, search_query, outputs_dir)
        elif choice in {"4", "s", "subs", "subscription", "subscriptions"}:
            custom_instructions = input(
                "Enter any extra guidance for detecting subscriptions (press Enter for default): "
            ).strip()
            instructions = custom_instructions or None
            _run_subscription_detection_flow(statement_path, instructions, outputs_dir)
        elif choice in {"5", "compare", "comparison", "compare statements"}:
            _run_statement_comparison_flow(
                statement_path,
                default_statement,
                comparison_sample_statement,
                outputs_dir,
            )
        elif choice in {"6", "c", "change"}:
            statement_path = _prompt_statement_path(default_statement)
        elif choice in {"7", "q", "quit", "exit"}:
            print("Goodbye!")
            break
        else:
            print("Unrecognized option. Please choose 1, 2, 3, 4, 5, 6, or 7.\n")


if __name__ == "__main__":
    main()
