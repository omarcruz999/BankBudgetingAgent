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
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
from dotenv import load_dotenv  # type: ignore[import]


LOG_PATH = Path(__file__).parent / "agent.log"
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
        summary = {
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
        overview_raw = json.loads(spending_overview(path))
        if isinstance(overview_raw, dict) and overview_raw.get("error"):
            return f"Unable to create report: {overview_raw['error']}"
        summary = overview_raw
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
            events.append(
                {
                    "label": "User request",
                    "details": [_coerce_message_text(msg.content).strip()],
                }
            )
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for call in msg.tool_calls:
                    args = call.get("args", {})
                    label = f"Assistant called `{call['name']}`"
                    detail = f"Args: {_summarize_args(args)}"
                    event = {"label": label, "details": [detail]}
                    events.append(event)
                    call_id = call.get("id")
                    if call_id:
                        tool_events_by_id[str(call_id)] = event
            elif msg.content:
                events.append(
                    {
                        "label": "Assistant draft",
                        "details": [_coerce_message_text(msg.content).strip()],
                    }
                )
        elif isinstance(msg, ToolMessage):
            summary = _summarize_tool_response(msg.name, msg.content)
            if msg.tool_call_id and msg.tool_call_id in tool_events_by_id:
                tool_events_by_id[msg.tool_call_id]["details"].append(f"Result: {summary}")
            else:
                events.append(
                    {
                        "label": f"Tool `{msg.name}` result",
                        "details": [summary],
                    }
                )

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


def main() -> None:
    """CLI entry point that accepts a statement path and runs an example prompt."""

    workspace = Path(__file__).parent
    default_statement = workspace / "samples" / "sample_statement_large.csv"
    outputs_dir = workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print("Bank Statement Budgeting Agent")
    print("--------------------------------------------------")
    print(f"Sample statements available under: {default_statement.parent}")
    print(f" - Default sample: {default_statement}")
    user_path = input(
        f"Enter the path to your CSV bank statement (press Enter for sample: {default_statement}): "
    ).strip()
    if not user_path:
        user_path = str(default_statement)
    example_prompt = (
        "Summarize overall income vs expenses, highlight any category exceeding $500, "
        "call out the five largest expenses, describe spending trends you notice, "
        "and provide three actionable budgeting recommendations. Format amounts in USD "
        "and deliver a detailed narrative without asking follow-up questions."
    )
    try:
        print("Working on your budgeting analysis...\n")
        result = run_budget_review(user_path, example_prompt)
        generated_at = datetime.utcnow()
        timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
        output_file = outputs_dir / f"budget_agent_result_{timestamp}.md"
        final_reply = extract_final_reply(result).strip()
        trace_block = format_tool_trace(result.get("messages", []))
        with output_file.open("w", encoding="utf-8") as handle:
            handle.write("# Budget Analysis Report\n\n")
            handle.write(f"- **Generated on:** {generated_at.isoformat()}Z\n")
            handle.write(f"- **Statement reviewed:** {Path(user_path).expanduser().resolve()}\n")
            handle.write(f"- **Guidance prompt:** {example_prompt}\n\n")
            handle.write("## Summary\n\n")
            handle.write(final_reply or "_No response returned._")
            handle.write("\n\n## LLM Interaction Highlights\n\n")
            if trace_block.strip():
                handle.write(trace_block)
                if not trace_block.endswith("\n"):
                    handle.write("\n")
            else:
                handle.write("- No tool interactions were recorded.\n")
        print(f"Analysis complete! Results saved to: {output_file}")
    except Exception as exc:
        logger.exception("main execution failed")
        print(f"An error occurred while running the agent: {exc}")


if __name__ == "__main__":
    main()
