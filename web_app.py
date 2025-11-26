"""Web interface for the Bank Budgeting Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from bank_agent import (
    COMPARISON_DEFAULT_INSTRUCTIONS,
    DEFAULT_SUMMARY_PROMPT,
    ROOT_DIR,
    SUBSCRIPTION_DEFAULT_INSTRUCTIONS,
    generate_budget_report,
    generate_comparison_report,
    generate_subscription_report,
    generate_transaction_report,
)

app = Flask(__name__)
app.secret_key = os.environ.get("BUDGET_AGENT_APP_SECRET", "dev-secret-change-me")

UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SAMPLES_DIR = ROOT_DIR / "samples"
ALLOWED_EXTENSIONS = {".csv"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _sample_options() -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    if not SAMPLES_DIR.exists():
        return samples
    for sample_path in sorted(SAMPLES_DIR.glob("*.csv")):
        samples.append(
            {
                "value": sample_path.name,
                "label": sample_path.name.replace("_", " "),
            }
        )
    return samples


def _store_uploaded_file(field_name: str) -> Optional[Path]:
    file_storage = request.files.get(field_name)
    if not file_storage or not file_storage.filename:
        return None
    filename = secure_filename(file_storage.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only CSV statement files are supported.")
    unique_name = f"{uuid4().hex}_{filename}"
    destination = UPLOAD_DIR / unique_name
    file_storage.save(destination)
    return destination


def _resolve_statement_source(field_name: str, sample_field: str) -> Path:
    uploaded = _store_uploaded_file(field_name)
    if uploaded:
        return uploaded

    sample_choice = request.form.get(sample_field, "").strip()
    if sample_choice:
        sample_path = (SAMPLES_DIR / sample_choice).resolve()
        if sample_path.is_file():
            return sample_path
        raise ValueError("Selected sample statement could not be found.")

    raise ValueError("Please upload a CSV statement or choose a sample file.")


def _relative_output_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        return path.name
    return relative.as_posix()


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        sample_files=_sample_options(),
        default_summary_prompt=DEFAULT_SUMMARY_PROMPT,
        default_subscription_instructions=SUBSCRIPTION_DEFAULT_INSTRUCTIONS,
        default_comparison_instructions=COMPARISON_DEFAULT_INSTRUCTIONS,
    )


@app.post("/run")
def run_action():
    action = request.form.get("action", "").strip()
    try:
        if action == "quick-summary":
            statement_path = _resolve_statement_source("statement_file", "statement_sample")
            instructions = DEFAULT_SUMMARY_PROMPT
            report_path = generate_budget_report(str(statement_path), instructions)
        elif action == "custom-analysis":
            statement_path = _resolve_statement_source(
                "statement_file", "statement_sample"
            )
            instructions = request.form.get("custom_instructions", "").strip()
            if not instructions:
                raise ValueError("Please provide custom instructions for the analysis.")
            report_path = generate_budget_report(str(statement_path), instructions)
        elif action == "transaction-query":
            statement_path = _resolve_statement_source("statement_file", "statement_sample")
            query = request.form.get("transaction_query", "").strip()
            if not query:
                raise ValueError("Please specify the transaction query you want to run.")
            report_path = generate_transaction_report(str(statement_path), query)
        elif action == "subscription-detection":
            statement_path = _resolve_statement_source("statement_file", "statement_sample")
            instructions = request.form.get("subscription_instructions", "").strip()
            instructions_value: Optional[str] = instructions or None
            report_path = generate_subscription_report(str(statement_path), instructions_value)
        elif action == "statement-comparison":
            primary_path = _resolve_statement_source("primary_statement", "primary_sample")
            comparison_path = _resolve_statement_source(
                "comparison_statement", "comparison_sample"
            )
            instructions = request.form.get("comparison_instructions", "").strip()
            instructions_value = instructions or None
            report_path = generate_comparison_report(
                str(primary_path), str(comparison_path), instructions_value
            )
        else:
            raise ValueError("Unknown action. Please select a valid option.")
    except Exception as exc:  # noqa: BLE001
        flash(str(exc), "error")
        return redirect(url_for("index"))

    relative = _relative_output_path(Path(report_path))
    return redirect(url_for("serve_output", filename=relative))


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    host = os.environ.get("BUDGET_AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("BUDGET_AGENT_PORT", "5000"))
    debug = os.environ.get("BUDGET_AGENT_DEBUG", "true").lower() in {"1", "true", "yes"}
    print(f"Web UI available at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
