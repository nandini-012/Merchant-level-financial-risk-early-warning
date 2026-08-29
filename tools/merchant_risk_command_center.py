"""Read-only local Merchant Risk Command Center for existing alert outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALERT_FILE = ROOT / "data" / "outputs" / "merchant_risk_output.csv"
EXPLANATION_FILE = ROOT / "data" / "outputs" / "merchant_risk_explanations.csv"
REPORT_FILE = ROOT / "data" / "outputs" / "operational_risk_report.md"

ALERT_COLUMNS = ["merchant", "prediction_date", "risk_score", "alert"]
EXPLANATION_COLUMNS = [
    *ALERT_COLUMNS,
    "previous_7d_average_transaction_amount",
    "previous_7d_maximum_transaction_amount",
    "previous_7d_total_transaction_amount",
    "previous_14d_transaction_count",
    "previous_14d_fraud_rate",
    "previous_7d_transaction_count_change",
    "explanation",
]


def file_digest(path: Path) -> str:
    """Return a digest used to confirm source outputs stayed unchanged."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    """Read and validate an existing CSV output without changing it."""
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != expected_columns:
            raise RuntimeError(
                f"{path.name} columns do not match the inspected output schema."
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{path.name} contains no rows.")
    return rows


def report_metric(report: str, label: str) -> str:
    """Read a displayed metric from the existing Markdown report."""
    match = re.search(rf"\| {re.escape(label)} \| ([^|]+) \|", report)
    if not match:
        raise RuntimeError(f"Operational report does not contain '{label}'.")
    return match.group(1).strip()


def load_dashboard_data() -> tuple[list[dict[str, object]], int, str]:
    """Join existing alert and explanation outputs for read-only display."""
    input_files = [ALERT_FILE, EXPLANATION_FILE, REPORT_FILE]
    for path in input_files:
        if not path.is_file():
            raise RuntimeError(f"Required existing output is missing: {path}")
    digests_before = {path: file_digest(path) for path in input_files}

    alerts = read_csv(ALERT_FILE, ALERT_COLUMNS)
    explanations = read_csv(EXPLANATION_FILE, EXPLANATION_COLUMNS)
    report = REPORT_FILE.read_text(encoding="utf-8")

    alert_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for alert in alerts:
        key = (alert["merchant"], alert["prediction_date"])
        if key in alert_by_key:
            raise RuntimeError("merchant_risk_output.csv contains duplicate merchant/date pairs.")
        if alert["alert"] != "1":
            raise RuntimeError("merchant_risk_output.csv contains a non-alert row.")
        alert_by_key[key] = alert

    queue: list[dict[str, object]] = []
    seen_explanations: set[tuple[str, str]] = set()
    for row in explanations:
        key = (row["merchant"], row["prediction_date"])
        if key in seen_explanations:
            raise RuntimeError("merchant_risk_explanations.csv contains duplicate merchant/date pairs.")
        seen_explanations.add(key)
        alert = alert_by_key.get(key)
        if alert is None:
            raise RuntimeError("Explanation merchant/date pairs do not match the alert output.")
        if row["risk_score"] != alert["risk_score"] or row["alert"] != alert["alert"]:
            raise RuntimeError("Explanation risk scores or alerts do not match the alert output.")
        if not row["explanation"].strip():
            raise RuntimeError("An explanation row is empty.")
        queue.append(
            {
                "merchant": row["merchant"],
                "prediction_date": row["prediction_date"],
                "risk_score": float(row["risk_score"]),
                "alert": int(row["alert"]),
                "explanation": row["explanation"],
            }
        )

    if len(queue) != len(alert_by_key) or seen_explanations != set(alert_by_key):
        raise RuntimeError("Alert and explanation merchant/date pairs are not an exact match.")
    queue.sort(key=lambda row: (-float(row["risk_score"]), str(row["prediction_date"]), str(row["merchant"])))

    total_alerts = int(report_metric(report, "Alert rows").replace(",", ""))
    alerts_per_day = report_metric(report, "Alerts per calendar day")
    if total_alerts != len(queue):
        raise RuntimeError("Operational report alert count does not match the alert outputs.")
    if total_alerts != 912 or alerts_per_day != "20.2667":
        raise RuntimeError("Existing operational report does not contain the locked dashboard metrics.")

    digests_after = {path: file_digest(path) for path in input_files}
    if digests_before != digests_after:
        raise RuntimeError("An existing output changed while the dashboard was loading.")
    return queue, total_alerts, alerts_per_day


def render_page(queue: list[dict[str, object]], total_alerts: int, alerts_per_day: str) -> str:
    """Render a self-contained browser interface with no write operations."""
    serialized_queue = json.dumps(queue, ensure_ascii=False).replace("</", "<\\/")
    first_alert = queue[0]
    title = html.escape("Merchant Risk Command Center")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, sans-serif; }}
    body {{ margin: 0; background: #f4f7fb; color: #172033; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 6px; }}
    .note {{ margin: 0 0 22px; color: #566277; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(180px, 260px)); gap: 16px; margin-bottom: 24px; }}
    .card, .panel {{ background: white; border: 1px solid #dbe2ee; border-radius: 8px; box-shadow: 0 1px 2px #1720330d; }}
    .card {{ padding: 16px; }} .card span {{ color: #566277; display: block; font-size: .9rem; }} .card strong {{ font-size: 1.7rem; }}
    .layout {{ display: grid; grid-template-columns: minmax(620px, 1.7fr) minmax(290px, .8fr); gap: 18px; align-items: start; }}
    .panel {{ padding: 16px; }} .table-wrap {{ max-height: 640px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }} th, td {{ padding: 10px; border-bottom: 1px solid #e7ecf4; text-align: left; }} th {{ background: #f8faff; position: sticky; top: 0; }}
    tbody tr {{ cursor: pointer; }} tbody tr:hover, tbody tr.selected {{ background: #eaf2ff; }}
    .score {{ font-variant-numeric: tabular-nums; }} .badge {{ color: #075f38; font-weight: 700; }}
    .detail-label {{ margin: 14px 0 3px; color: #566277; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }}
    .evidence {{ line-height: 1.55; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} .table-wrap {{ max-height: 430px; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Merchant Risk Command Center</h1>
    <p class="note">Read-only view of the existing locked merchant-alert outputs. Select an alert row to view its existing evidence text.</p>
    <section class="metrics" aria-label="Alert summary">
      <div class="card"><span>Total alerts</span><strong>{total_alerts}</strong></div>
      <div class="card"><span>Alerts/day</span><strong>{html.escape(alerts_per_day)}</strong></div>
    </section>
    <section class="layout">
      <div class="panel">
        <h2>Alert Queue</h2>
        <div class="table-wrap"><table><thead><tr><th>Merchant</th><th>Prediction date</th><th>Risk score</th><th>Alert</th></tr></thead><tbody id="queue"></tbody></table></div>
      </div>
      <aside class="panel" aria-live="polite">
        <h2>Selected Alert</h2>
        <div id="details"></div>
      </aside>
    </section>
  </main>
  <script>
    const alerts = {serialized_queue};
    const queue = document.getElementById('queue');
    const details = document.getElementById('details');
    function show(index) {{
      const item = alerts[index];
      details.replaceChildren();
      const entries = [
        ['Merchant', item.merchant], ['Prediction date', item.prediction_date],
        ['Risk score', item.risk_score.toFixed(12)], ['Alert', String(item.alert)],
        ['Explanation / evidence', item.explanation]
      ];
      entries.forEach(([label, value]) => {{
        const labelNode = document.createElement('div'); labelNode.className = 'detail-label'; labelNode.textContent = label;
        const valueNode = document.createElement('div'); valueNode.textContent = value;
        if (label === 'Explanation / evidence') valueNode.className = 'evidence';
        details.append(labelNode, valueNode);
      }});
      document.querySelectorAll('#queue tr').forEach((row, rowIndex) => row.classList.toggle('selected', rowIndex === index));
    }}
    alerts.forEach((item, index) => {{
      const row = document.createElement('tr');
      [item.merchant, item.prediction_date, item.risk_score.toFixed(12), String(item.alert)].forEach((value, column) => {{
        const cell = document.createElement('td'); cell.textContent = value;
        if (column === 2) cell.className = 'score';
        if (column === 3) cell.className = 'badge';
        row.appendChild(cell);
      }});
      row.addEventListener('click', () => show(index));
      row.tabIndex = 0; row.addEventListener('keydown', event => {{ if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); show(index); }} }});
      queue.appendChild(row);
    }});
    show(0);
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only Merchant Risk Command Center.")
    parser.add_argument("--host", default="127.0.0.1", help="Local host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Local port to bind (default: 8000).")
    parser.add_argument("--check", action="store_true", help="Validate source outputs without starting a server.")
    args = parser.parse_args()

    queue, total_alerts, alerts_per_day = load_dashboard_data()
    if args.check:
        print("Merchant Risk Command Center validation: passed")
        print(f"Total alerts: {total_alerts}")
        print(f"Alerts/day: {alerts_per_day}")
        print("Existing output files were read only and were not modified")
        return

    page = render_page(queue, total_alerts, alerts_per_day).encode("utf-8")

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required handler method name
            if self.path not in {"/", "/index.html"}:
                self.send_error(404, "Not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, _format: str, *_args: object) -> None:
            """Keep the local dashboard terminal output concise."""

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Merchant Risk Command Center running at http://{args.host}:{args.port}")
    print("Read-only: existing outputs are not modified. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMerchant Risk Command Center stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
