"""A portable HTML research report, with no server or external dependencies."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .comparison import measurement_comparison
from .selection import MAX_CANDIDATES


def refresh_report(output):
    output = Path(output)
    payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    if payload.get("kind") == "trade_selection":
        from .trade_quantum import render_trade_quantum_report
        render_trade_quantum_report(output)
        return
    write_report(output, payload)


def write_report(output, payload):
    output = Path(output)
    payload = dict(payload)
    payload["limitations"] = [
        f"This version supports up to {MAX_CANDIDATES} candidate decisions. Ideal simulation and exact enumeration scale exponentially."
        if text.startswith("Optimization and exact enumeration are deliberately limited to") else text
        for text in payload.get("limitations", [])
    ]
    hardware_path = output / "ibm" / "hardware_results.json"
    receipt_path = output / "ibm" / "submission.json"
    if hardware_path.exists():
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
        if not hardware.get("hardware_executed") or not receipt_path.exists():
            raise ValueError("Hardware visualization requires confirmed hardware results and a submission receipt.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256((output / "problem.json").read_bytes()).hexdigest()
        if receipt["problem_sha256"] != digest or hardware["job_id"] != receipt["job_id"] or receipt["source_run_id"] != payload["run_id"]:
            raise ValueError("Hardware receipt does not match this run and model.")
        payload["hardware"] = hardware
        payload["submission"] = receipt
        payload["comparison"] = measurement_comparison(payload["problem"], payload["quantum"], hardware)
    benchmark_path = output / "benchmarks" / "latest.json"
    if benchmark_path.exists():
        payload["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    trading_path = output / "trading" / "latest.json"
    if trading_path.exists():
        payload["trading"] = json.loads(trading_path.read_text(encoding="utf-8"))
    template_path = Path(__file__).with_name("report.html")
    template = template_path.read_text(encoding="utf-8")
    # A ticker, filename, or provider message must never terminate the script tag.
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    report = output / "report.html"
    report.write_text(template.replace("__REPORT_DATA__", data), encoding="utf-8")
    # The analysis remains unchanged. Record updates to its derived report and
    # preserve the original manifest's analysis/code/data provenance.
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
        if manifest["files_sha256"].get("report.html") != digest:
            revision = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "previous_report_sha256": manifest["files_sha256"].get("report.html"),
                        "report_sha256": digest,
                        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest()}
            for path in (hardware_path, receipt_path, benchmark_path, trading_path):
                if path.exists():
                    name = path.relative_to(output).as_posix()
                    value = hashlib.sha256(path.read_bytes()).hexdigest()
                    manifest["files_sha256"][name] = value
                    revision[name] = value
            manifest["files_sha256"]["report.html"] = digest
            manifest.setdefault("report_updates", []).append(revision)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
