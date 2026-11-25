"""A portable HTML research report, with no server or external dependencies."""

import json
from pathlib import Path


def write_report(output, payload):
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    # A ticker, filename, or provider message must never terminate the script tag.
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (Path(output) / "report.html").write_text(template.replace("__REPORT_DATA__", data), encoding="utf-8")
