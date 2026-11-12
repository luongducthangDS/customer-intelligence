"""Regression gate: một lần train mới không được tệ hơn baseline đã chốt, và model phải thắng luật đơn giản.

    python -m ci.gate [artifacts/report.json]     # exit 1 nếu có gate fail

Baseline (`eval/baseline.json`) là số của lần train được công bố trong README. Chỉ cập nhật baseline có chủ đích,
kèm lý do trong commit; không nới để CI xanh (docs/RULES.md).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "eval" / "baseline.json"
TOL = 0.02  # dao động giữa các lần train (seed, phiên bản thư viện)


def get(report: dict, path: str) -> float:
    for k in path.split("."):
        report = report[k]
    return float(report)


def check(report: dict, baseline: dict, tol: float = TOL) -> list[str]:
    fails = []
    # 1. Không regression so với baseline đã chốt
    for path, base in baseline["metrics"].items():
        if (v := get(report, path)) < base - tol:
            fails.append(f"{path} = {v:.4f} < baseline {base:.4f} - {tol}")
    # 2. Model phải thắng baseline đơn giản trên cùng tập test
    for better, worse in baseline["must_beat"]:
        if get(report, better) <= get(report, worse):
            fails.append(f"{better} = {get(report, better):.4f} không thắng {worse} = {get(report, worse):.4f}")
    return fails


def main(path: Path = ROOT / "artifacts" / "report.json") -> int:
    fails = check(json.loads(Path(path).read_text()), json.loads(BASELINE.read_text()))
    for f in fails:
        print("FAIL", f)
    print("gate:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(*map(Path, sys.argv[1:2])))
