"""Render checkpointed guided experiments as a human-readable Markdown report."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.features.build_features import ROOT


def render(state):
    lines = ["# Kết quả XGBoost T07", "",
             ("**Đã hoàn tất lựa chọn và xuất dự báo.**" if "selection" in state
              else "**Đang chạy; bảng dưới đây là kết quả tạm thời.**"), "",
             "Các điểm dưới đây là RMSE trên thang log1p sau khi cắt dự báo âm. "
             "Thí nghiệm cấu trúc và feature chọn theo trung bình B/C/D; "
             "20 cấu hình tune so sánh trên B/C. Fold A được dùng ở bước cuối. "
             "Fold A đã từng được xem ở các thử nghiệm trước nên không độc lập hoàn toàn.", "",
             "| Thí nghiệm | B | C | D | Trung bình có sẵn | Vòng tốt nhất B/C/D |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, entry in state.get("experiments", {}).items():
        folds = entry["folds"]
        values = [folds[f]["rmsle"] for f in ("B", "C", "D") if f in folds]
        fmt = lambda f: f"{folds[f]['rmsle']:.6f}" if f in folds else "—"
        nrounds = "/".join(str(folds[f]["rounds"]) if f in folds else "—" for f in ("B", "C", "D"))
        avg = f"{np.mean(values):.6f}" if values else "—"
        lines.append(f"| {name} | {fmt('B')} | {fmt('C')} | {fmt('D')} | {avg} | {nrounds} |")
    if "selection" in state:
        selected = dict(state["selection"])
        if selected.get("submission"):
            try:
                selected["submission"] = Path(selected["submission"]).relative_to(ROOT).as_posix()
            except ValueError:
                selected["submission"] = Path(selected["submission"]).name
        lines += ["", "## Cấu hình đã chọn", "", "```json",
                  json.dumps(selected, ensure_ascii=False, indent=2), "```"]
    lines += ["", "Trung bình trên B/C và B/C/D không có cùng số fold, không so trực tiếp. "
              "T02 dùng toàn bộ train để xác định ngày bán đầu và chuỗi toàn 0; "
              "dữ liệu dầu/khuyến mãi được dùng theo giả định biết trước của T03/T05.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "predictions" / "experiments" / "guided_results.json")
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "xgboost_guided_results.md")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(render(json.loads(a.input.read_text(encoding="utf-8"))), encoding="utf-8")
    print(a.output)
