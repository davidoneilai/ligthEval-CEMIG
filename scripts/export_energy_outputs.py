from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _find_latest_details_parquet(details_root: Path) -> Path:
    all_parquets = sorted(details_root.rglob("*.parquet"))
    candidates = [p for p in all_parquets if p.name.startswith("details_energy_eval|0_")]
    if not candidates:
        workspace_fallback = sorted(Path(".").rglob("*.parquet"))
        candidates = [
            p for p in workspace_fallback if p.name.startswith("details_energy_eval|0_")
        ]
    if not candidates:
        raise FileNotFoundError(
            "Nao encontrei arquivo details_energy_eval|0_*.parquet. "
            "Rode o benchmark com --save-details antes de exportar."
        )
    return candidates[-1]


def _safe_choice(choices: list[str], idx: int | None) -> str | None:
    if idx is None:
        return None
    if 0 <= idx < len(choices):
        return choices[idx]
    return None


def _argmax(values: list[float] | None) -> int | None:
    if not values:
        return None
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values[1:], start=1):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def _extract_timestamp_from_name(file_name: str) -> str:
    match = re.search(r"details_energy_eval\|0_(.+)\.parquet$", file_name)
    if match:
        return match.group(1)
    return "latest"


def _to_python_rows(parquet_file: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente: pyarrow. Rode este script dentro da imagem docker de avaliacao."
        ) from exc

    table = pq.read_table(parquet_file)
    return table.to_pylist()


def export_outputs(details_root: Path, out_dir: Path) -> Path:
    parquet_file = _find_latest_details_parquet(details_root)
    rows = _to_python_rows(parquet_file)

    run_id = _extract_timestamp_from_name(parquet_file.name)
    run_out_dir = out_dir / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    rows_jsonl = run_out_dir / "rows.jsonl"
    wrong_jsonl = run_out_dir / "wrong_only.jsonl"
    summary_json = run_out_dir / "summary.json"

    total = 0
    correct = 0

    with rows_jsonl.open("w", encoding="utf-8") as f_all, wrong_jsonl.open(
        "w", encoding="utf-8"
    ) as f_wrong:
        for i, row in enumerate(rows):
            doc = row.get("doc") or {}
            model_response = row.get("model_response") or {}
            metric = row.get("metric") or {}

            choices = doc.get("choices") or []
            gold_index = doc.get("gold_index")
            logprobs = model_response.get("logprobs")
            pred_index = _argmax(logprobs)

            gold_choice = _safe_choice(choices, gold_index)
            pred_choice = _safe_choice(choices, pred_index)
            is_correct = bool(pred_index is not None and gold_index == pred_index)

            output = {
                "row_id": i,
                "task_name": doc.get("task_name"),
                "question_prompt": doc.get("query"),
                "choices": choices,
                "gold_index": gold_index,
                "gold_choice": gold_choice,
                "pred_index": pred_index,
                "pred_choice": pred_choice,
                "is_correct": is_correct,
                "acc_metric_row": metric.get("acc"),
                "logprobs": logprobs,
                "model_text": model_response.get("text"),
                "model_text_post_processed": model_response.get("text_post_processed"),
            }

            total += 1
            correct += int(is_correct)

            line = json.dumps(output, ensure_ascii=False)
            f_all.write(line + "\n")
            if not is_correct:
                f_wrong.write(line + "\n")

    summary = {
        "source_parquet": str(parquet_file),
        "rows_total": total,
        "rows_correct": correct,
        "rows_wrong": total - correct,
        "acc_from_export": (correct / total) if total else None,
        "rows_file": str(rows_jsonl),
        "wrong_only_file": str(wrong_jsonl),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta outputs por linha a partir dos detalhes do LightEval"
    )
    parser.add_argument(
        "--details-root",
        default="test_energy_eval/details",
        help="Diretorio raiz com os arquivos de details parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Diretorio de saida para os arquivos exportados",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details_root = Path(args.details_root)
    out_dir = Path(args.out_dir)

    summary_file = export_outputs(details_root=details_root, out_dir=out_dir)
    print(f"Export concluido: {summary_file}")


if __name__ == "__main__":
    main()
