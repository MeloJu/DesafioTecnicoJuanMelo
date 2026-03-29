"""
Agrega todos os resultados em results/ e calcula métricas descritivas do pipeline.

Uso:
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --results results/ --out results/metrics.json
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

STATUS_CONFORME = "Conforme"
STATUS_NAO_CONF = "Não conforme"
STATUS_INDET    = "Indeterminado"
ALL_STATUS = [STATUS_CONFORME, STATUS_NAO_CONF, STATUS_INDET]


def load_results(results_dir: pathlib.Path) -> list[dict]:
    records = []
    for json_file in sorted(results_dir.rglob("*.json")):
        if json_file.name == "metrics.json":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for person in data.get("response", {}).get("results", []):
                records.append({
                    "empresa": data.get("empresa", json_file.parent.name),
                    "image": data.get("image", json_file.stem),
                    "pessoa_id": person.get("pessoa_id"),
                    "status": person.get("status", STATUS_INDET),
                })
        except json.JSONDecodeError as exc:
            print(f"[AVISO] JSON inválido em {json_file}: {exc}", file=sys.stderr)
        except OSError as exc:
            print(f"[AVISO] Erro ao ler {json_file}: {exc}", file=sys.stderr)
    return records


def _stats(subset: list[dict]) -> dict:
    total = len(subset)
    counts = {s: sum(1 for r in subset if r["status"] == s) for s in ALL_STATUS}
    pct    = {s: round(counts[s] / total * 100, 1) if total else 0.0 for s in ALL_STATUS}
    images = len({r["image"] for r in subset})
    return {"total": total, "images": images, "counts": counts, "pct": pct}


def compute(records: list[dict]) -> dict:
    by_empresa: dict[str, list] = defaultdict(list)
    for r in records:
        by_empresa[r["empresa"]].append(r)

    return {
        "global": _stats(records),
        "por_empresa": {
            emp: _stats(recs) for emp, recs in sorted(by_empresa.items())
        },
    }


def print_report(metrics: dict) -> None:
    g = metrics["global"]
    sep  = "=" * 54
    thin = "-" * 54

    print(sep)
    print(" COMPLIANCE AI - Relatorio de Metricas")
    print(sep)
    print(f"\nGLOBAL  ({g['images']} imagens | {len(metrics['por_empresa'])} empresas)")
    print(f"  {'Total de pessoas analisadas:':<30} {g['total']}")
    for s in ALL_STATUS:
        label = s + ":"
        print(f"  {label:<30} {g['counts'][s]:>3}  ({g['pct'][s]}%)")

    print(f"\n{thin}")
    print("POR EMPRESA\n")
    for emp, e in metrics["por_empresa"].items():
        print(f"  {emp}  ({e['images']} imagens | {e['total']} pessoas)")
        for s in ALL_STATUS:
            label = s + ":"
            print(f"    {label:<28} {e['counts'][s]:>3}  ({e['pct'][s]}%)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula métricas descritivas do pipeline")
    parser.add_argument("--results", default="results", help="Diretório de resultados")
    parser.add_argument("--out", default="results/metrics.json", help="Arquivo de saída JSON")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results)
    if not results_dir.exists():
        print(f"Diretório não encontrado: {results_dir}")
        print("Execute primeiro: python scripts/run_pipeline.py")
        raise SystemExit(1)

    records = load_results(results_dir)
    if not records:
        print("Nenhum resultado encontrado em", results_dir)
        raise SystemExit(1)

    metrics = compute(records)
    print_report(metrics)

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório salvo em: {out}")


if __name__ == "__main__":
    main()
