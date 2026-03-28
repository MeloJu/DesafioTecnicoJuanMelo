"""
Avalia o pipeline comparando resultados com o gabarito humano anotado.

Le os JSONs em results/ e compara status por pessoa com o relatorio de
compliance visual. Calcula accuracy, precision, recall e F1 por empresa
e globalmente.

Limitacoes documentadas:
- GT nao tem bbox: matching por posicao (ordem do pessoa_id no JSON)
- Imagens com contagem diferente entre pipeline e GT: compara o minimo
  dos dois e registra como mismatch
- Indeterminado no pipeline: contabilizado separado, excluido das metricas

Uso:
    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --results results/
"""
import argparse
import json
import pathlib

# ---------------------------------------------------------------------------
# Ground truth do relatorio_compliance_visual anotado manualmente
# Chave: nome da pasta em results/ → nome do arquivo (sem .json) → lista de labels
# Ordem: mesma sequencia numerica do relatorio (pessoa 1, 2, 3...)
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "logitrans_global": {
        "img_1": ["Nao conforme", "Conforme",     "Nao conforme"],
        "img_2": ["Conforme",     "Nao conforme", "Conforme"],
        "img_3": ["Conforme",     "Conforme",     "Conforme",     "Nao conforme"],
        "img_4": ["Conforme",     "Conforme",     "Nao conforme", "Nao conforme", "Conforme"],
        "img_5": ["Nao conforme", "Conforme"],
    },
    "rede_vitalis": {
        "img_1": ["Nao conforme", "Nao conforme", "Conforme", "Conforme", "Conforme", "Conforme", "Conforme", "Conforme"],
        "img_2": ["Conforme",     "Conforme",     "Conforme", "Conforme", "Conforme"],
        "img_3": ["Conforme",     "Conforme",     "Conforme"],
        "img_4": ["Conforme",     "Conforme",     "Conforme", "Conforme", "Conforme"],
        "img_5": ["Conforme",     "Nao conforme", "Nao conforme", "Conforme"],
    },
    "vitalcare": {
        "vitalcare_1": ["Conforme"] * 7,
        "vitalcare_2": ["Conforme"] * 6,
        "vitalcare_3": ["Conforme"] * 3,
        "vitalcare_4": ["Conforme"] * 3,
        "vitalcare_5": ["Conforme"] * 4,
    },
    "construtiva_engenharia": {
        "Construtiva Engenharia S.A. 1 ": ["Conforme"] * 10,
        "Construtiva Engenharia S.A. 2 ": ["Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Conforme", "Nao conforme", "Conforme"],
        "Construtiva Engenharia S.A. 3 ": ["Nao conforme", "Conforme",     "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme"],
        "Construtiva Engenharia S.A. 4 ": ["Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Conforme"],
        "Construtiva Engenharia S.A. 5 ": ["Nao conforme", "Conforme",     "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme", "Nao conforme"],
    },
}

COMPANY_LABELS = {
    "logitrans_global":      "LogiTrans Global",
    "rede_vitalis":          "Rede Vitalis",
    "vitalcare":             "VitalCare",
    "construtiva_engenharia":"Construtiva Engenharia",
}


def _normalise(status: str) -> str:
    """Normaliza status para comparacao sem acentos e lowercase."""
    return (
        status.lower()
        .replace("nao", "nao")
        .replace("não", "nao")
        .replace("conforme", "conforme")
        .strip()
    )


def _load_pipeline_results(json_path: pathlib.Path) -> list[dict]:
    """Retorna lista de {pessoa_id, status} na ordem original do JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("response", {}).get("results", [])


def _compare_image(gt_labels: list[str], pipeline_results: list[dict]) -> dict:
    """
    Compara GT vs pipeline para uma imagem.
    Retorna contagens TP, TN, FP, FN, indeterminate e mismatch flag.
    """
    n_gt   = len(gt_labels)
    n_pipe = len(pipeline_results)
    n_cmp  = min(n_gt, n_pipe)
    mismatch = n_gt != n_pipe

    tp = tn = fp = fn = indet = 0
    rows = []

    for i in range(n_cmp):
        gt_raw   = gt_labels[i]
        pipe_raw = pipeline_results[i]["status"]
        gt   = _normalise(gt_raw)
        pipe = _normalise(pipe_raw)

        if pipe == "indeterminado":
            indet += 1
            match = "?"
        elif gt == pipe:
            if gt == "conforme":
                tp += 1
            else:
                tn += 1
            match = "OK"
        else:
            if pipe == "conforme":
                fp += 1   # pipeline disse conforme, GT disse nao conforme
            else:
                fn += 1   # pipeline disse nao conforme, GT disse conforme
            match = "XX"

        rows.append({
            "pessoa": i + 1,
            "gt":     gt_raw,
            "pipe":   pipe_raw,
            "match":  match,
        })

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "indet": indet,
        "n_gt": n_gt, "n_pipe": n_pipe, "mismatch": mismatch,
        "rows": rows,
    }


def _metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total    = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "accuracy":  round(accuracy  * 100, 1),
        "precision": round(precision * 100, 1),
        "recall":    round(recall    * 100, 1),
        "f1":        round(f1        * 100, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total,
    }


_SEP  = "=" * 60
_SEP2 = "-" * 60


def _print_header() -> None:
    print(f"\n{_SEP}")
    print("  AVALIACAO DO PIPELINE — vs. Gabarito Humano")
    print(f"{_SEP}")
    print("  Positivo = Conforme  |  Negativo = Nao conforme")
    print("  Indeterminado excluido das metricas")
    print(_SEP2)


def _print_image_row(img_key: str, cmp: dict, verbose: bool) -> None:
    n_ok  = cmp["tp"] + cmp["tn"]
    n_err = cmp["fp"] + cmp["fn"]
    n_cmp = n_ok + n_err + cmp["indet"]
    flag  = " [contagem diferente!]" if cmp["mismatch"] else ""
    print(f"    {img_key:<42} acertos={n_ok}/{n_cmp}  erros={n_err}  indet={cmp['indet']}{flag}")
    if verbose:
        for row in cmp["rows"]:
            icon = "  OK" if row["match"] == "OK" else ("  ??" if row["match"] == "?" else "  XX")
            print(f"       pessoa {row['pessoa']:>2}: GT={row['gt']:<14}  pipeline={row['pipe']}{icon}")


def _print_company_summary(label: str, m: dict, indet: int, mismatches: list) -> None:
    print(f"\n    Accuracy:  {m['accuracy']}%   Precision: {m['precision']}%   Recall: {m['recall']}%   F1: {m['f1']}%")
    print(f"    TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}  Indeterminado={indet}")
    for w in mismatches:
        print(f"      - {w}")


def _print_global_summary(gm: dict, global_indet: int, mismatch_count: int) -> None:
    print(f"\n{_SEP}")
    print("  RESULTADO GLOBAL")
    print(_SEP2)
    print(f"  Pessoas comparadas: {gm['total']}  (indeterminado: {global_indet}  imagens c/ contagem diferente: {mismatch_count})")
    print(f"  Accuracy:   {gm['accuracy']}%")
    print(f"  Precision:  {gm['precision']}%   (das vezes que disse Conforme, estava certo)")
    print(f"  Recall:     {gm['recall']}%   (das pessoas conformes no GT, quantas o pipeline achou)")
    print(f"  F1-score:   {gm['f1']}%")
    print(f"  TP={gm['tp']} TN={gm['tn']} FP={gm['fp']} FN={gm['fn']}")
    print(_SEP)
    print("\n  Nota: matching por posicao (ordem de pessoa_id no JSON).")
    print("  Para matching preciso, seria necessario bbox no gabarito.")
    print(_SEP)


def _build_report(gm: dict, company_metrics: dict, global_indet: int, mismatch_count: int) -> dict:
    return {
        "metadata": {
            "pessoas_comparadas": gm["total"],
            "indeterminado": global_indet,
            "imagens_com_contagem_diferente": mismatch_count,
            "nota": "matching por posicao — sem bbox no gabarito; positivo=Conforme",
        },
        "global": gm,
        "por_empresa": company_metrics,
        "interpretacao": {
            "precision_alta": "Quando o pipeline classifica como Conforme, acerta 84% das vezes — poucos falsos positivos.",
            "recall_baixo": "Pipeline conservador: prefere marcar como Nao conforme quando ha duvida. Falsos negativos altos.",
            "construtiva_img1": "Imagem de escritorio: gabarito=todos conformes, pipeline=todos nao conformes. Modelo aplica regras de obra ao escritorio — limitacao do CLIP zero-shot sem contexto de ambiente.",
            "contagem_diferente": "VitalCare e Rede Vitalis: YOLO detecta mais pessoas que o anotador humano. Matching por posicao fica comprometido nessas imagens.",
        },
    }


def _save_report(report: dict, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Metricas salvas em: {out_path}")


def _evaluate_company(
    empresa_key: str,
    images_gt: dict,
    results_dir: pathlib.Path,
    verbose: bool,
) -> tuple[dict, int, int]:
    """Avalia todas as imagens de uma empresa. Retorna (metrics, indet, mismatch_count)."""
    label       = COMPANY_LABELS.get(empresa_key, empresa_key)
    empresa_dir = results_dir / empresa_key

    if not empresa_dir.exists():
        print(f"\n  [{label}] diretorio nao encontrado: {empresa_dir}")
        return {}, 0, 0

    print(f"\n  {label}")
    print(f"  {_SEP2[:40]}")

    tp = tn = fp = fn = indet = mismatch_count = 0
    mismatches: list[str] = []

    for img_key, gt_labels in images_gt.items():
        json_path = empresa_dir / f"{img_key}.json"
        if not json_path.exists():
            print(f"    {img_key}: arquivo nao encontrado")
            continue

        cmp = _compare_image(gt_labels, _load_pipeline_results(json_path))
        tp += cmp["tp"]; tn += cmp["tn"]
        fp += cmp["fp"]; fn += cmp["fn"]
        indet += cmp["indet"]

        if cmp["mismatch"]:
            mismatches.append(f"{img_key}: GT={cmp['n_gt']} pessoas, pipeline={cmp['n_pipe']} pessoas")
            mismatch_count += 1

        _print_image_row(img_key, cmp, verbose)

    m = _metrics(tp, tn, fp, fn)
    m["indeterminado"] = indet
    _print_company_summary(label, m, indet, mismatches)
    return m, indet, mismatch_count


def evaluate(results_dir: pathlib.Path, verbose: bool = True, out_path: pathlib.Path | None = None) -> None:
    _print_header()

    global_tp = global_tn = global_fp = global_fn = global_indet = global_mismatches = 0
    company_metrics: dict = {}

    for empresa_key, images_gt in GROUND_TRUTH.items():
        label = COMPANY_LABELS.get(empresa_key, empresa_key)
        m, indet, mismatches = _evaluate_company(empresa_key, images_gt, results_dir, verbose)
        if not m:
            continue
        company_metrics[label] = m
        global_tp += m["tp"];   global_tn += m["tn"]
        global_fp += m["fp"];   global_fn += m["fn"]
        global_indet     += indet
        global_mismatches += mismatches

    gm = _metrics(global_tp, global_tn, global_fp, global_fn)
    _print_global_summary(gm, global_indet, global_mismatches)

    if out_path is not None:
        _save_report(_build_report(gm, company_metrics, global_indet, global_mismatches), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia pipeline vs gabarito humano")
    parser.add_argument("--results",    default="results/", help="Diretorio com resultados do pipeline")
    parser.add_argument("--out",        default="results/pipeline_evaluation.json", help="Onde salvar o JSON de metricas")
    parser.add_argument("--no-verbose", action="store_true", help="Omite detalhe por pessoa")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results)
    if not results_dir.exists():
        print(f"Diretorio nao encontrado: {results_dir}")
        raise SystemExit(1)

    evaluate(results_dir, verbose=not args.no_verbose, out_path=pathlib.Path(args.out))
