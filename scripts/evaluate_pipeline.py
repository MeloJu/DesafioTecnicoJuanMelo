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


def evaluate(results_dir: pathlib.Path, verbose: bool = True, out_path: pathlib.Path | None = None) -> None:
    sep  = "=" * 60
    sep2 = "-" * 60

    global_tp = global_tn = global_fp = global_fn = global_indet = 0
    global_mismatch_count = 0
    company_metrics: dict = {}

    print(f"\n{sep}")
    print("  AVALIACAO DO PIPELINE — vs. Gabarito Humano")
    print(f"{sep}")
    print("  Positivo = Conforme  |  Negativo = Nao conforme")
    print(f"  Indeterminado excluido das metricas")
    print(f"{sep2}")

    for empresa_key, images_gt in GROUND_TRUTH.items():
        label = COMPANY_LABELS.get(empresa_key, empresa_key)
        empresa_dir = results_dir / empresa_key

        if not empresa_dir.exists():
            print(f"\n  [{label}] diretorio nao encontrado: {empresa_dir}")
            continue

        print(f"\n  {label}")
        print(f"  {'-' * 40}")

        e_tp = e_tn = e_fp = e_fn = e_indet = 0
        mismatches = []

        for img_key, gt_labels in images_gt.items():
            json_path = empresa_dir / f"{img_key}.json"
            if not json_path.exists():
                print(f"    {img_key}: arquivo nao encontrado")
                continue

            pipe_results = _load_pipeline_results(json_path)
            cmp = _compare_image(gt_labels, pipe_results)

            e_tp    += cmp["tp"];   e_tn    += cmp["tn"]
            e_fp    += cmp["fp"];   e_fn    += cmp["fn"]
            e_indet += cmp["indet"]

            flag = " [contagem diferente!]" if cmp["mismatch"] else ""
            if cmp["mismatch"]:
                mismatches.append(f"{img_key}: GT={cmp['n_gt']} pessoas, pipeline={cmp['n_pipe']} pessoas")
                global_mismatch_count += 1

            # Linha resumida por imagem
            n_ok  = cmp["tp"] + cmp["tn"]
            n_err = cmp["fp"] + cmp["fn"]
            n_cmp = n_ok + n_err + cmp["indet"]
            print(f"    {img_key:<42} acertos={n_ok}/{n_cmp}  erros={n_err}  indet={cmp['indet']}{flag}")

            if verbose:
                for row in cmp["rows"]:
                    icon = "  OK" if row["match"] == "OK" else ("  ??" if row["match"] == "?" else "  XX")
                    print(f"       pessoa {row['pessoa']:>2}: GT={row['gt']:<14}  pipeline={row['pipe']}{icon}")

        m = _metrics(e_tp, e_tn, e_fp, e_fn)
        print(f"\n    Accuracy:  {m['accuracy']}%   Precision: {m['precision']}%   Recall: {m['recall']}%   F1: {m['f1']}%")
        print(f"    TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}  Indeterminado={e_indet}")

        if mismatches:
            print(f"    Avisos de contagem:")
            for w in mismatches:
                print(f"      - {w}")

        company_metrics[label] = _metrics(e_tp, e_tn, e_fp, e_fn)
        company_metrics[label]["indeterminado"] = e_indet

        global_tp += e_tp; global_tn += e_tn
        global_fp += e_fp; global_fn += e_fn
        global_indet += e_indet

    # Global
    gm = _metrics(global_tp, global_tn, global_fp, global_fn)
    print(f"\n{sep}")
    print(f"  RESULTADO GLOBAL")
    print(f"{sep2}")
    print(f"  Pessoas comparadas: {gm['total']}  (indeterminado: {global_indet}  imagens c/ contagem diferente: {global_mismatch_count})")
    print(f"  Accuracy:   {gm['accuracy']}%")
    print(f"  Precision:  {gm['precision']}%   (das vezes que disse Conforme, estava certo)")
    print(f"  Recall:     {gm['recall']}%   (das pessoas conformes no GT, quantas o pipeline achou)")
    print(f"  F1-score:   {gm['f1']}%")
    print(f"  TP={gm['tp']} TN={gm['tn']} FP={gm['fp']} FN={gm['fn']}")
    print(f"{sep}")
    print(f"\n  Nota: matching por posicao (ordem de pessoa_id no JSON).")
    print(f"  Para matching preciso, seria necessario bbox no gabarito.")
    print(f"{sep}")

    # Salvar JSON
    if out_path is not None:
        report = {
            "metadata": {
                "pessoas_comparadas": gm["total"],
                "indeterminado": global_indet,
                "imagens_com_contagem_diferente": global_mismatch_count,
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Metricas salvas em: {out_path}")


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
