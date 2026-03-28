"""
Script de demonstração: roda o pipeline em todas as imagens de data/raw/.

Para adicionar uma nova empresa: basta criar data/raw/<empresa>/company.yaml
Nenhuma alteração no código é necessária.

Para customizar os EPIs de uma empresa, adicione a seção `epi_attributes` ao
company.yaml. Se omitida, usa DEFAULT_EPI_ATTRIBUTES (4 EPIs padrão).

Uso:
    .venv/Scripts/python scripts/run_pipeline.py
    .venv/Scripts/python scripts/run_pipeline.py --clip-model models/clip_ppe

Pré-requisito: rodar index_documents.py primeiro para popular o ChromaDB.

Saídas:
  results/<empresa>/<imagem>.json  → JSON do PipelineResponse por imagem
  logs/pipeline_YYYY-MM-DD.log     → logs JSON estruturados (structlog)
  terminal                         → resumo legível dos resultados
"""
import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import yaml

from app.pipeline.factory import create_pipeline
from app.logging.logger import configure_logging
from app.schemas.epi_config import DEFAULT_EPI_ATTRIBUTES, EPIAttribute

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
_log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
_log_file = open(_log_filename, "w", encoding="utf-8")
configure_logging(stream=_log_file)

DATA_ROOT = Path("data/raw")
CHROMA_PATH = "./chroma_db"
RESULTS_DIR = Path("results")


def _discover_companies(data_root: Path) -> list[dict]:
    companies = []
    for config_path in sorted(data_root.glob("*/company.yaml")):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["folder"] = config_path.parent
        companies.append(cfg)
    return companies


def _load_epi_attributes(cfg: dict) -> list:
    """Carrega epi_attributes do company.yaml ou usa o padrão."""
    raw = cfg.get("epi_attributes")
    if not raw:
        return DEFAULT_EPI_ATTRIBUTES
    return [EPIAttribute(**item) for item in raw]


def _save_result(empresa: str, image_path: Path, response) -> Path:
    """Serializa o PipelineResponse como JSON e salva em results/<empresa>/."""
    import re
    empresa_slug = re.sub(r"[^a-z0-9]+", "_", empresa.lower()).strip("_")
    out_dir = RESULTS_DIR / empresa_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{image_path.stem}.json"
    payload = {
        "image": image_path.name,
        "empresa": empresa,
        "timestamp": datetime.now().isoformat(),
        "response": json.loads(response.model_dump_json()),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _process_image(pipeline, image_path: Path, empresa: str, setor: str) -> None:
    """Roda o pipeline em uma imagem e imprime o resultado no terminal."""
    print(f"\nImagem: {image_path.name}")
    try:
        response = pipeline.run(str(image_path), empresa=empresa, setor=setor)
        out_path = _save_result(empresa, image_path, response)
        print(f"  → {out_path}")

        if not response.results:
            print("  Nenhuma pessoa detectada.")
            return

        for result in response.results:
            icon = "✓" if result.status == "Conforme" else "✗" if result.status == "Não conforme" else "?"
            print(f"  [{icon}] Pessoa {result.pessoa_id}: {result.status}")
            print(f"      {result.justificativa}")

    except Exception as exc:
        print(f"  [ERRO] {exc}")


def _process_company(company: dict, clip_model_path) -> None:
    """Carrega o pipeline para uma empresa e processa todas as suas imagens."""
    empresa    = company["empresa"]
    setor      = company["setor"]
    images_dir = company["folder"] / company["images_folder"]

    if not images_dir.exists():
        print(f"[AVISO] Pasta de imagens não encontrada: {images_dir}")
        return

    images = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not images:
        print(f"[AVISO] Nenhuma imagem em: {images_dir}")
        return

    epi_attributes = _load_epi_attributes(company)
    pipeline = create_pipeline(
        chroma_path=CHROMA_PATH,
        epi_attributes=epi_attributes,
        clip_model_path=clip_model_path,
    )

    print(f"{'='*60}")
    print(f"Empresa: {empresa} | Setor: {setor}")
    print(f"EPIs configurados: {[e.name for e in epi_attributes]}")
    print(f"{'='*60}")

    for image_path in images:
        _process_image(pipeline, image_path, empresa, setor)
    print()


def main():
    parser = argparse.ArgumentParser(description="Roda o pipeline de verificação de EPIs")
    parser.add_argument(
        "--clip-model",
        default=None,
        metavar="PATH",
        help="Caminho para modelo CLIP fine-tunado (ex: models/clip_ppe). "
             "Se omitido, usa o modelo base OpenAI.",
    )
    args = parser.parse_args()

    companies = _discover_companies(DATA_ROOT)
    if not companies:
        print(f"Nenhum company.yaml encontrado em {DATA_ROOT}/")
        return

    print(f"Logs      → {_log_filename}")
    print(f"Resultados → {RESULTS_DIR}/")
    if args.clip_model:
        print(f"CLIP model → {args.clip_model}")
    print("Carregando pipeline...\n")

    for company in companies:
        _process_company(company, args.clip_model)

    _log_file.close()
    print("Pipeline finalizado.")
    print(f"  Resultados JSON: {RESULTS_DIR}/")
    print(f"  Logs:            {_log_filename}")


if __name__ == "__main__":
    main()
