"""
Script de demonstração: roda o pipeline em todas as imagens de data/raw/.

Para adicionar uma nova empresa: basta criar data/raw/<empresa>/company.yaml
Nenhuma alteração no código é necessária.

Uso:
    .venv/Scripts/python scripts/run_pipeline.py

Pré-requisito: rodar index_documents.py primeiro para popular o ChromaDB.

Logs JSON estruturados → logs/pipeline_YYYY-MM-DD_HH-MM-SS.log
Saída legível         → terminal (apenas resultados, sem JSON)
"""
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silencia output de YOLO e HuggingFace
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import yaml

# Importa os módulos app/ — cada um chama get_logger() que usa o config global
from app.pipeline.factory import create_pipeline
from app.logging.logger import configure_logging

# Redireciona TODOS os logs estruturados para arquivo
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
_log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
_log_file = open(_log_filename, "w", encoding="utf-8")
configure_logging(stream=_log_file)  # redireciona globalmente — afeta todos os loggers

DATA_ROOT = Path("data/raw")
CHROMA_PATH = "./chroma_db"


def _discover_companies(data_root: Path) -> list[dict]:
    companies = []
    for config_path in sorted(data_root.glob("*/company.yaml")):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["folder"] = config_path.parent
        companies.append(cfg)
    return companies


def main():
    companies = _discover_companies(DATA_ROOT)

    if not companies:
        print(f"Nenhum company.yaml encontrado em {DATA_ROOT}/")
        return

    print(f"Logs JSON → {_log_filename}")
    print(f"Carregando pipeline...\n")
    pipeline = create_pipeline(chroma_path=CHROMA_PATH)

    for company in companies:
        empresa = company["empresa"]
        setor = company["setor"]
        images_dir = company["folder"] / company["images_folder"]

        if not images_dir.exists():
            print(f"[AVISO] Pasta de imagens não encontrada: {images_dir}")
            continue

        images = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if not images:
            print(f"[AVISO] Nenhuma imagem em: {images_dir}")
            continue

        print(f"{'='*60}")
        print(f"Empresa: {empresa} | Setor: {setor}")
        print(f"{'='*60}")

        for image_path in images:
            print(f"\nImagem: {image_path.name}")
            try:
                response = pipeline.run(str(image_path), empresa=empresa, setor=setor)

                if not response.results:
                    print("  Nenhuma pessoa detectada.")
                    continue

                for result in response.results:
                    icon = "✓" if result.status == "Conforme" else "✗" if result.status == "Não conforme" else "?"
                    print(f"  [{icon}] Pessoa {result.pessoa_id}: {result.status}")
                    print(f"      {result.justificativa}")

            except Exception as exc:
                print(f"  [ERRO] {exc}")

        print()

    _log_file.close()
    print(f"\nPipeline finalizado. Logs em: {_log_filename}")


if __name__ == "__main__":
    main()
