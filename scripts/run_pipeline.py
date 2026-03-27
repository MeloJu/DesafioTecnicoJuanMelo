"""
Script de demonstração: roda o pipeline em todas as imagens de data/raw.

Uso:
    .venv/Scripts/python scripts/run_pipeline.py

Pré-requisito: rodar index_documents.py primeiro para popular o ChromaDB.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.factory import create_pipeline

DATA_ROOT = Path("data/raw")
CHROMA_PATH = "./chroma_db"

# Mapeamento pasta → (empresa, setor)
COMPANIES = [
    {
        "folder": "Construtiva Engenharia S.A_",
        "images_folder": "images",
        "empresa": "Construtiva Engenharia",
        "setor": "obras",
    },
    {
        "folder": "LogiTrans Global S.A_",
        "images_folder": "imagens",
        "empresa": "LogiTrans Global",
        "setor": "logistica",
    },
    {
        "folder": "Rede_Vitalis",
        "images_folder": "imagens",
        "empresa": "Rede Vitalis",
        "setor": "saude",
    },
    {
        "folder": "VITALCARE SERVIÇOS DE SAÚDE INTEGRADOS S.A_",
        "images_folder": "images",
        "empresa": "VitalCare",
        "setor": "saude",
    },
]


def main():
    print("Carregando pipeline...\n")
    pipeline = create_pipeline(chroma_path=CHROMA_PATH)

    for company in COMPANIES:
        images_dir = DATA_ROOT / company["folder"] / company["images_folder"]
        empresa = company["empresa"]
        setor = company["setor"]

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
                    print("  Nenhuma pessoa detectada na imagem.")
                    continue

                for result in response.results:
                    status_icon = "✓" if result.status == "Conforme" else "✗" if result.status == "Não conforme" else "?"
                    print(f"  [{status_icon}] Pessoa {result.pessoa_id}: {result.status}")
                    print(f"      Justificativa: {result.justificativa}")
                    bbox = result.bbox
                    print(f"      BBox: ({bbox.x1:.0f},{bbox.y1:.0f}) → ({bbox.x2:.0f},{bbox.y2:.0f})")

            except Exception as exc:
                print(f"  [ERRO] {exc}")

        print()

    print("Pipeline finalizado.")


if __name__ == "__main__":
    main()
