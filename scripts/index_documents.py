"""
Script de indexação: lê todos os documentos de data/raw e indexa no ChromaDB.

Uso:
    .venv/Scripts/python scripts/index_documents.py

O ChromaDB é criado em ./chroma_db (persistente, uma collection por empresa).
Só precisa rodar uma vez — ou quando os documentos mudam.
"""
import sys
from pathlib import Path

# Garante que app/ é encontrado independente de onde o script é chamado
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from app.rag.document_parser import DocumentParser
from app.rag.embedding_service import EmbeddingService
from app.rag.ollama_embedder import OllamaEmbedder

# ---------------------------------------------------------------------------
# Mapeamento: pasta → (empresa, setor, arquivo do documento)
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw")
CHROMA_PATH = "./chroma_db"

COMPANIES = [
    {
        "folder": "Construtiva Engenharia S.A_",
        "empresa": "Construtiva Engenharia",
        "setor": "obras",
        "doc": "Construtiva_Engenharia_Manual_Completo_Texto_Rendido.pdf",
    },
    {
        "folder": "LogiTrans Global S.A_",
        "empresa": "LogiTrans Global",
        "setor": "logistica",
        "doc": "Manual_Integracao_LogiTrans_v2023.pdf",
    },
    {
        "folder": "Rede_Vitalis",
        "empresa": "Rede Vitalis",
        "setor": "saude",
        "doc": "Guia_Integracao_Rede_Vitalis_2024.docx",
    },
    {
        "folder": "VITALCARE SERVIÇOS DE SAÚDE INTEGRADOS S.A_",
        "empresa": "VitalCare",
        "setor": "saude",
        "doc": "VitalCare_Manual_Institucional_Completo.pdf",
    },
]


def main():
    print("Iniciando indexação de documentos...\n")

    parser = DocumentParser()
    embedder = OllamaEmbedder(model="nomic-embed-text")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_service = EmbeddingService(
        chroma_client=chroma_client,
        embedding_client=embedder,
    )

    total_chunks = 0

    for company in COMPANIES:
        doc_path = DATA_ROOT / company["folder"] / company["doc"]
        empresa = company["empresa"]
        setor = company["setor"]

        if not doc_path.exists():
            print(f"  [AVISO] Arquivo não encontrado: {doc_path}")
            continue

        print(f"Indexando: {empresa} ({setor})")
        print(f"  Arquivo: {doc_path.name}")

        try:
            chunks = parser.parse(str(doc_path), empresa=empresa, setor=setor)
            print(f"  Chunks extraídos: {len(chunks)}")

            if not chunks:
                print(f"  [AVISO] Nenhum chunk extraído — documento pode estar vazio ou muito curto")
                continue

            # collection_name é gerado automaticamente dentro do EmbeddingService
            # mas aqui precisamos passá-lo explicitamente
            import re
            collection = re.sub(r"[^a-z0-9]+", "_", empresa.lower()).strip("_")
            embedding_service.index(chunks, collection=collection)

            print(f"  Indexado na collection: '{collection}'")
            total_chunks += len(chunks)

        except Exception as exc:
            print(f"  [ERRO] {empresa}: {exc}")

        print()

    print(f"Indexação concluída. Total de chunks indexados: {total_chunks}")
    print(f"ChromaDB persistido em: {CHROMA_PATH}/")


if __name__ == "__main__":
    main()
