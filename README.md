# Compliance AI — EPI Verification System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Tests](https://img.shields.io/badge/tests-175%20passing-brightgreen?logo=pytest)
![Coverage](https://img.shields.io/badge/coverage-100%25%20unit-brightgreen)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> Pipeline multimodal de IA para verificação automática de EPIs (Equipamentos de Proteção Individual) em imagens, com raciocínio baseado nas regras documentais de cada empresa.

---

## Stack

| Componente | Tecnologia |
|---|---|
| Detecção de pessoas | YOLO v8 (Ultralytics) |
| Classificação de EPIs | CLIP zero-shot (HuggingFace) |
| Embeddings | nomic-embed-text via Ollama |
| Banco vetorial | ChromaDB |
| LLM (raciocínio) | Llama 3.2 via Ollama |
| Schemas | Pydantic v2 |
| Logging | structlog (JSON + correlation_id) |
| Testes | pytest · 175 testes · 100% cobertura unitária |

**Princípio:** nenhuma API externa, nenhum custo por chamada, nenhum dado sai da máquina.

---

## Arquitetura

```
Imagem + Empresa
       │
       ▼
┌──────────────────┐
│  VisionService   │  YOLO → detecta pessoas
│  YOLO + CLIP     │  CLIP → classifica EPIs (capacete, colete, botas, luvas)
└────────┬─────────┘
         │ List[PersonDetection]
         ▼
┌──────────────────┐
│   RagService     │  busca semântica nos PDFs da empresa
│  ChromaDB +      │  retorna trechos de regras relevantes
│  nomic-embed     │
└────────┬─────────┘
         │ List[Rule]
         ▼
┌──────────────────┐
│ ReasoningService │  Llama analisa atributos vs. regras
│  Llama 3.2       │  gera status + justificativa por pessoa
└────────┬─────────┘
         │
         ▼
  results/<empresa>/<imagem>.json
```

---

## Quick Start — Docker (recomendado)

```bash
# 1. Build
docker compose build

# 2. Subir o Ollama
docker compose up ollama -d

# 3. Baixar modelos (primeira vez)
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text

# 4. Indexar documentos das empresas
docker compose run --rm app python scripts/index_documents.py

# 5. Rodar o pipeline
docker compose run --rm app python scripts/run_pipeline.py
```

Resultados em `results/` · Logs em `logs/` (volumes montados no host).

---

## Quick Start — Local

```bash
# 1. Ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 2. Dependências
pip install -r requirements.txt

# 3. Ollama
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text

# 4. Indexar + rodar
python scripts/index_documents.py
python scripts/run_pipeline.py
```

---

## Estrutura de dados (`data/`)

```
data/raw/
└── <NomeDaEmpresa>/
    ├── company.yaml       ← configuração da empresa
    ├── manual.pdf         ← documento de regras
    └── images/            ← fotos para análise
```

**`company.yaml` mínimo:**
```yaml
empresa: Construtiva Engenharia
setor: obras
doc: manual.pdf
images_folder: images
```

**Override de EPIs por empresa (opcional):**
```yaml
empresa: Lab Clean
setor: laboratorio
doc: manual.pdf
images_folder: images
epi_attributes:
  - name: lab_coat
    label_pt: jaleco
    clip_positive: "a person wearing a lab coat"
    clip_negative: "a person not wearing a lab coat"
    threshold_positive: 0.65
    threshold_negative: 0.35
  - name: gloves
    label_pt: luvas
    clip_positive: "a person wearing protective gloves"
    clip_negative: "a person not wearing gloves"
```

Se `epi_attributes` for omitido, usa os 4 EPIs padrão: capacete, colete refletivo, botas de segurança, luvas.

---

## Saída

Cada imagem gera `results/<empresa>/<imagem>.json`:

```json
{
  "results": [
    {
      "pessoa_id": 1,
      "bbox": { "x1": 45.0, "y1": 12.0, "x2": 180.0, "y2": 410.0 },
      "status": "Não conforme",
      "justificativa": "Capacete ausente — viola NR-18 item 18.23.1 exigido para área de obras."
    }
  ]
}
```

---

## Testes

```bash
# Unitários (sem modelos reais, rápido)
pytest tests/unit/ --cov=app

# Integração (lógica entre módulos, sem modelos reais)
pytest tests/integration/

# E2E (requer Ollama rodando + dados reais em data/)
pytest tests/e2e/ -m e2e -s

# Dentro do Docker
docker compose run --rm app python -m pytest tests/unit/ tests/integration/ -q
```

| Camada | Testes |
|---|---|
| Schemas (Pydantic) | 37 unitários |
| Logging (structlog) | 22 unitários |
| RAG (ChromaDB + nomic) | 34 unitários |
| Vision (YOLO + CLIP) | 24 unitários |
| Reasoning (Llama) | 16 unitários |
| Pipeline (orquestrador) | 16 unitários |
| Integração | 10 testes |
| E2E | 2 testes (requerem ambiente real) |

---

## Estrutura do projeto

```
project/
├── app/
│   ├── schemas/
│   │   ├── output.py          ← BoundingBox, PersonDetection, Rule, PersonResult
│   │   └── epi_config.py      ← EPIAttribute dataclass + DEFAULT_EPI_ATTRIBUTES
│   ├── logging/
│   │   └── logger.py          ← structlog JSON + correlation_id
│   ├── pipeline/
│   │   ├── orchestrator.py    ← Pipeline.run()
│   │   └── factory.py         ← create_pipeline() (composição de dependências)
│   ├── vision/
│   │   ├── detector.py        ← PersonDetector (YOLO)
│   │   ├── extractor.py       ← AttributeExtractor (CLIP + EPIAttribute)
│   │   ├── service.py         ← VisionService
│   │   └── clip_client.py     ← wrapper HuggingFace CLIP
│   ├── rag/
│   │   ├── document_parser.py ← PDF/DOCX → List[Chunk]
│   │   ├── embedding_service.py ← ChromaDB index + query
│   │   ├── service.py         ← RagService
│   │   └── ollama_embedder.py ← wrapper nomic-embed-text
│   └── reasoning/
│       ├── service.py         ← ReasoningService
│       └── ollama_llm.py      ← wrapper Llama via Ollama
├── scripts/
│   ├── index_documents.py     ← indexa PDFs no ChromaDB
│   └── run_pipeline.py        ← roda o pipeline para todas as empresas em data/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Decisões técnicas

| Decisão | Escolha | Alternativa rejeitada |
|---|---|---|
| Classificação de EPIs | CLIP zero-shot | Classificador supervisionado — exigiria dataset rotulado por tipo de EPI |
| LLM | Llama 3.2 local (Ollama) | OpenAI API — custo por chamada + dados saem da máquina |
| Banco vetorial | ChromaDB (arquivo local) | Weaviate/Pinecone — infraestrutura adicional desnecessária |
| RAG vs. contexto completo | RAG (top-k chunks) | Passar o PDF inteiro ao LLM — lento e caro em tokens |
| Configuração de EPIs | `EPIAttribute` dataclass via DI | Hardcoding por atributo — impede extensão sem modificar código |
