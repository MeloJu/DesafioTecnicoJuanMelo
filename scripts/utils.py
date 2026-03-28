"""
Utilitários compartilhados entre os scripts do pipeline.
"""
from pathlib import Path

import yaml


def discover_companies(data_root: Path) -> list[dict]:
    """Encontra todas as pastas com company.yaml em data/raw/."""
    companies = []
    for config_path in sorted(data_root.glob("*/company.yaml")):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["folder"] = config_path.parent
        companies.append(cfg)
    return companies
