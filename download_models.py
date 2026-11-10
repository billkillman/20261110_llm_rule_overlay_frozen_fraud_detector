"""Télécharge les poids des modèles de l'expérience, aux révisions exactes utilisées pour les résultats publiés.

À lancer sur un poste connecté à Internet ; copier ensuite le dossier dans l'environnement cible et le désigner par
LOCAL_MODELS_DIR : le notebook charge alors les modèles sans aucun accès réseau.

Usage :
    pip install huggingface_hub
    python download_models.py DOSSIER [--scale reduced|full]
"""
import argparse
from pathlib import Path

# Modèles chargés à chaque échelle (SCALE dans le notebook).
MODELS = {'reduced': ['Qwen/Qwen3-0.6B'], 'full': ['HuggingFaceTB/SmolLM3-3B', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen3-1.7B']}
# Révision (commit Hugging Face) de chaque modèle : les mêmes poids que pour les résultats publiés.
REVISIONS = {'Qwen/Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca', 'HuggingFaceTB/SmolLM3-3B': 'a07cc9a04f16550a088caea529712d1d335b0ac1', 'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'}


def main() -> None:
    """Télécharge chaque modèle de l'échelle demandée dans DOSSIER/org--nom."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="dossier de destination")
    parser.add_argument("--scale", choices=sorted(MODELS), default="full", help="échelle (défaut : full)")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for model_id in MODELS[args.scale]:
        target = Path(args.folder) / model_id.replace("/", "--")
        snapshot_download(model_id, revision=REVISIONS.get(model_id), local_dir=target)
        print(f"{model_id} -> {target}")


if __name__ == "__main__":
    main()
