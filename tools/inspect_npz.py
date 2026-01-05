"""
Inspecte la structure d'un fichier .npz pour comprendre son contenu.
"""

import numpy as np
import sys

def inspect_npz(file_path):
    """Affiche la structure d'un fichier .npz."""
    print(f"="*80)
    print(f"INSPECTION: {file_path}")
    print(f"="*80)

    data = np.load(file_path, allow_pickle=True)

    print(f"\n📦 Clés disponibles:")
    for key in data.keys():
        print(f"   - {key}")

    print(f"\n📊 Détails par clé:")
    for key in data.keys():
        value = data[key]

        if isinstance(value, np.ndarray):
            print(f"\n   {key}:")
            print(f"      Type: numpy.ndarray")
            print(f"      Shape: {value.shape}")
            print(f"      Dtype: {value.dtype}")
            if value.size < 10:
                print(f"      Valeurs: {value}")
            else:
                print(f"      Min: {value.min() if np.issubdtype(value.dtype, np.number) else 'N/A'}")
                print(f"      Max: {value.max() if np.issubdtype(value.dtype, np.number) else 'N/A'}")
                print(f"      Premiers éléments: {value.flat[:3]}")
        else:
            print(f"\n   {key}:")
            print(f"      Type: {type(value)}")
            print(f"      Valeur: {value}")

    data.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <file.npz>")
        sys.exit(1)

    inspect_npz(sys.argv[1])
