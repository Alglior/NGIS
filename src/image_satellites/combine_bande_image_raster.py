import rasterio
from rasterio.merge import merge
import numpy as np

def combine_sentinel_bands(band_paths):
    # Liste pour stocker les données de chaque bande
    bands = []
    
    # Lire chaque bande
    for band_path in band_paths:
        with rasterio.open(band_path) as src:
            bands.append(src.read(1))
    
    # Empiler les bandes
    combined = np.stack(bands)
    
    # Obtenir les métadonnées de la première bande pour la référence
    with rasterio.open(band_paths[0]) as src:
        meta = src.meta.copy()
    
    # Mettre à jour les métadonnées pour l'image composite
    meta.update({
        'count': 4,  # Nombre de bandes
        'dtype': 'uint16'  # Type de données
    })
    
    # Sauvegarder l'image composite
    output_path = 'sentinel2_composite.tif'
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(combined)
    
    return output_path

# Liste des chemins des bandes (à adapter selon votre structure de fichiers)
band_paths = [
    'S2A_OPER_MSI_L1C_TL_SGS__20160826T160708_A006154_T31TFL_B02.jp2',
    'S2A_OPER_MSI_L1C_TL_SGS__20160826T160708_A006154_T31TFL_B03.jp2',
    'S2A_OPER_MSI_L1C_TL_SGS__20160826T160708_A006154_T31TFL_B04.jp2',
    'S2A_OPER_MSI_L1C_TL_SGS__20160826T160708_A006154_T31TFL_B08.jp2'
]

# Exécuter la combinaison
result_path = combine_sentinel_bands(band_paths)
print(f"Image composite créée : {result_path}")