import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_ndvi(red_band_path, nir_band_path, output_path=None):
    # Set default output path if none provided
    if output_path is None:
        output_path = 'ndvi_result.tif'  # Changed to .tif extension
    
    try:
        # Ouvrir les bandes rouge et proche infrarouge
        with rasterio.open(red_band_path) as red:
            red_band = red.read(1).astype(float)
            meta = red.meta.copy()
        
        with rasterio.open(nir_band_path) as nir:
            nir_band = nir.read(1).astype(float)
        
        # Éviter la division par zéro en créant un masque
        mask = (red_band + nir_band) != 0
        
        # Initialiser le NDVI avec des zéros
        ndvi = np.zeros(red_band.shape)
        
        # Calculer le NDVI où le masque est valide
        ndvi[mask] = (nir_band[mask] - red_band[mask]) / (nir_band[mask] + red_band[mask])
        
        # Scale NDVI to int16 range (-10000 to 10000) for better storage
        ndvi_scaled = (ndvi * 10000).astype(np.int16)
        
        # Mettre à jour les métadonnées pour le fichier NDVI
        meta.update({
            'driver': 'GTiff',
            'dtype': 'int16',
            'count': 1,
            'nodata': -32768
        })
        
        # Sauvegarder le NDVI
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(ndvi_scaled, 1)
        
        # Créer une visualisation
        plt.figure(figsize=(10, 6), dpi=300)  # Taille modérée avec haute résolution
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)  # Palette Rouge-Jaune-Vert plus adaptée pour NDVI
        plt.colorbar(label='NDVI')
        plt.title('Indice de végétation NDVI')
        
        # Ajouter une grille et ajuster le style
        plt.grid(False)
        plt.tight_layout()
        
        # Save visualization with same basename as output
        viz_path = os.path.splitext(output_path)[0] + '_viz.png'
        plt.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"Erreur lors du calcul du NDVI: {str(e)}")
        raise

# Chemins des bandes (à adapter selon vos fichiers)
red_band = 'S2A_OPER_MSI_L1C_TL_SGS__20160826T160708_A006154_T31TFL_B04.jp2'
nir_band = 'S2A_OPER_MSI_L1C_TL_SGS__20160826T160708_A006154_T31TFL_B08.jp2'

# Calculer le NDVI
ndvi_path = calculate_ndvi(red_band, nir_band)
print(f"NDVI calculé et sauvegardé : {ndvi_path}")