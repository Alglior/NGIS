import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import pandas as pd

def read_geodata(file_path):
    """
    Read various types of geographic data files (GeoPackage, Shapefile, GeoJSON)
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    try:
        if extension == '.gpkg':
            layers = gpd.read_file(file_path, layer=None)
            if len(layers) > 1:
                print(f"Available layers: {layers}")
                layer_name = input("Multiple layers found. Please specify layer name: ")
                gdf = gpd.read_file(file_path, layer=layer_name)
            else:
                gdf = gpd.read_file(file_path)
        elif extension in ['.shp', '.geojson']:
            gdf = gpd.read_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
            
        if gdf.empty:
            raise ValueError("The file contains no data")
            
        print(f"Successfully loaded {len(gdf)} features from {os.path.basename(file_path)}")
        return gdf
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def process_chunk(chunk_data):
    """
    Process a chunk of points to calculate distances to all other points
    """
    chunk, all_points_array, start_idx = chunk_data
    distance_matrix = []
    
    # Convert all points to numpy array once
    all_coords = np.array([(p.x, p.y) for p in all_points_array])
    
    for i, (point_idx, point_geom) in enumerate(chunk.items()):
        # Vectorized distance calculation using broadcasting
        point_coords = np.array([point_geom.x, point_geom.y])
        dists = np.sqrt(np.sum((all_coords - point_coords) ** 2, axis=1))
        distance_matrix.append((point_idx, dists))
        
    return distance_matrix

def calculate_distance_matrix(gdf):
    """
    Calculate the full distance matrix between all points
    """
    if not all(geom.geom_type == 'Point' for geom in gdf.geometry):
        raise ValueError("All geometries must be points")
    
    n_cores = cpu_count()
    total_points = len(gdf)
    
    # Use larger chunks for matrix calculation
    chunk_size = max(50, total_points // n_cores)
    
    # Prepare chunks
    chunks = []
    for i in range(0, total_points, chunk_size):
        chunk = gdf.geometry[i:i + chunk_size]
        chunks.append((chunk, gdf.geometry, i))
    
    print(f"Calculating distances for {total_points} points using {n_cores} cores")
    print(f"Matrix size will be {total_points}x{total_points}")
    
    # Process chunks in parallel
    all_distances = []
    
    with Pool(processes=n_cores) as pool:
        with tqdm(total=total_points, desc="Calculating distance matrix") as pbar:
            for chunk_distances in pool.imap(process_chunk, chunks):
                all_distances.extend(chunk_distances)
                pbar.update(len(chunk_distances))
    
    # Convert to DataFrame
    print("Converting results to DataFrame...")
    matrix_dict = {idx: dists for idx, dists in all_distances}
    distance_df = pd.DataFrame(matrix_dict).sort_index()
    distance_df.index = distance_df.columns
    
    return distance_df

def save_matrix(df, output_path):
    """
    Save the distance matrix to a file
    """
    _, ext = os.path.splitext(output_path)
    if ext.lower() == '.csv':
        df.to_csv(output_path)
    else:
        df.to_pickle(output_path)
    print(f"Matrix saved to: {output_path}")

def create_matrix_html_report(df, layer_name, output_path=None):
    """
    Create an HTML report with the distance matrix and statistics
    
    Parameters:
    -----------
    df : DataFrame
        Distance matrix
    layer_name : str
        Name of the layer being processed
    output_path : str, optional
        Custom path for the HTML report
    """
    if output_path is None:
        output_path = f"matrice_{layer_name}_report.html"
    
    # Prepare matrix preview (first 10x10 elements or smaller)
    preview_size = min(10, len(df))
    matrix_preview = df.iloc[:preview_size, :preview_size].to_html(
        float_format=lambda x: f"{x:.2f}",
        classes="matrix-table"
    )
    
    html_content = f"""
    <html>
    <head>
        <title>Distance Matrix Report - {layer_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .matrix-table {{ font-size: 12px; }}
            .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Distance Matrix Report - {layer_name}</h1>
        
        <div class="stats">
            <h2>Matrix Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Matrix dimensions</td><td>{df.shape[0]} × {df.shape[1]}</td></tr>
                <tr><td>Mean distance</td><td>{df.mean().mean():.2f}</td></tr>
                <tr><td>Minimum distance</td><td>{df.min().min():.2f}</td></tr>
                <tr><td>Maximum distance</td><td>{df.max().max():.2f}</td></tr>
                <tr><td>Standard deviation</td><td>{df.std().std():.2f}</td></tr>
            </table>
        </div>

        <h2>Matrix Preview (first {preview_size}×{preview_size} elements)</h2>
        {matrix_preview}
        
        <p><small>Full matrix saved separately in CSV/Pickle format</small></p>
    </body>
    </html>
    """
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report saved to: {output_path}")
    except Exception as e:
        print(f"Error saving HTML report: {str(e)}")

def main():
    # Get input file path from user
    input_file = input("Enter path to your geographic file (GeoPackage, Shapefile, or GeoJSON): ")
    
    # Read the data
    gdf = read_geodata(input_file)
    if gdf is None:
        return
    
    try:
        # Get layer name from file path
        layer_name = os.path.splitext(os.path.basename(input_file))[0]
        print(f"Processing layer: {layer_name}")
        
        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(gdf)
        
        # Create HTML report with explicit filename
        html_output = f"matrice_{layer_name}_report.html"
        create_matrix_html_report(distance_matrix, layer_name, html_output)
        print(f"Created HTML report: {html_output}")
        
        # Display basic statistics
        print("\nDistance Matrix Statistics:")
        print("-" * 40)
        print(f"Matrix shape: {distance_matrix.shape}")
        print(f"Mean distance: {distance_matrix.mean().mean():.2f}")
        print(f"Min distance: {distance_matrix.min().min():.2f}")
        print(f"Max distance: {distance_matrix.max().max():.2f}")
        print("-" * 40)
        
        # Save matrix
        save_format = input("\nSave format (csv/pickle): ").lower()
        output_path = f"matrice_{layer_name}.{'csv' if save_format == 'csv' else 'pkl'}"
        save_matrix(distance_matrix, output_path)
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == '__main__':
    main()
