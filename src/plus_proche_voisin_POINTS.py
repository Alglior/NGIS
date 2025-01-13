import geopandas as gpd
from shapely.ops import nearest_points
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import pandas as pd

def read_geodata(file_path):
    """
    Read various types of geographic data files (GeoPackage, Shapefile, GeoJSON)
    
    Parameters:
    -----------
    file_path : str
        Path to the geographic data file
        
    Returns:
    --------
    GeoDataFrame
        Loaded geographic data
    """
    # Get file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    try:
        if extension == '.gpkg':
            # For GeoPackage, we might need to specify the layer
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
            
        # Verify that we have a valid GeoDataFrame
        if gdf.empty:
            raise ValueError("The file contains no data")
            
        return gdf
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def process_chunk(chunk_data):
    """
    Process a chunk of points to find nearest neighbors
    """
    chunk, all_points_array, start_idx = chunk_data
    distances = []
    
    # Convert all points to numpy array once
    all_coords = np.array([(p.x, p.y) for p in all_points_array])
    
    for i, (point_idx, point_geom) in enumerate(chunk.items()):
        # Create mask for other points
        mask = np.ones(len(all_coords), dtype=bool)
        mask[point_idx] = False
        
        # Vectorized distance calculation using broadcasting
        point_coords = np.array([point_geom.x, point_geom.y])
        dists = np.sqrt(np.sum((all_coords[mask] - point_coords) ** 2, axis=1))
        
        min_distance = np.min(dists) if len(dists) > 0 else np.nan
        distances.append((point_idx, min_distance))
        
    return distances

def nearest_neighbor_distance(gdf):
    """
    Calculate the nearest neighbor distance using all CPU cores
    """
    if not all(geom.geom_type == 'Point' for geom in gdf.geometry):
        raise ValueError("All geometries must be points")
    
    n_cores = cpu_count()
    total_points = len(gdf)
    
    # Use larger chunks to reduce overhead
    chunk_size = max(100, total_points // (n_cores * 2))
    
    # Prepare chunks
    chunks = []
    for i in range(0, total_points, chunk_size):
        chunk = gdf.geometry[i:i + chunk_size]
        chunks.append((chunk, gdf.geometry, i))
    
    print(f"Processing {total_points} points in {len(chunks)} chunks using {n_cores} cores")
    
    # Process chunks in parallel with improved progress tracking
    all_distances = []
    processed_points = 0
    
    with Pool(processes=n_cores) as pool:
        with tqdm(total=total_points, desc="Processing points") as pbar:
            for chunk_distances in pool.imap(process_chunk, chunks):
                if chunk_distances:
                    all_distances.extend(chunk_distances)
                    points_in_chunk = len(chunk_distances)
                    processed_points += points_in_chunk
                    pbar.update(points_in_chunk)
                    
                    # Print progress every 5%
                    if processed_points % max(1, total_points // 20) == 0:
                        print(f"Processed {processed_points}/{total_points} points", flush=True)
    
    if not all_distances:
        raise RuntimeError("No points were processed successfully")
    
    print(f"\nCompleted processing {len(all_distances)} points")
    
    # Sort distances and update GeoDataFrame
    all_distances.sort(key=lambda x: x[0])
    gdf['nn_distance'] = [d[1] for d in all_distances]
    
    return gdf

def save_results(gdf, output_path, driver='GPKG'):
    """
    Save results to a new file
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame to save
    output_path : str
        Path where to save the file
    driver : str
        Output format ('GPKG', 'ESRI Shapefile', or 'GeoJSON')
    """
    try:
        gdf.to_file(output_path, driver=driver)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def create_html_report(gdf, layer_name, output_path=None):
    """
    Create an HTML report with the nearest neighbor analysis results
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing the nearest neighbor distances
    layer_name : str
        Name of the layer being processed
    output_path : str, optional
        Custom path for the HTML report
    """
    if output_path is None:
        output_path = f"ppv_{layer_name}_report.html"
    
    html_content = f"""
    <html>
    <head>
        <title>Nearest Neighbor Analysis - {layer_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Nearest Neighbor Analysis - {layer_name}</h1>
        <div class="stats">
            <h2>Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Layer name</td><td>{layer_name}</td></tr>
                <tr><td>Number of points</td><td>{len(gdf)}</td></tr>
                <tr><td>Mean distance</td><td>{gdf['nn_distance'].mean():.2f}</td></tr>
                <tr><td>Minimum distance</td><td>{gdf['nn_distance'].min():.2f}</td></tr>
                <tr><td>Maximum distance</td><td>{gdf['nn_distance'].max():.2f}</td></tr>
                <tr><td>Standard deviation</td><td>{gdf['nn_distance'].std():.2f}</td></tr>
            </table>
        </div>
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
        
        # Calculate nearest neighbor distances
        gdf_with_distances = nearest_neighbor_distance(gdf)
        
        # Display detailed statistics in terminal
        print("\nNearest neighbor distance statistics:")
        print("-" * 40)
        print(f"Number of points: {len(gdf_with_distances)}")
        print(f"Mean distance: {gdf_with_distances['nn_distance'].mean():.2f}")
        print(f"Minimum distance: {gdf_with_distances['nn_distance'].min():.2f}")
        print(f"Maximum distance: {gdf_with_distances['nn_distance'].max():.2f}")
        print(f"Standard deviation: {gdf_with_distances['nn_distance'].std():.2f}")
        print("-" * 40)
        
        # Create HTML report with layer name
        html_output = f"ppv_{layer_name}_report.html"
        create_html_report(gdf_with_distances, layer_name, html_output)
        print(f"Created HTML report: {html_output}")
        
        # Save results with layer name
        output_format = input("\nChoose output format (gpkg/shp/geojson): ").lower()
        output_path = f"ppv_{layer_name}.{output_format}"
        driver = 'GPKG' if output_format == 'gpkg' else 'ESRI Shapefile' if output_format == 'shp' else 'GeoJSON'
        save_results(gdf_with_distances, output_path, driver)
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == '__main__':
    main()