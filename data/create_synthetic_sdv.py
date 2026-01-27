import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import os

# Definizione dei percorsi
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, 'test_data.csv')
CSV_OUTPUT = os.path.join(DATA_DIR, 'synthetic_data_sdv.csv')
PARQUET_OUTPUT = os.path.join(DATA_DIR, 'synthetic_data_sdv.parquet')

def generate_synthetic_data(num_rows=2000):      # ho impostato 2000 righe come default per non esagerare, poichè faccio anche il salvataggio in CSV 
    print(f"Caricamento dati da {INPUT_FILE}...")
    data = pd.read_csv(INPUT_FILE)
    
    # PULIZIA E GESTIONE DATI PER SDV
    # Gestione Timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Gestiamo i valori mancanti nelle categorie (causa comune di OverflowError) e li convertiamo in stringa 
    if 'Anomaly_Type' in data.columns:
        data['Anomaly_Type'] = data['Anomaly_Type'].fillna('None').astype(str)
    
    # ci assicuriamo che Machine_ID sia stringa
    if 'Machine_ID' in data.columns:
        data['Machine_ID'] = data['Machine_ID'].astype(str)

    # rilevamento metadati
    print("Rilevamento metadati...")
    metadata = Metadata.detect_from_dataframe(data)
    
    # Forzatura sui metadati per evitare problemi a sdv
    metadata.update_column(column_name='timestamp', sdtype='datetime')
    metadata.update_column(column_name='Machine_ID', sdtype='categorical')
    metadata.update_column(column_name='Anomaly_Type', sdtype='categorical')
    
    # inizializzazione del modello di sdv, pare sia il migliore in questo caso
    synthesizer = GaussianCopulaSynthesizer(metadata)
    
    # Fitting
    synthesizer.fit(data)
    
    # generazione dei dati con il numero di righe specificato 
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    # salvataggio in Parquet
    synthetic_data.to_parquet(PARQUET_OUTPUT, index=False)
    
    # salvataggio in CSV
    synthetic_data.to_csv(CSV_OUTPUT, index=False)


    print("Fatto! Generazione completata con successo.")
    return synthetic_data

if __name__ == "__main__":
    generate_synthetic_data()