import logging
import os
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCheck")

def check_datasets():
    path = "data/offline/machines_batch_features"
    if not os.path.exists(path):
        logger.error(f"Percorso non trovato: {path}")
        return

    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    logger.info(f"File trovati nella cartella: {len(files)}")
    
    if not files:
        logger.warning("Nessun file parquet trovato.")
        return

    schemas = {}
    
    for f in files:
        file_path = os.path.join(path, f)
        try:
            # Leggiamo solo i metadati se possibile, o poche righe
            df = pd.read_parquet(file_path)
            schema = df.dtypes.to_dict()
            
            # Check colonne tutte None
            none_cols = df.columns[df.isnull().all()].tolist()
            
            logger.info(f"File: {f}")
            logger.info(f"  Shape: {df.shape}")
            if none_cols:
                logger.warning(f"  ⚠️ Colonne interamente NULL: {none_cols}")
            
            # Check consistenza schema
            schema_str = str(sorted(schema.items()))
            if schema_str not in schemas:
                schemas[schema_str] = []
            schemas[schema_str].append(f)
            
        except Exception as e:
            logger.error(f"Errore nella lettura di {f}: {e}")

    if len(schemas) > 1:
        logger.error(f"🚨 RILEVATA INCONSISTENZA SCHEMI! Trovati {len(schemas)} schemi diversi.")
        for i, (schema_val, filenames) in enumerate(schemas.items()):
            logger.info(f"Schema {i+1} (presente in {len(filenames)} file):")
            logger.info(f"  Esempio file: {filenames[0]}")
    else:
        logger.info("✅ Tutti i file hanno lo stesso schema.")

if __name__ == "__main__":
    check_datasets()
