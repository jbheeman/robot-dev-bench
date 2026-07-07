import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataExporter:
    """Exports parsed state data to Parquet or HDF5."""

    @staticmethod
    def to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts a list of dicts (states) into a Pandas DataFrame."""
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    @staticmethod
    def export_parquet(data: List[Dict[str, Any]], filepath: str):
        """Exports data to a Parquet file."""
        df = DataExporter.to_dataframe(data)
        if df.empty:
            logger.warning("No data to export to Parquet.")
            return
        
        df.to_parquet(filepath, engine='pyarrow', index=False)
        logger.info(f"Exported data to Parquet: {filepath}")

    @staticmethod
    def export_hdf5(data: List[Dict[str, Any]], filepath: str, key: str = "dataset"):
        """Exports data to an HDF5 file."""
        df = DataExporter.to_dataframe(data)
        if df.empty:
            logger.warning("No data to export to HDF5.")
            return
        
        df.to_hdf(filepath, key=key, mode='w', format='table')
        logger.info(f"Exported data to HDF5: {filepath} (key: {key})")
