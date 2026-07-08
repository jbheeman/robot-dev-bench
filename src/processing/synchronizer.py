import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataSynchronizer:
    """
    Synchronizes different streams of telemetry data (e.g., LowState and HighState)
    based on their timestamps or hardware ticks.
    """
    
    @staticmethod
    def sync_low_high_states(low_df: pd.DataFrame, high_df: pd.DataFrame, 
                           on_column: str = 'tick', 
                           direction: str = 'nearest',
                           tolerance: int = 100) -> pd.DataFrame:
        """
        Merges LowState and HighState DataFrames using a nearest-neighbor join on ticks.
        
        Args:
            low_df (pd.DataFrame): The higher-frequency LowState data.
            high_df (pd.DataFrame): The lower-frequency HighState data.
            on_column (str): The column representing time/ticks to merge on.
            direction (str): 'backward', 'forward', or 'nearest' mapping strategy.
            tolerance (int): Maximum tick distance allowed for a match.
            
        Returns:
            pd.DataFrame: A synchronized dataframe containing both sets of features.
        """
        if low_df.empty or high_df.empty:
            logger.warning("One of the input DataFrames is empty. Cannot synchronize.")
            return pd.DataFrame()
            
        if on_column not in low_df.columns or on_column not in high_df.columns:
            logger.error(f"Synchronization column '{on_column}' missing from inputs.")
            raise ValueError(f"Column '{on_column}' required in both dataframes.")

        # Ensure both dataframes are sorted by the merge key (required by merge_asof)
        low_sorted = low_df.sort_values(by=on_column).reset_index(drop=True)
        high_sorted = high_df.sort_values(by=on_column).reset_index(drop=True)
        
        # We usually merge the lower-freq data onto the higher-freq timeline,
        # or we could merge both ways. Here we bring HighState info into the LowState timeline.
        merged_df = pd.merge_asof(
            low_sorted, 
            high_sorted, 
            on=on_column, 
            direction=direction,
            tolerance=tolerance,
            suffixes=('_low', '_high')
        )
        
        # Optional: forward-fill any NaNs in the HighState columns that didn't match within tolerance
        # (if tolerance was exceeded for a few samples).
        # We will leave them as NaN so the user knows there was a dropout.
        
        logger.info(f"Synchronized DataFrames: resulted in {len(merged_df)} rows.")
        return merged_df
