<<<<<<< HEAD
# Signal filtering for raw telemetry streams.
# Intended to provide low-pass, median, and other smoothing filters for joint
# position / velocity / torque data before downstream feature computation.
=======
import numpy as np
from scipy import signal
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TelemetryFilter:
    """
    Applies zero-phase digital filtering to noisy telemetry data.
    """
    
    def __init__(self, sample_rate: float, cutoff_freq: float = 15.0, order: int = 4):
        """
        Initializes the Butterworth lowpass filter.
        
        Args:
            sample_rate (float): The sampling frequency of the data in Hz.
            cutoff_freq (float): The cutoff frequency for the lowpass filter in Hz.
            order (int): The order of the Butterworth filter.
        """
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.order = order
        
        # Calculate the Nyquist frequency
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        # Get the filter coefficients
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
    def filter_array(self, data: np.ndarray) -> np.ndarray:
        """
        Applies a zero-phase forward-backward filter to a 1D or 2D array.
        
        Args:
            data (np.ndarray): The noisy input data. If 2D, filtering is applied along axis 0.
            
        Returns:
            np.ndarray: The smoothed output data.
        """
        if len(data) <= 15:
            logger.warning("Data too short for effective filtering. Returning raw data.")
            return data
            
        # Use filtfilt to apply the filter forwards and backwards (zero-phase)
        return signal.filtfilt(self.b, self.a, data, axis=0)

    def filter_dataframe_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Applies the filter to specific columns of a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            columns (list[str]): The column names to filter.
            
        Returns:
            pd.DataFrame: A new dataframe with the specified columns filtered.
        """
        df_filtered = df.copy()
        
        for col in columns:
            if col in df_filtered.columns:
                # Handle cases where the column contains lists (e.g., arrays of joints)
                sample_val = df_filtered.iloc[0][col]
                if isinstance(sample_val, (list, np.ndarray)):
                    # Convert series of lists to a 2D numpy array
                    stacked = np.stack(df_filtered[col].values)
                    filtered_stacked = self.filter_array(stacked)
                    # Convert back to a series of lists
                    df_filtered[col] = list(filtered_stacked)
                else:
                    # Standard 1D numeric column
                    df_filtered[col] = self.filter_array(df_filtered[col].to_numpy())
            else:
                logger.warning(f"Column '{col}' not found in DataFrame.")
                
        return df_filtered
>>>>>>> d8d255ef7cce25e829b0eef8d4032f0ebc4ac185
