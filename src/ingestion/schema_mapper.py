import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SchemaMapper:
    """
    Auto-detects and normalises DataFrames from different dataset schemas (e.g., our pipeline,
    HuggingFace LeRobot format) into a canonical format required by our metrics engine.
    """
    
    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        """
        Identifies which known schema a DataFrame matches, without modifying it.
        Mirrors the branch order of normalise(). Returns one of:
        'native', 'hf_subarrays', 'hf_body', 'hf_flat', or 'unknown'.
        """
        if df.empty:
            return "unknown"
        columns = set(df.columns)
        if 'q' in columns and ('tick' in columns or 'timestamp' in columns):
            return "native"
        if 'observation.state.left_leg' in columns:
            return "hf_subarrays"
        if 'observation.body' in columns:
            return "hf_body"
        if 'observation.state' in columns and 'action' in columns:
            return "hf_flat"
        return "unknown"

    @staticmethod
    def normalise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes an input DataFrame of unknown schema and returns a normalised DataFrame
        with guaranteed 'tick', 'q', and 'q_cmd' columns. The 'q' array is enforced to be
        the canonical G1 29-DoF order: Left Leg (6), Right Leg (6), Waist (3), Left Arm (7), Right Arm (7).
        """
        if df.empty:
            return df
            
        columns = set(df.columns)
        out_df = df.copy()
        
        # Ensure timestamp exists
        if 'timestamp' in columns:
            out_df['tick'] = out_df['timestamp'] * 1000.0
        
        # Format A: Already in our internal schema
        if 'q' in columns and 'tick' in columns:
            logger.info("SchemaMapper: Detected Format A (Native Pipeline)")
            return out_df
            
        # Format B: HuggingFace with explicit sub-arrays (e.g. file-000.parquet)
        if 'observation.state.left_leg' in columns:
            logger.info("SchemaMapper: Detected Format B (Explicit Sub-arrays)")
            
            def build_q(row, prefix):
                q_arr = []
                # Canonical G1 order
                suffixes = ['left_leg', 'right_leg', 'waist_state_joint', 'left_arm', 'right_arm']
                for s in suffixes:
                    col = f"{prefix}.{s}"
                    # Fallback for action which names it waist_action_joint
                    if col not in row and s == 'waist_state_joint':
                        col = f"{prefix}.waist_action_joint"
                        
                    if col in row:
                        val = row[col]
                        if isinstance(val, (list, np.ndarray)):
                            q_arr.extend(val)
                        else:
                            q_arr.append(val)
                return q_arr
                
            out_df['q'] = out_df.apply(lambda row: build_q(row, 'observation.state'), axis=1)
            out_df['q_cmd'] = out_df.apply(lambda row: build_q(row, 'action'), axis=1)
            return out_df
            
        # Format C: HuggingFace with observation.body (e.g. episode_000000.parquet)
        if 'observation.body' in columns:
            logger.info("SchemaMapper: Detected Format C (Body array)")
            out_df['q'] = out_df['observation.body']
            if 'action.body' in columns:
                out_df['q_cmd'] = out_df['action.body']
            else:
                out_df['q_cmd'] = out_df['observation.body']
            return out_df
            
        # Fallback Format: Single observation state
        if 'observation.state' in columns and 'action' in columns:
            logger.info("SchemaMapper: Detected Fallback Format (observation.state)")
            out_df['q'] = out_df['observation.state']
            out_df['q_cmd'] = out_df['action']
            return out_df
            
        logger.warning("SchemaMapper: Unknown schema. Returning unchanged.")
        return out_df
