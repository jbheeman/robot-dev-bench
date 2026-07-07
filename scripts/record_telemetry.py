# CLI entry point for capturing live telemetry from a Unitree G1 robot.
# Connects to the robot over the specified network interface, records DDS messages
# for --duration seconds (or until Ctrl-C), then exports the frames to Parquet or HDF5.
#
# Usage:
#   python record_telemetry.py --interface eth0 --duration 30 --output_dir ./data
import argparse
import logging
import os
import sys
import time

# Ensure src is in the python path if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ingestion.live_subscriber import UnitreeLiveSubscriber
from ingestion.exporter import DataExporter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Record live telemetry from Unitree G1 via DDS.")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface (e.g., eth0, wlan0).")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted parquet/hdf5 files.")
    parser.add_argument("--format", type=str, choices=['parquet', 'hdf5'], default='parquet', help="Output format.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subscriber = UnitreeLiveSubscriber(interface=args.interface)

    # initialize() sets up the CycloneDDS ChannelFactory; must succeed before subscribing
    try:
        subscriber.initialize()
    except Exception as e:
        logger.error(f"Initialization failed. Are you connected to the robot and is unitree_sdk2py installed? {e}")
        sys.exit(1)

    subscriber.start_recording()

    # Block for the requested duration; Ctrl-C triggers an early but clean stop
    logger.info(f"Recording for {args.duration} seconds...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user.")
    finally:
        subscriber.stop_recording()

    logger.info(f"Recorded {len(subscriber.low_states)} low states and {len(subscriber.high_states)} high states.")

    # Stamp filenames with the wall-clock time so successive runs don't overwrite each other
    timestamp = int(time.time())
    lowstate_file = os.path.join(args.output_dir, f"run_{timestamp}_lowstate.{args.format}")
    highstate_file = os.path.join(args.output_dir, f"run_{timestamp}_highstate.{args.format}")

    if args.format == 'parquet':
        DataExporter.export_parquet(subscriber.low_states, lowstate_file)
        DataExporter.export_parquet(subscriber.high_states, highstate_file)
    else:
        DataExporter.export_hdf5(subscriber.low_states, lowstate_file, key="lowstate")
        DataExporter.export_hdf5(subscriber.high_states, highstate_file, key="highstate")

    logger.info("Extraction complete.")

if __name__ == "__main__":
    main()
