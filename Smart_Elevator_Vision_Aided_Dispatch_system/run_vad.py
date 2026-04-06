"""
=============================================================================
VAD System – Entry Point
=============================================================================
Run this file to start the system.

MODES:
  python run_vad.py --mode sim       # Text simulation (no camera needed)
  python run_vad.py --mode live      # Live webcam mode
  python run_vad.py --mode live --camera 1   # Use camera index 1

SIMULATION PARAMETERS (all optional):
  --floors   10       Number of building floors  (default 10)
  --steps    50       Simulation ticks            (default 50)
  --ghost    0.3      Ghost call probability       (default 0.3)

LIVE PARAMETERS:
  --camera   0        Camera index                (default 0)
  --floor    0        Floor being monitored        (default 0)
  --roi      0 0 640 480   ROI x1 y1 x2 y2       (default = full frame)
  --capacity 12       Max elevator capacity        (default 12)
=============================================================================
"""

import argparse
import sys
from vad_main import VADSystem


def parse_args():
    p = argparse.ArgumentParser(description="Vision-Aided Dispatch (VAD) System v2.0")
    p.add_argument("--mode",     choices=["sim", "live"], default="sim",
                   help="Run mode: 'sim' = text simulation, 'live' = webcam")

    # Simulation args
    p.add_argument("--floors",   type=int,   default=10)
    p.add_argument("--steps",    type=int,   default=50)
    p.add_argument("--ghost",    type=float, default=0.3,
                   help="Probability [0-1] that a generated call is a ghost call")

    # Live camera args
    p.add_argument("--camera",   type=int,   default=0)
    p.add_argument("--floor",    type=int,   default=0,
                   help="Floor ID that this camera is watching")
    p.add_argument("--roi",      type=int,   nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="Region-of-interest bounding box in pixels")
    p.add_argument("--capacity", type=int,   default=12)

    # Dispatch weight tuning
    p.add_argument("--w1",       type=float, default=10.0,
                   help="Weight for people count (throughput)")
    p.add_argument("--w2",       type=float, default=1.0,
                   help="Weight for distance  (efficiency)")
    p.add_argument("--w3",       type=float, default=0.1,
                   help="Weight for wait age  (anti-starvation)")
    return p.parse_args()


def main():
    args = parse_args()

    # Build ROI tuple if provided
    roi = tuple(args.roi) if args.roi else None

    system = VADSystem(
        roi=roi,
        confidence=0.45,
        ghost_frames=90,
        max_capacity=args.capacity,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
    )

    if args.mode == "sim":
        print("\n🏢  VAD System – Simulation Mode\n")
        results = system.run_simulation(
            num_floors=args.floors,
            simulation_steps=args.steps,
            ghost_call_probability=args.ghost,
        )
        print(f"\n✅  Ghost Stop Reduction Rate : {results['gsrr_percent']:.1f}%")
        print(f"✅  Trips Completed           : {results['trips_completed']}")

    elif args.mode == "live":
        print("\n📷  VAD System – Live Camera Mode")
        print("      Press  'b'  to simulate a hall button press")
        print("      Press  'q'  to quit\n")
        system.run_live(camera_index=args.camera, floor_id=args.floor)


if __name__ == "__main__":
    main()
