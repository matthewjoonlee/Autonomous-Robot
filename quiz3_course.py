"""Run the four requested five-foot Quiz 3 trials, with stops between trials."""

import argparse
import sys

from quiz3_autonomous import main as run_centering
from calibrate_motors import bounded_int


TRIALS = ("straight", "left", "right", "nudge")


def run_suite(execute, runner=run_centering, exposure_us=5000):
    for index, trial in enumerate(TRIALS, 1):
        print(f"\nTrial {index}/4: {trial.upper()}")
        print("Reposition at the marked start, centered and straight, before capturing the reference.")
        arguments = ["--trial", trial, "--distance-ft", "5"]
        if execute:
            arguments.append("--execute")
        if exposure_us != 5000:
            arguments.extend(["--exposure-us", str(exposure_us)])
        result = runner(arguments)
        if result != 0:
            print("Suite stopped after a fault/interruption/cancellation. Completed logs are saved.")
            return result
    print("All four trials finished. Review the logs and operator notes for recovery performance.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable physical motors for each trial")
    parser.add_argument("--exposure-us", type=bounded_int(0, 20000), default=5000,
                        help="shutter microseconds, default 5000; 0 keeps auto exposure")
    args = parser.parse_args()
    return run_suite(args.execute, exposure_us=args.exposure_us)


if __name__ == "__main__":
    sys.exit(main())
