"""Run the four requested five-foot Quiz 3 trials, with stops between trials."""

import argparse
import sys

from quiz3_autonomous import main as run_centering


TRIALS = ("straight", "left", "right", "nudge")


def run_suite(execute, runner=run_centering):
    for index, trial in enumerate(TRIALS, 1):
        print(f"\nTrial {index}/4: {trial.upper()}")
        print("Reposition at the marked start, centered and straight, before capturing the reference.")
        arguments = ["--trial", trial, "--distance-ft", "5"]
        if execute:
            arguments.append("--execute")
        result = runner(arguments)
        if result != 0:
            print("Suite stopped after a fault/interruption/cancellation. Completed logs are saved.")
            return result
    print("All four trials finished. Review the logs and operator notes for recovery performance.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable physical motors for each trial")
    args = parser.parse_args()
    return run_suite(args.execute)


if __name__ == "__main__":
    sys.exit(main())
