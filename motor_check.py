"""Spin each motor alone to find which side is not working. Lift the wheels first.

    python motor_check.py            # power 30, 1 s per step

Uses the Sabertooth's independent motor commands (motor 1: 0/1, motor 2: 4/5),
bypassing the mixed drive/turn mode the robot programs use. For each step, note
which wheel turns and which way, then type it in. A wheel that never turns on
its own channel points at its motor, wiring, or terminals; if the other channel
turns that same wheel, the M1/M2 wiring is swapped.
"""

import argparse
import sys
import time

from calibrate_motors import bounded_int

STEPS = [("motor 1 forward", 0), ("motor 1 backward", 1), ("motor 2 forward", 4), ("motor 2 backward", 5)]


def stop_all(saber):
    for command in (0, 4):  # Independent motor 1 / motor 2 forward at 0.
        saber.send_command(command, 0)
    saber.stop()  # Mixed-mode stop as well.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", type=bounded_int(5, 60), default=30, help="motor power 0-127 (default 30)")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    from sabertooth import Sabertooth
    saber = Sabertooth()
    if saber.ser is None or not saber.ser.is_open:
        print("Motor serial port unavailable; check the Sabertooth connection.", file=sys.stderr)
        return 1
    notes = []
    try:
        stop_all(saber)
        input("Lift the robot so both wheels spin freely, then press Enter: ")
        for label, command in STEPS:
            input(f"\nNext: {label} at power {args.power} for {args.seconds:g} s. Press Enter to run...")
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:  # Repeat so the 200 ms serial timeout never stops it early.
                saber.send_command(command, args.power)
                time.sleep(0.05)
            stop_all(saber)
            seen = input("Which wheel turned, and which way? (e.g. 'left fwd', 'none'): ").strip()
            notes.append((label, seen))
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        stop_all(saber)
        saber.close()
    print("\nSummary:")
    for label, seen in notes:
        print(f"  {label:18s} -> {seen or '(no answer)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
