"""Exam-style session: the same single program as quiz3_autonomous.py.

One run covers every test: place the robot (centered, half off at an edge, ...),
it drives up to 15 ft, stops, and waits to be picked up or slid to the next start.
"""

import sys

from quiz3_autonomous import main


if __name__ == "__main__":
    sys.exit(main())
