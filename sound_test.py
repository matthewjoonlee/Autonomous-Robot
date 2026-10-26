"""Standalone speech test for the robot's USB sound hardware. No motor or driving code.

    python sound_test.py              # check the audio setup, then say the test phrase
    python sound_test.py --use-usb    # send this program's audio straight to the USB card (ALSA)
    python sound_test.py --beep       # also play a short beep (checks speakers without speech)

Uses USB_SoundController from usb_sound_controller.py: queues text-to-speech
(Festival's text2wave), waits until the queued speech has been generated and
playback has finished, then closes the controller. Prints what it found:
Python modules, text2wave, USB audio cards, the default output, and mute/volume.
"""

import argparse
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import wave

TEXT = "Hello! My sound system is working."


def run(command):
    """(return code, output) of a short system command, or None if it is not installed."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.returncode, (result.stdout + result.stderr).strip()


def usb_cards():
    """[(card number, description)] for USB playback devices listed by `aplay -l`."""
    listing = run(["aplay", "-l"])
    if listing is None:
        return []
    cards = []
    for line in listing[1].splitlines():
        match = re.match(r"card (\d+): (.*)", line)
        if match and "usb" in line.lower():
            cards.append((int(match.group(1)), match.group(2).strip()))
    return cards


def diagnose():
    """Print the audio setup; return a list of problems found."""
    problems = []
    print("== Audio setup ==")
    try:
        import pygame
        import pygame.mixer  # noqa: F401  (some builds lack audio support)
        print(f"pygame {pygame.version.ver} with audio mixer: OK")
    except ImportError as exc:
        problems.append("pygame audio unavailable "
                        f"({exc}): pip install --force-reinstall pygame, or sudo apt install python3-pygame")
    try:
        from pydub import AudioSegment  # noqa: F401  (usb_sound_controller imports it)
        print("pydub: OK")
    except ImportError as exc:
        if "audioop" in str(exc):  # Python 3.13+ removed audioop, which pydub needs.
            problems.append(f"pydub needs audioop on Python {sys.version_info.major}.{sys.version_info.minor}: "
                            "pip install audioop-lts")
        else:
            problems.append("pydub is not installed (pip install pydub)")
    text2wave = shutil.which("text2wave")
    print(f"text2wave (Festival speech): {text2wave or 'NOT FOUND'}")
    if not text2wave:
        problems.append("Festival is not installed: sudo apt install festival")

    cards = usb_cards()
    listing = run(["aplay", "-l"])
    if listing is None:
        print("aplay not found (ALSA tools); cannot list sound cards")
    else:
        print("Playback devices (aplay -l):")
        for line in listing[1].splitlines():
            if line.startswith("card"):
                print("  " + line)
        if not cards:
            problems.append("No USB playback device found by `aplay -l`: check the USB audio cable/adapter")

    default = run(["pactl", "get-default-sink"])
    if default and default[0] == 0:
        print(f"Default output (PipeWire/PulseAudio): {default[1]}")
        if cards and "usb" not in default[1].lower():
            problems.append("The default output is not the USB device: run with --use-usb, or select it "
                            "(pactl list short sinks; pactl set-default-sink <usb sink name>)")
        mute = run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
        volume = run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        if mute:
            print(f"  {mute[1]}")
            if "yes" in mute[1].lower():
                problems.append("Default output is muted: pactl set-sink-mute @DEFAULT_SINK@ 0")
        if volume:
            print("  " + volume[1].splitlines()[0])
            levels = [int(v) for v in re.findall(r"(\d+)%", volume[1])]
            if levels and max(levels) < 30:
                problems.append("Default output volume is low: pactl set-sink-volume @DEFAULT_SINK@ 80%")
    else:
        print("No PipeWire/PulseAudio default output found (plain ALSA)")

    for card, name in cards:
        mixer = run(["amixer", "-c", str(card), "scontents"])
        if mixer is None:
            break
        muted = "[off]" in mixer[1]
        levels = [int(v) for v in re.findall(r"\[(\d+)%\]", mixer[1])]
        print(f"USB card {card} mixer: {'some controls MUTED' if muted else 'not muted'}"
              + (f", levels {min(levels)}-{max(levels)}%" if levels else ""))
        if muted:
            problems.append(f"USB card {card} has a muted control: alsamixer -c {card} (press M to unmute)")
        if levels and max(levels) < 30:
            problems.append(f"USB card {card} volume is low: amixer -c {card} sset PCM 80%")
    return problems


def write_beep(path, seconds=0.6, frequency=880):
    """A short 16-bit stereo tone, to test the speakers without speech synthesis."""
    rate = 44100
    with wave.open(path, "wb") as beep:
        beep.setnchannels(2)
        beep.setsampwidth(2)
        beep.setframerate(rate)
        for i in range(int(rate * seconds)):
            sample = int(12000 * math.sin(2 * math.pi * frequency * i / rate))
            beep.writeframes(struct.pack("<hh", sample, sample))


def wait_until_done(controller, timeout=20.0):
    """Wait for queued tasks to finish, then for playback to end. Returns True if sound played."""
    import pygame
    deadline = time.monotonic() + timeout
    while controller.task_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.05)
    played = False
    while pygame.mixer.get_init() and pygame.mixer.get_busy() and time.monotonic() < deadline:
        played = True
        time.sleep(0.05)
    return played


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--volume", type=float, default=0.9, help="0.0-1.0 (default 0.9)")
    parser.add_argument("--use-usb", action="store_true",
                        help="play through the first USB card via ALSA, bypassing the default output")
    parser.add_argument("--beep", action="store_true", help="also play a short test tone")
    args = parser.parse_args(argv)

    problems = diagnose()
    if args.use_usb:
        cards = usb_cards()
        if cards:
            # Must be set before pygame starts; affects only this program.
            os.environ["SDL_AUDIODRIVER"] = "alsa"
            os.environ["AUDIODEV"] = f"plughw:{cards[0][0]},0"
            print(f"Routing this program's audio to USB card {cards[0][0]} ({os.environ['AUDIODEV']})")
        else:
            print("--use-usb: no USB card found; using the default output")

    print("\n== Sound test ==")
    try:
        from usb_sound_controller import USB_SoundController
        import pygame
    except ImportError as exc:
        print(f"Cannot start the sound controller: {exc}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    try:
        controller = USB_SoundController(volume=args.volume)
    except Exception as exc:  # e.g. pygame built without its mixer module.
        print(f"Cannot start the sound controller: {exc}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    results = {}
    try:
        mixer = pygame.mixer.get_init()
        results["audio device opened"] = bool(mixer)
        if not mixer:
            print("pygame could not open an audio output device.")
        else:
            print(f"Audio output opened: {mixer[0]} Hz, {mixer[2]} channels")
            if shutil.which("text2wave"):
                if os.path.exists("temp_tts.wav"):
                    os.remove("temp_tts.wav")
                print(f'Speaking: "{args.text}"')
                controller.play_text_to_speech(args.text)
                results["speech played"] = wait_until_done(controller)
                results["speech audio generated"] = os.path.exists("temp_tts.wav") and \
                    os.path.getsize("temp_tts.wav") > 1000
            if args.beep or not results.get("speech played"):
                write_beep("sound_test_beep.wav")
                print("Playing a test beep...")
                controller.play_audio("sound_test_beep.wav")
                results["beep played"] = wait_until_done(controller)
    finally:
        controller.close()

    print("\n== Result ==")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    if problems:
        print("Problems found:")
        for problem in problems:
            print(f"  - {problem}")
    if results.get("speech played") or results.get("beep played"):
        print("Software played audio. If you heard nothing: check the speaker is powered/switched on, "
              "its volume knob, the cable into the USB adapter, and try --use-usb.")
    return 0 if results.get("speech played") else 1


if __name__ == "__main__":
    sys.exit(main())
