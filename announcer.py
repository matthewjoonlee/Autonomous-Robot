"""Non-blocking spoken status for the robot (Week 9: sound must not block control).

Each phrase is rendered to a WAV once with Festival's text2wave (cached in
sounds/tts/), so speaking at run time is just playing a file. A new phrase
replaces anything still playing or queued (latest wins), so announcements never
lag behind the robot. Uses the course's USB_SoundController.
"""

import hashlib
from pathlib import Path
import re
import subprocess

CACHE = Path(__file__).resolve().parent / "sounds" / "tts"


class Announcer:
    def __init__(self, phrases=(), volume=1.0, controller=None):
        if controller is None:
            from usb_sound_controller import USB_SoundController
            controller = USB_SoundController(volume=volume)
        self.sound = controller
        self.files = {}
        for phrase in phrases:
            self.prepare(phrase)

    def prepare(self, phrase):
        """Render phrase to a cached WAV (a second or so the first time); return its path."""
        if phrase in self.files:
            return self.files[phrase]
        CACHE.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")[:40]
        digest = hashlib.sha1(phrase.encode()).hexdigest()[:8]
        path = CACHE / f"{stem}_{digest}.wav"
        if not path.exists():
            result = subprocess.run(["text2wave", "-scale", "2.0", "-o", str(path)],
                                    input=phrase + "\n", text=True, capture_output=True)
            if result.returncode != 0 or not path.exists():
                print(f"Speech render failed for {phrase!r}: {result.stderr.strip()}")
                return None
        self.files[phrase] = str(path)
        return self.files[phrase]

    def say(self, phrase):
        """Speak now, cutting off any earlier phrase."""
        path = self.prepare(phrase)
        if path is None:
            return
        self.sound.stop_sound()   # stops playback and purges queued sounds
        self.sound.play_audio(path)

    def stop(self):
        self.sound.stop_sound()

    def close(self):
        try:
            self.sound.stop_sound()
        finally:
            self.sound.close()
