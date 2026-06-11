"""
Flow - borderless desktop audio visualizer
==========================================
Reacts to system audio (Spotify, games, anything playing) or a microphone.
20 visual styles, 8 color palettes, all switchable live.

Hotkeys:
  Left / Right ....... previous / next style
  Up / Down .......... cycle color palette
  Space .............. random style
  A .................. auto-cycle styles every 25s
  D .................. next audio source (system audio / microphones)
  F .................. toggle fullscreen <-> windowed
  + / - .............. sensitivity
  H .................. show / hide help
  Drag with mouse .... move window (windowed mode)
  Esc or Q ........... quit

Requires: pygame, numpy, pyaudiowpatch  (see launcher / README)
"""

import json
import math
import os
import random
import sys
import threading
import time

import numpy as np
import pygame

# ---------------------------------------------------------------- audio backend
try:
    import pyaudiowpatch as pyaudio  # Windows: supports WASAPI loopback
    HAS_AUDIO = True
except Exception:
    try:
        import pyaudio  # fallback: mic only
        HAS_AUDIO = True
    except Exception:
        pyaudio = None
        HAS_AUDIO = False

CHUNK = 1024
NBANDS = 64
BG = (8, 8, 14)
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "visualizer_settings.json")
TEST_MODE = "--test" in sys.argv


# ================================================================ audio engine
class AudioEngine:
    """Captures audio on a thread into a rolling sample buffer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.buf = np.zeros(4096, np.float32)
        self.rate = 48000
        self.devices = []
        self.dev_i = 0
        self.running = True
        self.demo = not HAS_AUDIO or TEST_MODE
        self.pa = None
        self._switch = False
        if not self.demo:
            try:
                self.pa = pyaudio.PyAudio()
                self._scan()
            except Exception:
                self.demo = True
        if not self.devices:
            self.demo = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _scan(self):
        devs = []
        try:  # WASAPI loopback devices = "what the PC is playing"
            for d in self.pa.get_loopback_device_info_generator():
                devs.append(dict(name="System audio: " + d["name"][:40],
                                 index=d["index"],
                                 rate=int(d["defaultSampleRate"]),
                                 channels=max(1, min(2, int(d["maxInputChannels"])))))
        except Exception:
            pass
        try:  # put the default output's loopback first
            d = self.pa.get_default_wasapi_loopback()
            devs.sort(key=lambda x: 0 if x["index"] == d["index"] else 1)
        except Exception:
            pass
        try:  # microphones
            for i in range(self.pa.get_device_count()):
                d = self.pa.get_device_info_by_index(i)
                if d.get("maxInputChannels", 0) > 0 and \
                   "loopback" not in d["name"].lower() and \
                   not any(x["index"] == i for x in devs):
                    devs.append(dict(name="Mic: " + d["name"][:40],
                                     index=i,
                                     rate=int(d["defaultSampleRate"]),
                                     channels=max(1, min(2, int(d["maxInputChannels"])))))
        except Exception:
            pass
        self.devices = devs

    def device_name(self):
        if self.demo:
            return "Demo audio (no capture device)"
        return self.devices[self.dev_i]["name"]

    def next_device(self):
        if self.devices:
            self.dev_i = (self.dev_i + 1) % len(self.devices)
            self._switch = True

    def _push(self, a):
        with self.lock:
            self.buf = np.roll(self.buf, -len(a))
            self.buf[-len(a):] = a

    def _run(self):
        while self.running:
            if self.demo:
                self._demo_loop()
                continue
            dev = self.devices[self.dev_i]
            self._switch = False
            try:
                stream = self.pa.open(format=pyaudio.paFloat32,
                                      channels=dev["channels"], rate=dev["rate"],
                                      input=True, input_device_index=dev["index"],
                                      frames_per_buffer=CHUNK)
            except Exception:
                time.sleep(0.5)
                self.dev_i = (self.dev_i + 1) % len(self.devices)
                continue
            self.rate = dev["rate"]
            while self.running and not self._switch:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception:
                    break
                a = np.frombuffer(data, dtype=np.float32)
                if dev["channels"] > 1:
                    a = a.reshape(-1, dev["channels"]).mean(axis=1)
                self._push(a.astype(np.float32))
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

    def _demo_loop(self):
        """Synthesized music so the app still flows with no audio device."""
        sr = 48000
        self.rate = sr
        pos = 0
        scale = [0, 3, 5, 7, 10, 12, 15, 17]
        t0 = time.time()
        while self.running and self.demo:
            t = np.arange(pos, pos + CHUNK) / sr
            beat_ph = (t * 2.0) % 1.0                      # 120 bpm
            kick = np.sin(2 * np.pi * 52 * t) * np.exp(-beat_ph * 16)
            bassn = 110 * 2 ** (scale[int(t[0]) % 8] / 12)
            bass = 0.35 * np.sin(2 * np.pi * bassn * t)
            arp = 0.22 * np.sin(2 * np.pi * bassn * 4 *
                                (1 + 0.5 * ((t * 8).astype(int) % 3)) * t)
            hat_ph = (t * 4.0 + 0.5) % 1.0
            hat = 0.15 * np.random.randn(CHUNK).astype(np.float32) * \
                np.exp(-hat_ph * 30)
            sweep = 0.1 * np.sin(2 * np.pi * (800 + 600 * np.sin(t * 0.4)) * t)
            self._push((kick + bass + arp + hat + sweep).astype(np.float32) * 0.8)
            pos += CHUNK
            lag = pos / sr - (time.time() - t0)
            if lag > 0 and not TEST_MODE:
                time.sleep(lag)

    def stop(self):
        self.running = False


# ================================================================ analysis
class Analyzer:
    def __init__(self, engine):
        self.e = engine
        self.bands = np.zeros(NBANDS)
        self.wave = np.zeros(1024, np.float32)
        self.gain = 1e-4
        self.wgain = 1e-3
        self.sens = 1.0
        self.bass = self.mid = self.treb = self.energy = 0.0
        self.beat = False
        self._bass_hist = np.zeros(43)
        self._beat_cool = 0.0

    def update(self, dt):
        with self.e.lock:
            buf = self.e.buf.copy()
            rate = self.e.rate
        win = np.hanning(2048)
        spec = np.abs(np.fft.rfft(buf[-2048:] * win))
        freqs = np.fft.rfftfreq(2048, 1.0 / rate)
        edges = np.geomspace(35, min(16000, rate / 2 - 1), NBANDS + 1)
        idx = np.searchsorted(freqs, edges)
        raw = np.zeros(NBANDS)
        for b in range(NBANDS):
            seg = spec[idx[b]:max(idx[b] + 1, idx[b + 1])]
            raw[b] = seg.max() if len(seg) else 0.0
        raw *= np.linspace(1.0, 3.2, NBANDS)          # treble tilt
        self.gain = max(raw.max(), self.gain * 0.996, 1e-4)
        target = np.clip(raw / self.gain * self.sens, 0, 1)
        up = target > self.bands
        self.bands[up] = self.bands[up] * 0.35 + target[up] * 0.65
        self.bands[~up] = self.bands[~up] * 0.86 + target[~up] * 0.14

        w = buf[-1024:]
        self.wgain = max(np.abs(w).max(), self.wgain * 0.996, 1e-3)
        self.wave = np.clip(w / self.wgain * self.sens, -1, 1)

        self.bass = float(self.bands[:8].mean())
        self.mid = float(self.bands[8:36].mean())
        self.treb = float(self.bands[36:].mean())
        self.energy = float(self.bands.mean())

        self._bass_hist = np.roll(self._bass_hist, -1)
        self._bass_hist[-1] = self.bass
        avg = self._bass_hist.mean()
        self._beat_cool = max(0.0, self._beat_cool - dt)
        self.beat = False
        if self.bass > 0.3 and self.bass > avg * 1.35 and self._beat_cool == 0:
            self.beat = True
            self._beat_cool = 0.18


# ================================================================ palettes
PALETTES = [
    ("Neon",    [(0, 255, 255), (170, 0, 255), (255, 0, 200)]),
    ("Sunset",  [(255, 200, 80), (255, 94, 58), (255, 42, 104), (110, 40, 150)]),
    ("Aurora",  [(80, 255, 170), (60, 140, 255), (190, 90, 255)]),
    ("Fire",    [(255, 240, 120), (255, 140, 30), (220, 40, 30), (90, 0, 30)]),
    ("Ocean",   [(160, 240, 255), (40, 140, 255), (20, 50, 160)]),
    ("Mono",    [(250, 250, 250), (130, 130, 140)]),
    ("Vapor",   [(255, 150, 220), (150, 160, 255), (90, 255, 230)]),
    ("Rainbow", None),
]


def make_col(pi):
    name, cols = PALETTES[pi]

    def col(x, bright=1.0):
        x = x % 1.0 if cols is None else max(0.0, min(1.0, x))
        if cols is None:
            c = pygame.Color(0)
            c.hsva = ((x * 360.0) % 360, 92, 100, 100)
            r, g, b = c.r, c.g, c.b
        else:
            f = x * (len(cols) - 1)
            i = min(int(f), len(cols) - 2)
            f -= i
            a, bb = cols[i], cols[i + 1]
            r, g, b = (a[k] + (bb[k] - a[k]) * f for k in range(3))
        return (min(255, int(r * bright)), min(255, int(g * bright)),
                min(255, int(b * bright)))
    return col


# ================================================================ draw helpers
def glow(surf, color, pos, r):
    if r < 1:
        return
    size = int(r * 3)
    tmp = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
    for rr, al in ((r * 3, 25), (r * 1.9, 55), (r, 255)):
        pygame.draw.circle(tmp, (*color, al), (size, size), max(1, int(rr)))
    surf.blit(tmp, (pos[0] - size, pos[1] - size))


class Ctx:
    """Per-frame data handed to every style function."""
    pass


# ================================================================ styles
# Each: draw(ctx).  FADE: None = full clear, 0 = never clear, n = trail alpha.

def s_bars(c):
    st = c.state.setdefault("bars", {"peaks": np.zeros(NBANDS)})
    bw = c.w / NBANDS
    st["peaks"] = np.maximum(st["peaks"] - c.dt * 0.35, c.bands)
    for i in range(NBANDS):
        v = c.bands[i]
        hh = v * c.h * 0.88
        x = i * bw
        pygame.draw.rect(c.s, c.col(i / NBANDS, 0.45 + 0.55 * v),
                         (x + 1, c.h - hh, bw - 2, hh))
        py = c.h - st["peaks"][i] * c.h * 0.88
        pygame.draw.rect(c.s, c.col(i / NBANDS), (x + 1, py - 3, bw - 2, 3))


def s_mirror(c):
    half = NBANDS // 2
    bw = c.w / NBANDS
    cy = c.h / 2
    for i in range(NBANDS):
        v = c.bands[abs(i - half) * 2 % NBANDS]
        hh = v * c.h * 0.46
        color = c.col(abs(i - half) / half, 0.4 + 0.6 * v)
        pygame.draw.rect(c.s, color, (i * bw + 1, cy - hh, bw - 2, hh * 2))


def s_circle(c):
    cx, cy = c.w / 2, c.h / 2
    base = min(c.w, c.h) * (0.18 + 0.05 * c.bass)
    for i in range(NBANDS):
        a = i / NBANDS * 2 * math.pi - math.pi / 2 + c.t * 0.25
        v = c.bands[i]
        r2 = base + v * min(c.w, c.h) * 0.30
        x1, y1 = cx + math.cos(a) * base, cy + math.sin(a) * base
        x2, y2 = cx + math.cos(a) * r2, cy + math.sin(a) * r2
        pygame.draw.line(c.s, c.col(i / NBANDS, 0.5 + 0.5 * v),
                         (x1, y1), (x2, y2), max(2, int(c.w / 300)))
    pygame.draw.circle(c.s, c.col(c.bass, 0.9), (cx, cy),
                       int(base * 0.55 * (0.7 + c.bass)), 2)


def s_wave(c):
    cy = c.h / 2
    step = max(1, len(c.wave) // c.w)
    pts = [(x * c.w / (len(c.wave) // step),
            cy + c.wave[x * step] * c.h * 0.38)
           for x in range(len(c.wave) // step)]
    if len(pts) > 1:
        pygame.draw.aalines(c.s, c.col(0.15 + 0.4 * c.energy), False, pts)
        pygame.draw.lines(c.s, c.col(0.5, 0.55), False,
                          [(p[0], p[1] + 3) for p in pts], 1)
        pygame.draw.lines(c.s, c.col(0.8, 0.4), False,
                          [(p[0], p[1] - 3) for p in pts], 1)


def s_ribbons(c):
    amps = (c.bass, c.mid, c.treb)
    for k in range(3):
        pts = []
        for x in range(0, c.w + 8, 8):
            ph = x * (0.006 + k * 0.002) + c.t * (1.0 + k * 0.5)
            bi = int(x / c.w * (NBANDS - 1))
            y = c.h / 2 + math.sin(ph) * c.h * 0.12 * (0.4 + amps[k] * 2.2) \
                + math.sin(ph * 0.37 + k * 2) * c.h * 0.08 \
                + (c.bands[bi] - 0.3) * c.h * 0.1
            pts.append((x, y))
        pygame.draw.lines(c.s, c.col(k / 3 + c.t * 0.02, 0.5 + amps[k]),
                          False, pts, 3)


def s_particles(c):
    st = c.state.setdefault("part", {"p": []})
    cx, cy = c.w / 2, c.h / 2
    n = (90 if c.beat else 0) + int(c.energy * 6)
    for _ in range(n):
        a = random.uniform(0, 2 * math.pi)
        sp = random.uniform(60, 280) * (0.5 + c.energy * 1.6)
        st["p"].append([cx, cy, math.cos(a) * sp, math.sin(a) * sp,
                        1.0, random.random()])
    alive = []
    for p in st["p"]:
        p[0] += p[2] * c.dt
        p[1] += p[3] * c.dt
        p[2] *= 0.985
        p[3] *= 0.985
        p[4] -= c.dt * 0.55
        if p[4] > 0:
            alive.append(p)
            pygame.draw.circle(c.s, c.col(p[5], p[4]),
                               (int(p[0]), int(p[1])), max(1, int(p[4] * 5)))
    st["p"] = alive[-900:]


def s_flow(c):
    st = c.state.setdefault("flow", {"p": [[random.uniform(0, c.w),
                                            random.uniform(0, c.h),
                                            random.random()]
                                           for _ in range(320)]})
    sp = 35 + c.energy * 360
    for p in st["p"]:
        a = (math.sin(p[0] * 0.004 + c.t * 0.25) +
             math.cos(p[1] * 0.004 - c.t * 0.21)) * math.pi
        nx = p[0] + math.cos(a) * sp * c.dt
        ny = p[1] + math.sin(a) * sp * c.dt
        if 0 <= nx < c.w and 0 <= ny < c.h:
            pygame.draw.line(c.s, c.col(p[2], 0.35 + c.energy), (p[0], p[1]),
                             (nx, ny), 2)
            p[0], p[1] = nx, ny
        else:
            p[0], p[1] = random.uniform(0, c.w), random.uniform(0, c.h)


def s_pulse(c):
    st = c.state.setdefault("pulse", {"rings": []})
    cx, cy = c.w // 2, c.h // 2
    if c.beat:
        st["rings"].append([0.0, random.random()])
    keep = []
    maxr = math.hypot(c.w, c.h) / 2
    for ring in st["rings"]:
        ring[0] += c.dt * (260 + 500 * c.energy)
        if ring[0] < maxr:
            keep.append(ring)
            fade = 1 - ring[0] / maxr
            pygame.draw.circle(c.s, c.col(ring[1], fade), (cx, cy),
                               int(ring[0]), max(1, int(6 * fade)))
    st["rings"] = keep[-30:]
    r = min(c.w, c.h) * (0.06 + c.bass * 0.13)
    glow(c.s, c.col(c.bass * 0.7), (cx, cy), r)


def s_stars(c):
    st = c.state.setdefault("stars",
                            {"s": [[random.uniform(-1, 1), random.uniform(-1, 1),
                                    random.uniform(0.05, 1)] for _ in range(260)]})
    cx, cy = c.w / 2, c.h / 2
    sp = 0.12 + c.energy * 1.4
    for s in st["s"]:
        oz = s[2]
        s[2] -= c.dt * sp * 0.4
        if s[2] <= 0.04:
            s[0], s[1], s[2] = random.uniform(-1, 1), random.uniform(-1, 1), 1.0
            oz = 1.0
        x1, y1 = cx + s[0] / oz * cx, cy + s[1] / oz * cy
        x2, y2 = cx + s[0] / s[2] * cx, cy + s[1] / s[2] * cy
        b = (1 - s[2]) * (0.5 + c.energy)
        pygame.draw.line(c.s, c.col(abs(s[0]) * 0.5 + c.treb * 0.5, min(1, b)),
                         (x1, y1), (x2, y2), 2)


def s_kaleido(c):
    cx, cy = c.w / 2, c.h / 2
    sym = 6
    maxr = min(c.w, c.h) * 0.46
    for i in range(0, NBANDS, 2):
        v = c.bands[i]
        if v < 0.04:
            continue
        r = (i / NBANDS) * maxr + 12
        a0 = c.t * 0.4 + i * 0.33
        for k in range(sym):
            a = a0 + k * 2 * math.pi / sym
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            pygame.draw.circle(c.s, c.col(i / NBANDS, 0.4 + 0.6 * v),
                               (int(x), int(y)), max(1, int(v * 14)))
            x, y = cx + math.cos(-a) * r, cy + math.sin(-a) * r
            pygame.draw.circle(c.s, c.col(i / NBANDS, 0.3 + 0.5 * v),
                               (int(x), int(y)), max(1, int(v * 9)))


def s_scope(c):
    cx, cy = c.w / 2, c.h / 2
    sc = min(c.w, c.h) * 0.42
    pts = []
    for i in range(0, 512, 2):
        pts.append((cx + c.wave[i] * sc, cy + c.wave[(i + 96) % 1024] * sc))
    if len(pts) > 1:
        pygame.draw.aalines(c.s, c.col(c.t * 0.05 + c.energy * 0.3,
                                       0.5 + c.energy), False, pts)


def s_grid(c):
    gx, gy = 16, 9
    cw, ch = c.w / gx, c.h / gy
    for j in range(gy):
        for i in range(gx):
            bi = (i * 7 + j * 13) % NBANDS
            v = c.bands[bi]
            r = 2 + v * min(cw, ch) * 0.48
            pygame.draw.circle(c.s, c.col(bi / NBANDS, 0.3 + 0.7 * v),
                               (int(i * cw + cw / 2), int(j * ch + ch / 2)),
                               int(r))


def s_tunnel(c):
    st = c.state.setdefault("tunnel", {"z": 0.0})
    st["z"] += c.dt * (0.25 + c.energy * 0.9)
    cx, cy = c.w / 2, c.h / 2
    maxr = math.hypot(c.w, c.h) * 0.6
    rings = 22
    items = sorted(((k / rings + st["z"]) % 1.0, k) for k in range(rings))
    for zz, k in items:
        r = zz * zz * maxr
        v = c.bands[(k * 3) % NBANDS]
        wob = 1 + v * 0.35
        pygame.draw.circle(c.s, c.col(zz * 0.8 + c.t * 0.03, 0.15 + zz * 0.85),
                           (int(cx + math.sin(c.t + k) * 30 * zz),
                            int(cy + math.cos(c.t * 0.7 + k) * 30 * zz)),
                           max(1, int(r * wob)), max(1, int(1 + zz * 4)))


def s_fireworks(c):
    st = c.state.setdefault("fw", {"r": [], "sp": []})
    if c.beat and len(st["r"]) < 6:
        st["r"].append([random.uniform(c.w * 0.2, c.w * 0.8), c.h,
                        random.uniform(-c.h * 0.75, -c.h * 0.5),
                        random.uniform(c.h * 0.25, c.h * 0.5), random.random()])
    rockets = []
    for r in st["r"]:
        r[1] += r[2] * c.dt
        r[2] += 300 * c.dt
        if r[1] > r[3] and r[2] < 0:  # still rising, not at target yet
            rockets.append(r)
            pygame.draw.circle(c.s, c.col(r[4]), (int(r[0]), int(r[1])), 3)
        else:
            for _ in range(70):
                a = random.uniform(0, 2 * math.pi)
                sp = random.uniform(40, 320)
                st["sp"].append([r[0], r[1], math.cos(a) * sp,
                                 math.sin(a) * sp, 1.0, r[4]])
    st["r"] = rockets
    alive = []
    for p in st["sp"]:
        p[0] += p[2] * c.dt
        p[1] += p[3] * c.dt
        p[3] += 220 * c.dt
        p[4] -= c.dt * 0.7
        if p[4] > 0:
            alive.append(p)
            pygame.draw.circle(c.s, c.col(p[5] + (1 - p[4]) * 0.2, p[4]),
                               (int(p[0]), int(p[1])), max(1, int(p[4] * 3.5)))
    st["sp"] = alive[-1500:]


def s_aurora(c):
    layers = 3
    for k in range(layers):
        pts = [(0, c.h)]
        for x in range(0, c.w + 16, 16):
            bi = int(x / c.w * 31)
            v = c.bands[(bi + k * 11) % NBANDS]
            y = c.h * (0.78 - k * 0.13) - v * c.h * 0.4 \
                - math.sin(x * 0.004 + c.t * (0.6 + k * 0.3)) * c.h * 0.06
            pts.append((x, y))
        pts.append((c.w, c.h))
        tmp = pygame.Surface((c.w, c.h), pygame.SRCALPHA)
        pygame.draw.polygon(tmp, (*c.col(k / layers + c.t * 0.015, 0.8), 60),
                            pts)
        pygame.draw.lines(tmp, (*c.col(k / layers + c.t * 0.015), 160),
                          False, pts[1:-1], 2)
        c.s.blit(tmp, (0, 0))


def s_helix(c):
    amp = c.h * 0.16 * (0.5 + c.mid * 2.0)
    cy = c.h / 2
    last = None
    for x in range(0, c.w + 6, 6):
        ph = x * 0.018 - c.t * 2.2
        bi = int(x / c.w * (NBANDS - 1))
        v = c.bands[bi]
        y1 = cy + math.sin(ph) * amp
        y2 = cy + math.sin(ph + math.pi) * amp
        depth1 = (math.cos(ph) + 1) / 2
        pygame.draw.circle(c.s, c.col(0.2 + v * 0.5, 0.35 + 0.65 * depth1),
                           (x, int(y1)), max(2, int(2 + v * 7)))
        pygame.draw.circle(c.s, c.col(0.7 + v * 0.3, 0.35 + 0.65 * (1 - depth1)),
                           (x, int(y2)), max(2, int(2 + v * 7)))
        if last is not None and x % 36 == 0:
            pygame.draw.line(c.s, c.col(0.5, 0.25), (x, y1), (x, y2), 1)
        last = x


def s_terrain(c):
    st = c.state.setdefault("terr", {"rows": [], "acc": 0.0})
    st["acc"] += c.dt
    if st["acc"] > 0.05:
        st["acc"] = 0.0
        prof = c.bands.reshape(32, 2).mean(axis=1)
        st["rows"].insert(0, prof)
        st["rows"] = st["rows"][:26]
    rows = st["rows"]
    for j in range(len(rows) - 1, -1, -1):
        depth = j / 26.0
        base = c.h * 0.88 - j * c.h * 0.028
        sc = 1 - depth * 0.55
        pts = []
        for i, v in enumerate(rows[j]):
            x = c.w / 2 + (i - 15.5) / 16 * c.w / 2 * sc
            pts.append((x, base - v * c.h * 0.3 * sc))
        pygame.draw.polygon(c.s, (*BG, ), pts + [(pts[-1][0], c.h),
                                                 (pts[0][0], c.h)])
        pygame.draw.lines(c.s, c.col(depth * 0.7 + c.t * 0.02, 1 - depth * 0.8),
                          False, pts, 2)


def s_waterfall(c):
    c.s.scroll(0, 3)
    bw = c.w / NBANDS
    for i in range(NBANDS):
        v = c.bands[i]
        pygame.draw.rect(c.s, c.col(i / NBANDS, v),
                         (i * bw, 0, math.ceil(bw), 3))


def s_orbits(c):
    st = c.state.setdefault("orb", {"a": [random.uniform(0, 6.28)
                                          for _ in range(9)]})
    cx, cy = c.w / 2, c.h / 2
    for i in range(9):
        e = float(c.bands[i * 7:(i + 1) * 7].mean())
        st["a"][i] += c.dt * (0.4 + i * 0.13 + e * 3.0) * (1 if i % 2 else -1)
        rad = min(c.w, c.h) * (0.10 + 0.042 * i) * (1 + c.bass * 0.18)
        x = cx + math.cos(st["a"][i]) * rad * 1.3
        y = cy + math.sin(st["a"][i]) * rad * 0.85
        glow(c.s, c.col(i / 9, 0.5 + e), (x, y), 3 + e * 16)
    pygame.draw.circle(c.s, c.col(c.bass, 0.8), (cx, cy),
                       int(6 + c.bass * 22))


def s_blobs(c):
    st = c.state.setdefault("blob", {"b": [[random.uniform(0, 1),
                                            random.uniform(0, 1),
                                            random.uniform(-0.04, 0.04),
                                            random.uniform(-0.04, 0.04),
                                            random.random()]
                                           for _ in range(12)]})
    tmp = pygame.Surface((c.w, c.h), pygame.SRCALPHA)
    for i, b in enumerate(st["b"]):
        b[0] = (b[0] + b[2] * c.dt) % 1.0
        b[1] = (b[1] + b[3] * c.dt) % 1.0
        e = float(c.bands[(i * 5) % NBANDS])
        r = min(c.w, c.h) * (0.05 + 0.13 * e + 0.04 * c.bass)
        x, y = int(b[0] * c.w), int(b[1] * c.h)
        col = c.col(b[4] + c.t * 0.01)
        for rr, al in ((r, 28), (r * 0.6, 45), (r * 0.3, 70)):
            pygame.draw.circle(tmp, (*col, al), (x, y), max(2, int(rr)))
    c.s.blit(tmp, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


def s_spiral(c):
    cx, cy = c.w / 2, c.h / 2
    maxr = min(c.w, c.h) * 0.48
    n = 220
    pts = []
    for i in range(n):
        f = i / n
        bi = int(f * (NBANDS - 1))
        v = c.bands[bi]
        a = f * math.pi * 7 + c.t * 0.6
        r = f * maxr * (1 + v * 0.3)
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        if v > 0.15:
            pygame.draw.circle(c.s, c.col(f, 0.4 + 0.6 * v), pts[-1],
                               max(1, int(v * 9)))
    pygame.draw.aalines(c.s, c.col(c.t * 0.04, 0.5), False, pts)


def s_rain(c):
    st = c.state.setdefault("rain", {"d": [[random.uniform(0, 1),
                                            random.uniform(-1, 0),
                                            random.uniform(0.4, 1.0)]
                                           for _ in range(140)]})
    for d in st["d"]:
        bi = int(d[0] * (NBANDS - 1))
        v = c.bands[bi]
        sp = (0.25 + v * 2.2) * d[2]
        d[1] += sp * c.dt
        if d[1] > 1.05:
            d[0], d[1] = random.uniform(0, 1), random.uniform(-0.2, 0)
        x, y = d[0] * c.w, d[1] * c.h
        ln = (8 + v * 60) * d[2]
        pygame.draw.line(c.s, c.col(d[0], (0.3 + v) * d[2]),
                         (x, y - ln), (x, y), 2)


def s_ring_eq(c):
    cx, cy = c.w / 2, c.h / 2
    for ring, (lo, hi, base) in enumerate(((0, 8, 0.14), (8, 24, 0.24),
                                           (24, 44, 0.34), (44, 64, 0.44))):
        e = float(c.bands[lo:hi].mean())
        seg = 36
        r = min(c.w, c.h) * base * (1 + e * 0.25)
        for k in range(seg):
            a = k / seg * 2 * math.pi + c.t * (0.2 + ring * 0.12) * \
                (1 if ring % 2 else -1)
            v = c.bands[lo + k % (hi - lo)]
            if v < 0.05:
                continue
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            pygame.draw.circle(c.s, c.col(ring / 4 + v * 0.2, 0.3 + 0.7 * v),
                               (int(x), int(y)), max(1, int(2 + v * 8)))


STYLES = [
    # (name, draw_fn, fade) - fade: None=clear each frame, 0=no clear, n=trail
    ("Spectrum Bars", s_bars, None),
    ("Mirror Bars", s_mirror, None),
    ("Radial Burst", s_circle, 60),
    ("Oscilloscope", s_wave, 70),
    ("Liquid Ribbons", s_ribbons, 45),
    ("Particle Burst", s_particles, 40),
    ("Flow Field", s_flow, 14),
    ("Beat Pulse", s_pulse, 50),
    ("Starfield", s_stars, 80),
    ("Kaleidoscope", s_kaleido, 22),
    ("Lissajous", s_scope, 35),
    ("Dot Matrix", s_grid, None),
    ("Tunnel", s_tunnel, 90),
    ("Fireworks", s_fireworks, 28),
    ("Aurora", s_aurora, None),
    ("DNA Helix", s_helix, 55),
    ("Terrain", s_terrain, None),
    ("Waterfall", s_waterfall, 0),
    ("Orbits", s_orbits, 30),
    ("Soft Blobs", s_blobs, 25),
    ("Spiral Galaxy", s_spiral, 45),
    ("Neon Rain", s_rain, 38),
    ("Ring EQ", s_ring_eq, 32),
]


# ================================================================ settings
def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


# ================================================================ main
def main():
    pygame.init()
    pygame.display.set_caption("Flow")
    info = pygame.display.Info()
    desk_w, desk_h = info.current_w, info.current_h

    cfg = load_settings()
    fullscreen = cfg.get("fullscreen", True) and not TEST_MODE
    win_size = (min(1100, desk_w - 80), min(620, desk_h - 80))

    def set_mode():
        if fullscreen:
            return pygame.display.set_mode((desk_w, desk_h), pygame.NOFRAME)
        return pygame.display.set_mode(win_size, pygame.NOFRAME)

    screen = set_mode()
    try:
        from pygame._sdl2.video import Window
        sdlwin = Window.from_display_module()
    except Exception:
        sdlwin = None

    engine = AudioEngine()
    an = Analyzer(engine)
    an.sens = cfg.get("sens", 1.0)
    si = cfg.get("style", 0) % len(STYLES)
    pi = cfg.get("palette", 0) % len(PALETTES)
    col = make_col(pi)
    auto_cycle = cfg.get("auto", False)
    auto_t = 0.0

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,arial", 16)
    bigfont = pygame.font.SysFont("segoeui,arial", 22, bold=True)
    state = {}
    hud_msg, hud_until = "", 0.0
    show_help = not cfg and not TEST_MODE
    help_until = time.time() + 8 if show_help else 0
    dragging = False
    drag_off = (0, 0)
    t0 = time.time()
    fade_layer = pygame.Surface((screen.get_width(), screen.get_height()))
    fade_layer.fill(BG)

    def notify(msg):
        nonlocal hud_msg, hud_until
        hud_msg, hud_until = msg, time.time() + 2.2

    def switch_style(new_i):
        nonlocal si
        si = new_i % len(STYLES)
        state.clear()
        screen.fill(BG)
        notify(STYLES[si][0])

    notify(f"{STYLES[si][0]}  |  {engine.device_name()}")

    running = True
    frame = 0
    test_shots = []
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        now = time.time()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_RIGHT:
                    switch_style(si + 1)
                elif k == pygame.K_LEFT:
                    switch_style(si - 1)
                elif k == pygame.K_SPACE:
                    switch_style(random.randrange(len(STYLES)))
                elif k in (pygame.K_UP, pygame.K_DOWN):
                    pi = (pi + (1 if k == pygame.K_UP else -1)) % len(PALETTES)
                    col = make_col(pi)
                    notify("Palette: " + PALETTES[pi][0])
                elif k == pygame.K_a:
                    auto_cycle = not auto_cycle
                    auto_t = 0
                    notify("Auto-cycle " + ("ON" if auto_cycle else "OFF"))
                elif k == pygame.K_d:
                    engine.next_device()
                    notify(engine.device_name())
                elif k == pygame.K_f:
                    fullscreen = not fullscreen
                    screen = set_mode()
                    fade_layer = pygame.Surface(screen.get_size())
                    fade_layer.fill(BG)
                    state.clear()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    an.sens = min(4.0, an.sens * 1.2)
                    notify(f"Sensitivity {an.sens:.1f}x")
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    an.sens = max(0.2, an.sens / 1.2)
                    notify(f"Sensitivity {an.sens:.1f}x")
                elif k == pygame.K_h:
                    show_help = not show_help
                    help_until = now + 3600 if show_help else 0
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if not fullscreen and sdlwin:
                    dragging = True
                    mx, my = pygame.mouse.get_pos()
                    wx, wy = sdlwin.position
                    drag_off = (mx, my)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                switch_style(si + 1)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging and sdlwin:
                wx, wy = sdlwin.position
                sdlwin.position = (wx + ev.pos[0] - drag_off[0],
                                   wy + ev.pos[1] - drag_off[1])

        if auto_cycle:
            auto_t += dt
            if auto_t > 25:
                auto_t = 0
                switch_style(si + 1)

        an.update(dt)

        name, fn, fade = STYLES[si]
        if fade is None:
            screen.fill(BG)
        elif fade > 0:
            fade_layer.set_alpha(fade)
            screen.blit(fade_layer, (0, 0))

        c = Ctx()
        c.s = screen
        c.w, c.h = screen.get_width(), screen.get_height()
        c.bands, c.wave = an.bands, an.wave
        c.bass, c.mid, c.treb = an.bass, an.mid, an.treb
        c.energy, c.beat = an.energy, an.beat
        c.t, c.dt = now - t0, dt
        c.col = col
        c.state = state
        try:
            fn(c)
        except Exception as e:
            notify(f"Style error: {e}")

        if now < hud_until:
            txt = bigfont.render(hud_msg, True, (235, 235, 245))
            sh = pygame.Surface((txt.get_width() + 24, txt.get_height() + 12),
                                pygame.SRCALPHA)
            sh.fill((0, 0, 0, 140))
            screen.blit(sh, (16, 16))
            screen.blit(txt, (28, 22))

        if show_help and now < help_until:
            lines = ["Left/Right  style      Up/Down  palette",
                     "Space  random          A  auto-cycle",
                     "D  audio source        F  fullscreen/window",
                     "+/-  sensitivity       drag  move window",
                     "H  hide help           Esc  quit"]
            hs = pygame.Surface((430, 24 * len(lines) + 20), pygame.SRCALPHA)
            hs.fill((0, 0, 0, 160))
            for i, ln in enumerate(lines):
                hs.blit(font.render(ln, True, (220, 220, 230)), (14, 10 + i * 24))
            screen.blit(hs, (16, screen.get_height() - hs.get_height() - 16))

        pygame.display.flip()

        if TEST_MODE:
            frame += 1
            per = 45
            if frame % per == 0:
                shot = f"/tmp/style_{si:02d}_{name.replace(' ', '_')}.png"
                pygame.image.save(screen, shot)
                test_shots.append(shot)
                if si == len(STYLES) - 1:
                    running = False
                else:
                    switch_style(si + 1)

    save_settings(dict(style=si, palette=pi, sens=an.sens,
                       fullscreen=fullscreen, auto=auto_cycle))
    engine.stop()
    pygame.quit()
    if TEST_MODE:
        print("TEST OK:", len(test_shots), "styles rendered")
        for s in test_shots:
            print(" ", s)


if __name__ == "__main__":
    main()
