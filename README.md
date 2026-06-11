Flow — Audio Visualizer

Borderless desktop visualizer that reacts to whatever your PC is playing (Spotify, games, anything) or a microphone. 23 styles, 8 color palettes.

Setup (one time)


Install Python from https://www.python.org/downloads/ — check "Add python.exe to PATH" during install.
Double-click Start Visualizer.bat. First run installs three packages automatically, then launches.


Hotkeys

KeyAction← / →previous / next style↑ / ↓color paletteSpacerandom styleAauto-cycle styles every 25sDswitch audio source (system audio ↔ microphones)Ffullscreen ↔ borderless window+ / −sensitivityHhelp overlayDragmove window (windowed mode)Right-clicknext styleEsc / Qquit

Notes


Starts on system audio by default — play music and it flows.
Settings (style, palette, sensitivity) save automatically on exit.
It captures all system audio mixed together; isolating one app (e.g. only Spotify while a game runs) needs virtual-audio-cable software, which Windows doesn't provide natively.
Troubleshooting: python visualizer.py --test renders every style with synthetic audio and reports OK.
