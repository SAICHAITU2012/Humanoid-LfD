"""Assemble a highlight reel from throw clips: title card -> throws (wide +
slow-mo bullseye close-ups) -> match-conditions throw -> scoreboard card.

Cards are rendered with Pillow (this ffmpeg build has no drawtext filter);
clips are re-encoded and concatenated with ffmpeg. CPU-light.

Run:
  ./.venv/bin/python build_reel.py --out demo/reel.mp4 \
      --throw demo/throw_seed92024_err0.9cm --match demo/match_nominal \
      --stats "line1" --stats "line2" ...
Each --throw/--match value is the clip base name (without _wide/_bullseye.mp4).
"""
import argparse, os, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
W, H, FPS = 800, 480, 30
BG = (0x10, 0x14, 0x1C)
WHITE = (255, 255, 255)
MUTED = (0xB6, 0xC2, 0xD9)

ap = argparse.ArgumentParser()
ap.add_argument('--throw', action='append', default=[], help='base name of a normal throw pair')
ap.add_argument('--match', action='append', default=[], help='base name of a match-conditions throw pair')
ap.add_argument('--title', default='G1 Learns Darts')
ap.add_argument('--subtitle', default='five fingers - one bullseye')
ap.add_argument('--stats', action='append', default=[], help='scoreboard lines (max 6)')
ap.add_argument('--out', default='demo/reel.mp4')
args = ap.parse_args()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('ffmpeg failed:\n' + r.stderr[-1500:])


def text_card(lines, seconds, path):
    """Solid dark card with one big first line + smaller lines (Pillow -> mp4)."""
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 150
    for i, line in enumerate(lines):
        big = i == 0 and len(lines) > 1
        font = ImageFont.truetype(FONT_BOLD if big else FONT_REG, 46 if big else 26)
        color = WHITE if big else MUTED
        w = d.textlength(line, font=font)
        d.text(((W - w) / 2, y), line, font=font, fill=color)
        y += (46 if big else 26) + 26
    png = path.replace('.mp4', '.png')
    img.save(png)
    run(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-i', png, '-t', str(seconds),
         '-r', str(FPS), '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
         '-pix_fmt', 'yuv420p', path])


def clip(src, path, slowmo=False):
    vf = f'setpts={2.0}*PTS' if slowmo else 'null'
    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', src, '-vf', vf, '-r', str(FPS),
           '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
           '-pix_fmt', 'yuv420p', path]
    if slowmo:
        cmd += ['-an']
    run(cmd)


segments = []
tmp = tempfile.mkdtemp(prefix='reel_')

text_card([args.title, args.subtitle], 2.6, p := os.path.join(tmp, '00_title.mp4'))
segments.append(p)

for i, base in enumerate(args.throw, 1):
    wide, close = base + '_wide.mp4', base + '_bullseye.mp4'
    if os.path.exists(wide):
        clip(wide, p := os.path.join(tmp, f'{i:02d}_wide.mp4'))
        segments.append(p)
    if os.path.exists(close):
        clip(close, p := os.path.join(tmp, f'{i:02d}_close.mp4'), slowmo=True)
        segments.append(p)

for base in args.match:
    text_card(['MATCH CONDITIONS', 'calibrated throw - the aim is set for the standard dart'], 1.8,
              p := os.path.join(tmp, 'match_card.mp4'))
    segments.append(p)
    close = base + '_bullseye.mp4'
    if os.path.exists(close):
        clip(close, p := os.path.join(tmp, 'match_close.mp4'), slowmo=True)
        segments.append(p)

if args.stats:
    text_card(['RESULTS'] + args.stats[:6], 4.5, p := os.path.join(tmp, '99_stats.mp4'))
    segments.append(p)

listfile = os.path.join(tmp, 'list.txt')
with open(listfile, 'w') as fh:
    for s in segments:
        fh.write(f"file '{s}'\n")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', listfile,
     '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
     '-movflags', '+faststart', args.out])
print('reel ->', args.out, f'({len(segments)} segments)')
