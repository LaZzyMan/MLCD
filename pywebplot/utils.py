import os
import shutil


def clamp(x):
    return int(max(0, min(x, 255)))


def rgb2hex(r, g, b):
    if r*g*b <= 1:
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
    return '#%02x%02x%02x' % (clamp(r), clamp(g), clamp(b))


def hex2rgb(hex):
    rgb = [(hex >> 16) & 0xff,
           (hex >> 8) & 0xff,
           hex & 0xff
           ]
    return rgb


def mkdir():
    if not os.path.exists('dist'):
        os.makedirs('dist')
    if not os.path.exists('dist/data'):
        os.mkdir('dist/data')
    if not os.path.exists('dist/js'):
        os.mkdir('dist/js')
    if not os.path.exists('dist/style'):
        os.mkdir('dist/style')
    if not os.path.exists('dist/image'):
        os.mkdir('dist/image')
    if not os.path.exists('dist/style/index.css'):
        shutil.copy('../src/style/index.css', 'dist/style')

