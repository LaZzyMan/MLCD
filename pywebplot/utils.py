def clamp(x):
    return max(0, min(x, 255))


def rgb2hex(r, g, b):
    if 0 < r < 1:
        r = r * 255
        g = g * 255
        b = b * 255
    return '#%02x%02x%02x' % (clamp(r), clamp(g), clamp(b))


def hex2rgb(hex):
    rgb = [(hex >> 16) & 0xff,
           (hex >> 8) & 0xff,
           hex & 0xff
           ]
    return rgb
