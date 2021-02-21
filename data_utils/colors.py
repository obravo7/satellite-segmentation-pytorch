
def hex_to_rgb(h: str) -> tuple:

    h = h.lstrip('#') if '#' in h else h

    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
