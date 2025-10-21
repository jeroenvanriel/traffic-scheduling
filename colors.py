from matplotlib.colors import ListedColormap

from_rgb = lambda *t: tuple(c / 255 for c in t)
pastelblue   = from_rgb(179, 205, 227)
pastelorange = from_rgb(251, 180, 174)
pastelgreen  = from_rgb(170, 210, 170)

pastel_colors = [pastelblue, pastelorange, pastelgreen]
pastel_cmap = ListedColormap(pastel_colors, name='pastel_colors')
