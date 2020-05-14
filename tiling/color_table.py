
color_map = {
    0 : ((255, 186, 186, 255), (127, 127, 127)), # light red
    1 : ((255, 244, 186, 255), (127, 127, 127)), # light yellow
    2 : ((186, 244, 186, 255), (127, 127, 127)), # light green
    3 : ((186, 217, 255, 255), (127, 127, 127)), # light blue
}

color_30_60_90 = ((186, 217, 255, 255), (127, 127, 127))
color_30_60_90_reflected = ((186, 244, 186, 255), (127, 127, 127))
color_equilateral = ((255, 244, 186, 255), (127, 127, 127))
color_rectangle = ((255, 186, 186, 255), (127, 127, 127))

tile_set_color_map = {
    '30-60-90' : {
        0 : color_30_60_90,
        1 : color_30_60_90_reflected,
    },
    '30-60-90+equilateral' : {
        0 : color_30_60_90,
        1 : color_equilateral,
        2 : color_30_60_90_reflected,
    },
    '30-60-90+rectangle' : {
        0: color_rectangle,
        1: color_30_60_90,
        2: color_rectangle,
        3: color_30_60_90_reflected,
    },
    '45-45-90+parlgram' : {
        0: color_rectangle,
        1: color_30_60_90,
    },
    '45-45-90+rectangle' : {
        0: color_rectangle,
        1: color_30_60_90,
    }
}