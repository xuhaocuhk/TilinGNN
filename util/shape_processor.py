import numpy as np
import multiprocessing
from shapely.geometry import Polygon
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default

# read coutour from a local file
def getSVGShapeAsNp(filename):
    for line in open(filename):
        listWords = line.split(",")
    listWords = np.array(listWords)
    coords = np.array(list(map(lambda word: [float(word.split(" ")[0]), float(word.split(" ")[1])], listWords)))
    return coords

## read get contour in shapely
def load_polygons(filename):
    ## ASSUME : line 0 -> exterior line 1.... -> interior
    lines = list(open(filename))
    ### exterior
    exterior_list_word = lines[0].split(",")
    exterior_coords = np.array(list(map(lambda word: [float(word.split(" ")[0]), float(word.split(" ")[1])], exterior_list_word)))

    ## interior
    all_interior_coords = []
    for line in lines[1:]:
        interior_list_word = line.split(",")
        interior_coords = np.array(list(map(lambda word: [float(word.split(" ")[0]), float(word.split(" ")[1])], interior_list_word)))
        all_interior_coords.append(interior_coords)

    return exterior_coords, all_interior_coords

if __name__ == "__main__":
    polygon = load_polygons('/Users/edwardhui/Desktop/previous_file/learn_assemble/data/silhouette_files/flower_with_hole.txt')
    print(polygon)

