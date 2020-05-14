import os
import cv2
import numpy as np
import copy
from shapely.geometry import Polygon

FILTER_EPS = 1e-5
def read_binary_image(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"contour of {file_path} : {len(contours)}")

    filtered_contour = []
    polygons_lists = [Polygon(np.array(contour).squeeze()) for contour in contours]

    ## CALCULATE THERSHOLD BY MAX AREA
    area_lists = map(lambda contour_poly : contour_poly.area, polygons_lists)
    filter_thershold = max(area_lists) * FILTER_EPS
    filtered_polygon_lists = list(filter(lambda contour_poly : contour_poly.area > filter_thershold, polygons_lists))


    assert len(filtered_polygon_lists) > 0

    #### HERE ASSUME ONLY 1 exterior polygon
    max_index = np.argmax([polygon.area for polygon in filtered_polygon_lists])
    exterior_polygon = filtered_polygon_lists.pop(max_index)
    exterior_polygon_coords = np.array(exterior_polygon.exterior.coords)
    interior_polygons_coords = [np.array(interior_polygon.exterior.coords) for interior_polygon in  filtered_polygon_lists]

    return exterior_polygon_coords, interior_polygons_coords


def export_contour_as_text(output_path, contours):
    exterior_polygon_coords, interior_polygons_coords = contours
    with open(output_path, 'w') as file:
        print(*['{0} {1}'.format(x, y) for x, y in exterior_polygon_coords], sep=',', file=file)
        for interior_polygon_coords in interior_polygons_coords:
            print(*['{0} {1}'.format(x, y) for x, y in interior_polygon_coords], sep=',', file=file)


def transform_all_binary_images(root_path):
    if os.path.isdir(root_path):
        # process the whole directory
        files = os.listdir(root_path)
        files = [file for file in files if file[0] != '.' and file[:2] != '__']
        for file in files:
            try:
                transform_all_binary_images(os.path.join(root_path, file))
            except:
                continue
    elif os.path.isfile(root_path):
        target_contour_name = root_path[:-4] + '.txt'
        if '.txt' not in root_path and not os.path.exists(target_contour_name):
            export_contour_as_text(target_contour_name, read_binary_image(root_path))
    else:
        raise FileNotFoundError('Invalid Filename')


if __name__ == '__main__':
    transform_all_binary_images(r"/home/edwardhui/data/large_result/sig_logo/silhoutee")
