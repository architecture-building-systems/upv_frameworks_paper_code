import skimage as ski
import numpy as np
import pandas as pd
import cv2
from sklearn.datasets import make_blobs
import pickle 
import math

def calculate_ratio(mask_type, coverage):
    if mask_type=="rows":
        mask_type = "row"
    if mask_type=="columns":
        mask_type = "column"
    with open('ratio_models_linear.pkl','rb') as f:
        models = pickle.load(f)
    model = models[mask_type]    
    return np.clip(model.predict(np.array(coverage).reshape(-1,1)),0,1)[0][0]


def open_image():
    img_fp = "tree2.png"
    # read the image file
    img = cv2.imread(img_fp, 2)

    scale_percent = 60  # percent of original size
    width = 764
    height = 275
    dim = (width, height)

    # resize image
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # converting to its binary form
    # bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # bw = np.flipud(np.clip(img,0,1))
    bw = np.flipud(img)
    return np.where(bw == 255, 0, 1)


def make_tree_mask(coverage_factor=0.25, n_clusters=100, cluster_spread=7, seed=42):
    tree_array = open_image()

    original_shape = tree_array.shape
    flat_tree_arr = tree_array.flatten()
    total_cells = len(flat_tree_arr)

    # calculate how many cells are branches
    branch_cell_count = np.sum(flat_tree_arr)

    # calculate the number of leaves that must remain
    n_cells_cover = int(total_cells * coverage_factor)
    # print(n_cells_cover, "cells must be covered.")

    # calculate the number of cells that must be added
    n_cells_add = n_cells_cover - int(branch_cell_count)
    # print(n_cells_add, "cells must be made into leaves.")

    current_coverage = branch_cell_count / total_cells

    print(
        "Using a while loop to generate clusters of leaves along branches until coverage target is met."
    )
    n = 0
    while current_coverage < coverage_factor:
        if n != 0:
            n_clusters = 1
        
        # get coordinates where there are branches
        y, x = np.where(tree_array == 1)

        # using a rnadom generator select indices to sample the point locations from
        rng = np.random.default_rng()
        cell_add_idx = rng.choice(len(y) - 1, size=50, replace=False)
        center_pts = list(zip(x[cell_add_idx], np.clip(y[cell_add_idx], 50, None)))

        X, y = make_blobs(
            n_samples=n_cells_add,
            n_features=n_clusters,
            cluster_std=cluster_spread,
            centers=center_pts,
        )
        X = np.array(X, dtype=int)
        x_coords = np.clip(X[:, 0], 0, original_shape[1] - 1)
        y_coords = np.clip(X[:, 1], 0, original_shape[0] - 1)
        # X = np.array(list(zip(x_coords,y_coords)))

        for pt in list(zip(y_coords, x_coords)):
            tree_array[pt[0], pt[1]] = 1

        # print(np.sum(tree_array), "cells are covered.")
        current_coverage = np.sum(tree_array) / total_cells
        n = n + 1
        if n==1000:
            print(f"Ending early. Coverage reached {current_coverage}")
            current_coverage = coverage_factor
    return np.array(tree_array,dtype=float)


def create_mask_rows(arr, coverage_factor=0.25):
    m, n = arr.shape
    cell_count = m * n
    cells_to_cover = int(cell_count * coverage_factor)
    rows_to_cover = int(cells_to_cover / n)
    arr[:rows_to_cover, :] = 1  # Set the lower half of the array to 1
    return arr


def create_mask_columns(arr, coverage_factor=0.25):
    m, n = arr.shape
    cell_count = m * n
    cells_to_cover = int(cell_count * coverage_factor)
    cols_to_cover = int(cells_to_cover / m)
    arr[:, :cols_to_cover] = 1  # Set the lower half of the array to 1
    return arr


def create_rectangle_floating(
    arr, origin=None, dims=None, square=False, coverage_factor=0.25
):
    """_summary_

    Args:
        arr (_type_): _description_
        origin (_type_, optional): _description_. Defaults to None.
        dims (tuple, optional): tuple of (height, width). Defaults to None.
        square (bool, optional): _description_. Defaults to False.
        coverage_factor (float, optional): _description_. Defaults to 0.25.

    Returns:
        _type_: _description_
    """
    m, n = arr.shape
    if origin == None:
        # origin center of the shape by default is the center of the array
        origin = (int(m / 2), int(n / 2))

    cell_count = m * n

    if dims == None:
        # this will be based on coverage factor
        cells_to_cover = cell_count * coverage_factor
        if square == True:
            # build a square
            width = int(np.sqrt(cells_to_cover))
            height = width
            new_origin = (int(origin[0] - (height / 2)), int(origin[1] - (width / 2)))

            # rr, cc = ski.draw.rectangle(start=new_origin,
            #                    extent=(height, width),
            #                    shape=arr.shape)

            rr, cc = ski.draw.rectangle(
                start=new_origin,
                end=(int(new_origin[0] + height), int(new_origin[1] + width)),
                shape=arr.shape,
            )
            arr[rr, cc] = 1
            # if the square extends past the bounds of the original array then the final area will not be enough
            # so here we went a loop to incremently make the actual shading polygon larger until the area is acheived
            a, b = arr.shape
            if np.sum(arr) / (a * b) < coverage_factor:
                print("Size not achieved. Making new polygon iteratively.")

                # start the while loop
                n = 0
                while_origin = new_origin
                while_end = (int(0 + height), int(while_origin[1] + width))
                while np.sum(arr) / (a * b) < coverage_factor:
                    if a <= b:
                        while_origin = (0, int(while_origin[1] - n))
                        while_end = (0 + height, int(while_end[1] + n))

                        rr_new, cc_new = ski.draw.rectangle(
                            start=while_origin, end=while_end, shape=arr.shape
                        )
                    else:
                        while_origin = (int(while_origin[1] - n), 0)
                        while_end = (int(while_end[1] + n), 0 + height)

                        rr_new, cc_new = ski.draw.rectangle(
                            start=while_origin, end=while_end, shape=arr.shape
                        )

                    # remask
                    arr[rr_new, cc_new] = 1

                    # store the new shape for recalculating the area
                    a, b = arr.shape

                    # add an increment
                    n = n + 1
            return arr

        else:
            # build a rectangle of the area using the same aspect ratio as the surface
            # aspect_ratio = max(m,n) / min(m,n)
            aspect_ratio = n / m  # length to width
            width = np.sqrt(cells_to_cover * aspect_ratio)
            height = width / aspect_ratio

            new_origin = (int(origin[0] - (height / 2)), int(origin[1] - (width / 2)))

            rr, cc = ski.draw.rectangle(
                start=new_origin, extent=(height, width), shape=arr.shape
            )
            arr[rr, cc] = 1
            return arr

    else:
        height = dims[0]
        width = dims[1]

        new_origin = (int(origin[0] - (height / 2)), int(origin[1] - (width / 2)))

        rr, cc = ski.draw.rectangle(
            start=new_origin, extent=(height, width), shape=arr.shape
        )
        arr[rr, cc] = 1
        return arr
        # this will be based on the dimensions and a
        # while loop will be used to achieve the coverage factor


def create_triangle(arr, coverage_factor=0.25):
    m, n = arr.shape

    # Calculate the total area of the array
    total_array_area = m * n

    # Calculate the maximum base length and height length
    max_base_length = n-1
    max_height_length = m-1
    # Calculate the triangle area based on the coverage factor
    triangle_area = coverage_factor * total_array_area

    # Calculate the length of a right isoceles triangle
    # triangle_side_length = int(np.sqrt(triangle_area))
    triangle_side_length = int(np.sqrt((2 * triangle_area) / math.tan(math.radians(45))))
    if (triangle_side_length > max_base_length) or (
        triangle_side_length > max_height_length
    ):
        # if the height is less than or equal to base length set this as the first triangle side
        if max_height_length <= max_base_length:
            first_length = max_height_length
            # calculate second length based on area and first length
            base_length = int((2 * triangle_area) / first_length)

            # create list of points along the triangle starting from origin
            r = np.array([0, first_length - 1, 0])
            c = np.array([0, 0, base_length - 1])

            # draw the triangle using scikit-image and extract the coordiantes
            rr, cc = ski.draw.polygon(r, c)

            # clip the coordinates if there is a possibility of them exceeding the limits
            if base_length >= max_base_length:
                rr = np.clip(rr, 0, max_height_length - 1)
                cc = np.clip(cc, 0, max_base_length - 1)

            # mask the array for shading
            arr[rr, cc] = 1

            # if the triangle extends past the bounds of the original array then the final area will not be enough
            # so here we went a loop to incremently make the actual shading polygon larger until the area is acheived
            a, b = arr.shape
            
            if np.sum(arr) / (a * b) < coverage_factor:
                print("Size not achieved. Making new polygon iteratively.")
                # start the while loop
                n = 0
                while np.sum(arr) / (a * b) < coverage_factor:

                    # create new coordinates, r_new is where the incrementing is happening
                    r_new = np.array(
                        [0, max_height_length, np.max(rr[cc == np.argmax(cc)]) + n, 0]
                    )
                    if coverage_factor < 0.50:
                        c_new = np.array([0, 0, base_length+n, base_length+n])
                    else:
                        c_new = np.array([0, 0, max_base_length, max_base_length])
                    # draw the polygon
                    rr_new, cc_new = ski.draw.polygon(r_new, c_new)

                    # clip like before
                    if base_length >= max_base_length:
                        rr_new = np.clip(rr_new, 0, max_height_length - 1)
                        cc_new = np.clip(cc_new, 0, max_base_length - 1)

                    # remask
                    arr[rr_new, cc_new] = 1

                    # store the new shape for recalculating the area
                    a, b = arr.shape

                    # add an increment
                    n = n + 1

        else:

            first_length = max_base_length
            # calculate second length based on area and first length
            height_length = int((2 * triangle_area) / first_length)

            # create list of points along the triangle starting from origin
            r = np.array([0, height_length - 1, 0])
            c = np.array([0, 0, first_length - 1])

            # draw the triangle and mask the array with the triangle
            rr, cc = ski.draw.polygon(r, c)

            # clip the coordinates if there is a possibility of them exceeding the limits
            if height_length >= max_height_length:
                rr = np.clip(rr, 0, max_height_length - 1)
                cc = np.clip(cc, 0, max_base_length - 1)

            # mask the array for shading
            arr[rr, cc] = 1

            # if the triangle extends past the bounds of the original array then the final area will not be enough
            # so here we went a loop to incremently make the actual shading polygon larger until the area is acheived
            a, b = arr.shape
            if np.sum(arr) / (a * b) < coverage_factor:
                print("Size not achieved. Making new polygon iteratively.")

                # start the while loop
                n = 0
                while np.sum(arr) / (a * b) < coverage_factor:

                    # create new coordinates, r_new is where the incrementing is happening
                    r_new = np.array([0, max_height_length, max_height_length, 0])
                    c_new = np.array(
                        [0, 0, np.max(cc[rr == np.argmax(rr)]) + n, max_base_length]
                    )

                    # draw the polygon
                    rr_new, cc_new = ski.draw.polygon(r_new, c_new)

                    # clip like before
                    if base_length >= max_base_length:
                        rr_new = np.clip(rr_new, 0, max_height_length - 1)
                        cc_new = np.clip(cc_new, 0, max_base_length - 1)

                    # remask
                    arr[rr_new, cc_new] = 1

                    # store the new shape for recalculating the area
                    a, b = arr.shape

                    # add an increment
                    n = n + 1

    else:
        triangle_indices = []

        for i in range(triangle_side_length):
            # Generate row and column indices for the triangle
            row = 0 + i  # + i#m - i - 1
            col_indices = np.arange(triangle_side_length - i)

            # Add the tuple of indices to the list
            triangle_indices.extend([(row, col) for col in col_indices])
        # Set the values in the array to 1 based on the generated indices
        for row, col in triangle_indices:
            if row >= m:
                row = m - 1

            if col >= n:
                col = n - 1
            arr[row, col] = 1

    return arr


def create_circle(arr, coverage_factor=0.25):
    m, n = arr.shape

    center_y = int(m / 2)
    center_x = int(n / 2)
    center = (center_y, center_x)

    total_array_area = m * n
    circle_area = coverage_factor * total_array_area

    radius = np.sqrt(circle_area / np.pi)

    rr, cc = ski.draw.disk(center, radius, shape=arr.shape)
    arr[rr, cc] = 1

    # if the circle extends past the bounds of the original array then the final area will not be enough
    # so here we went a loop to incremently make the actual shading polygon larger until the area is acheived
    a, b = arr.shape
    if np.sum(arr) / (a * b) < coverage_factor:
        print("Size not achieved. Making new polygon iteratively.")

        # start the while loop
        n = 0
        while np.sum(arr) / (a * b) < coverage_factor:

            rr_new, cc_new = ski.draw.disk(center, radius + n, shape=arr.shape)

            # remask
            arr[rr_new, cc_new] = 1

            # store the new shape for recalculating the area
            a, b = arr.shape

            # add an increment
            n = n + 1

    return arr


def create_ellipse(arr, primary="y", coverage_factor=0.25):
    m, n = arr.shape

    center_y = int(m / 2)
    center_x = int(n / 2)

    total_array_area = m * n
    ellipse_area = coverage_factor * total_array_area

    if primary == "y":
        r_radius = m // 2
        c_radius = int(ellipse_area / (np.pi * r_radius))  # based on area
    else:
        c_radius = n // 2
        r_radius = int(ellipse_area / (np.pi * c_radius))  # based on area

    # rr, cc = ski.draw.ellipse(center_y, center_x, r_radius, c_radius, shape=arr.shape, rotation=0.0)
    rr, cc = ski.draw.ellipse(
        center_y, center_x, r_radius, c_radius, shape=arr.shape, rotation=0.0
    )
    arr[rr, cc] = 1

    # if the ellipse extends past the bounds of the original array then the final area will not be enough
    # so here we went a loop to incremently make the actual shading polygon larger until the area is acheived
    a, b = arr.shape
    if np.sum(arr) / (a * b) < coverage_factor:
        print("Size not achieved. Making new polygon iteratively.")

        # start the while loop
        n = 0
        while np.sum(arr) / (a * b) < coverage_factor:

            rr_new, cc_new = ski.draw.ellipse(
                center_y,
                center_x,
                r_radius + n,
                c_radius + n,
                shape=arr.shape,
                rotation=0.0,
            )
            arr[rr_new, cc_new] = 1
            # remask
            arr[rr_new, cc_new] = 1

            # store the new shape for recalculating the area
            a, b = arr.shape

            # add an increment
            n = n + 1
    return arr


def create_random_squares(arr, square_area, seed=42, coverage_factor=0.25):
    np.random.seed(seed)
    m, n = arr.shape

    total_array_area = m * n

    # calculate the number of squares needed based on the array area size
    total_squares = int((1 * total_array_area) // square_area)
    n_squares = int((coverage_factor * total_array_area) // square_area)

    side_length = int(np.sqrt(square_area))

    # combine all possible origins into a list of tuples
    r_possible = np.arange(0, m - 1)
    c_possible = np.arange(0, n - 1)
    
    r_c_comb = []
    for r in r_possible:
        for c in c_possible:
            r_c_comb.append((r, c))
    
    # use a rnadom number generator to create a list of indicies from which to grab the origins
    # rng = np.random.default_rng(seed=seed)
    # r_c_idx = rng.choice(len(r_c_comb) - 1, size=total_squares, replace=False)[0:n_squares-1]
    
    # create an idx for the length of the combinations
    r_c_idx_og = np.arange(0, len(r_c_comb)-1)
    # shuffle it
    np.random.shuffle(r_c_idx_og)
    # split it for the initial selection based on the n_Squares needed
    r_c_idx = r_c_idx_og[0:n_squares-1]
    # the remainder will be used for the while loop
    r_c_not_selected_idx = r_c_idx_og[n_squares:]
    
    # using the randomly selected indicies create a sub array of only the selected origins
    r_c_comb_selected = np.array(r_c_comb)[r_c_idx]
    # select the origins that were not used    
    r_c_not_selected = np.array(r_c_comb)[r_c_not_selected_idx]
    
    # create squares at the selected points
    for r_c in r_c_comb_selected:
        mask_arr = create_rectangle_floating(
            arr,
            origin=(r_c[0], r_c[1]),
            square=True,
            dims=(side_length, side_length),
        )

    # use while loop to continuously add squares until coverage is met
    count = 0
    while np.sum(mask_arr) / (m * n) < coverage_factor:
        if count == 0:
            print(
                "Coverage not achieved due to overlap. Using a while loop to iteratively complete the coverage target."
            )

        # pulll from the shuffle not selected origin list based on the while loop count
        origin = r_c_not_selected[count]

        # create the square
        mask_arr = create_rectangle_floating(
            mask_arr,
            origin=(origin[0], origin[1]),
            square=True,
            dims=(side_length, side_length),
        )

        # store the new shape for recalculating the area at the start of the while
        m, n = mask_arr.shape

        # delete the used origin from the no selected list
        # np.delete(r_c_not_selected, idx, axis=0)

        # increase the count to bypass the print statement
        count = count + 1

    return mask_arr


def generate_mask_arr(
    sensor_pts_xyz_arr,
    shape,
    coverage_factor=0.25,
    seed=42,
    small_square_area=36,
    large_square_area=1600,
):
    if coverage_factor>1:
        coverage_factor = coverage_factor / 100
    irrad_x_unique = pd.Series(sensor_pts_xyz_arr[:, 0]).unique().shape[0]
    irrad_z_unique = pd.Series(sensor_pts_xyz_arr[:, 2]).unique().shape[0]

    base_array = np.zeros([irrad_z_unique, irrad_x_unique])
    m, n = base_array.shape

    if shape == "rows":
        mask_arr = create_mask_rows(base_array, coverage_factor)

    elif shape == "columns":
        mask_arr = create_mask_columns(base_array, coverage_factor)

    elif shape == "rectangle":
        mask_arr = create_rectangle_floating(
            base_array, coverage_factor=coverage_factor
        )

    elif shape == "square":
        mask_arr = create_rectangle_floating(
            base_array, square=True, coverage_factor=coverage_factor
        )

    elif shape == "triangle":
        mask_arr = create_triangle(base_array, coverage_factor)

    elif shape == "circle":
        mask_arr = create_circle(base_array, coverage_factor)

    elif shape == "ellipse":
        if m <= n:
            primary = "y"
        else:
            primary = "x"
        mask_arr = create_ellipse(
            base_array, primary=primary, coverage_factor=coverage_factor
        )

    elif shape == "random_squares_small":
        # the large square size was chosen based on the size of a single solar cell's minimum dimension (150mm)
        # each cell in the original array was 25mm, thus 150/25 = 6 grid cells, so 6*6=36
        mask_arr = create_random_squares(
            base_array,
            square_area=small_square_area,
            seed=seed,
            coverage_factor=coverage_factor,
        )

    elif shape == "random_squares_large":
        # the large square size was chosen based on the size of a single solar module's minimum dimension (982.5mm)
        # each cell in the original array was 25mm, thus 982.5/25 = ~40 grid cells, so 40*40=1600
        mask_arr = create_random_squares(
            base_array,
            square_area=large_square_area,
            seed=seed,
            coverage_factor=coverage_factor,
        )

    elif shape == "organic":
        mask_arr = make_tree_mask(
            coverage_factor=coverage_factor, n_clusters=3, cluster_spread=4
        )

    else:
        print(
            "Arg 'shape' not specified. Defaulting to mask_columns.\n"
            "Choose from rows, columns, rectangle, triangle, circle, ellipse, random_squares_small, random_squares_large, organic"
        )

        mask_arr = create_mask_rows(base_array, coverage_factor)

    return mask_arr


def load_shading_masks():
    with open("/Users/jmccarty/Nextcloud/Projects/17_framework/notebooks/shading_masks_saved.pickle","rb") as fp:
        load_dict = pickle.load(fp)
    return load_dict