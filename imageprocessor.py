import math
from PIL import Image

#########################
### Loading Functions ###
#########################


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.
    """
    with open(filename, 'rb') as image_file:
        img = Image.open(image_file)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

#######################
### Pixel Functions ###
#######################


def get_pixel(image, x, y):
    """
    This returns the pixel value for a greyscale image at position (x, y)
    """
    if 0 <= x < image['width']:
        x_prime = x
    elif x < 0:
        x_prime = 0
    else:
        x_prime = image['width'] - 1
    if 0 <= y < image['height']:
        y_prime = y
    elif y < 0:
        y_prime = 0
    else:
        y_prime = image['height'] - 1
    # the position in the pixel list isn't a tuple
    pos = image['width']*y_prime + x_prime
    return image['pixels'][pos]


def set_pixel(image, x, y, c):
    """
    This sets the pixel value for a greyscale image at position (x, y) to c
    """
    pos = image['width']*y + x  # this is the same issue as above
    image['pixels'][pos] = c


def apply_per_pixel(image, func):
    """
    This sets all pixels for a greyscale image to func(c) where c was the original pixel value
    """
    result = {
        'height': image['height'],
        'width': image['width'],  # width was spelt wrong
        # should be a list of the same len()
        'pixels': [0]*(image['height']*image['width']),
    }
    for y in range(image['height']):
        for x in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result

########################
### Helper Functions ###
########################


def multiply(image, kernel):
    """
    This computes the result of multiplying the given image with the given kernel.

    The output of this function is an image (a dictionary with 'height', 'width', 
    and 'pixels' keys), but its pixel values are not necessarily in the range [0,255].

    The kernel input is a dictionary with two keys: 'size' and 'values'. 'size' 
    maps onto a value (n) which represents an n-by-n kernel. 'values' 
    maps onto a list which contains the values in the kernel using 
    row-major order. e.g. the following kernel:
        1    0    1
        2    4    3
        5    -7   0
    is represented by:
        {'size': 3, 'values': [1, 0, 1, 2, 4, 3, 5, -7, 0]}
    """
    # We create the image dictionary to update with the image to return
    # This is done to not edit/modify the original image dictionary inputted into the function
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0]*(image['height']*image['width']),
    }
    for y in range(image['height']):
        for x in range(image['width']):
            sub_image = get_sub_image(image, x, y, kernel['size'])
            tot_prod = 0  # We get total products of the values by running through the pixels
            # We need the positional index and not just coordinates
            pos = image['width']*y + x
            for i in range(len(kernel['values'])):
                tot_prod = tot_prod + \
                    sub_image['pixels'][i]*kernel['values'][i]
            result['pixels'][pos] = tot_prod
    return result


def get_sub_image(image, x, y, size):
    '''
    This function takes in an image and returns a size-by-size square of pixels 
    around the position (x, y). It returns a 6.009 image format with 'height'
    and 'width' equal to size. size is an odd int.
    '''
    pixels = []
    # The range is the distance from the center square in the subimage
    rang = (size - 1)//2
    for j in range(y - rang, y + rang + 1):
        # We run through each row by the values of y
        for i in range(x - rang, x + rang + 1):
            # We run through each value through the row by the values of x
            # We add this pixel to our subimage
            pixels.append(get_pixel(image, i, j))
    result = {'height': size, 'width': size, 'pixels': pixels}
    return result


def round_and_clip_image(image):
    """
    Given a dictionary, this ensures that the values in the 'pixels' list are all
    integers in the range [0, 255].
    """
    for i in range(len(image['pixels'])):
        pixel = image['pixels'][i]
        if pixel <= 0:
            image['pixels'][i] = 0  # We change the numbers below 0
        elif pixel >= 255:
            image['pixels'][i] = 255  # We change the numbers above 255
        else:
            image['pixels'][i] = round(pixel)  # For the rest, just round


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.
    """
    new_pixels = []  # We create a pixel list to hold the pixel values of the new greyscale image
    for i in range(len(image['pixels'])):
        val = round(0.299*image['pixels'][i][0] + 0.587 *
                    image['pixels'][i][1] + 0.114*image['pixels'][i][2])
        new_pixels.append(val)  # We add the scaled greyscale pixel value
    # We create the image dictionary using the new pixels
    new_image = {'height': image['height'],
                 'width': image['width'], 'pixels': new_pixels}
    return new_image


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges.
    """
    return edges(grey)  # Call edges for the energy map


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 2 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    e_p = []  # We create a list to add the values to in the cummulative energy map
    h = energy['height']
    w = energy['width']  # For ease of calling
    p = energy['pixels']
    for i in range(w):  # We add the first row alone
        e_p.append(p[i])
    for pos in range(w, w*h):
        if pos % w == 0:  # At the left, we only care about the top and top right
            adjacent_pixels = [e_p[pos - w], e_p[pos - w + 1]]
            e_p.append(p[pos] + min(adjacent_pixels))
        elif pos % w == w - 1:  # At the right, we only care about the top and top left
            adjacent_pixels = [e_p[pos - w - 1], e_p[pos - w]]
            e_p.append(p[pos] + min(adjacent_pixels))
        else:
            adjacent_pixels = [e_p[pos - w - 1],
                               e_p[pos - w], e_p[pos - w + 1]]
            e_p.append(p[pos] + min(adjacent_pixels))
    # We create the image dictionary using the new pixels
    energy_map = {'height': h, 'width': w, 'pixels': e_p}
    return energy_map


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    index_list = []  # We create the list to be returned
    h = cem['height']
    w = cem['width']  # For ease of calling
    p = cem['pixels']
    bottom_row = p[w*(h - 1):]  # We get the bottom row values alone
    for ind in range(len(bottom_row)):
        # We run through and find the index of the minimum
        if bottom_row[ind] == min(bottom_row):
            # We add this first minimum index to our list
            index_list.append(w*(h - 1) + ind)
            break  # Since we take left preference, we stop the chain in case there is another minimum
    for i in range(h - 1):
        # We assign this variable to the previous position we were at
        pos = index_list[-1]
        if pos % w == 0:  # At the left
            # We only take the one above and then top right
            adj_pixels = [p[pos - w], p[pos - w + 1]]
            if p[pos - w] == min(adj_pixels):
                idx = pos - w
            elif p[pos - w + 1] == min(adj_pixels):
                idx = pos - w + 1
        elif pos % w == w - 1:  # At the right
            # We only take the one above and then top left
            adj_pixels = [p[pos - w - 1], p[pos - w]]
            if p[pos - w - 1] == min(adj_pixels):
                idx = pos - w - 1
            elif p[pos - w] == min(adj_pixels):
                idx = pos - w
        else:
            # Otherwise, we take all three
            adj_pixels = [p[pos - w - 1], p[pos - w], p[pos - w + 1]]
            if p[pos - w - 1] == min(adj_pixels):
                idx = pos - w - 1
            elif p[pos - w] == min(adj_pixels):
                idx = pos - w
            elif p[pos - w + 1] == min(adj_pixels):
                idx = pos - w + 1
        index_list.append(idx)  # We add the new index we got
    return index_list


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    new_pixels = image['pixels'][:]  # We do not want to alter the original image so we take a copy of the pixels
    # We do not want to alter the original list so we create a copy
    s_sorted = seam[:]
    s_sorted.sort()
    s_sorted.reverse()  # We put the indices in reverse order and use the pop function to remove them as we go through the list
    for ind in s_sorted:
        new_pixels.pop(ind)
    new_image = {'height': image['height'],
                 'width': image['width'] - 1, 'pixels': new_pixels}
    return new_image

###############
### Filters ###
###############


def inverted(image):
    """
    Given a greyscale image, we take 255-(pixel value) to invert it
    """
    return apply_per_pixel(image, lambda c: 255-c)


def blurred(image, n):
    """
    Returns a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.
    """
    kernel = {'size': n, 'values': [1/(n**2)]*(n**2)}

    blurred_image = multiply(image, kernel)

    round_and_clip_image(blurred_image)
    return blurred_image


def sharpened(image, n):
    """
    This takes in the image (image) and the size (n) of the kernel and creates the
    special kernel required to directly sharpen the image. It's just 2 subtracting the 
    middle value and zero subtracting the rest. This is then used for multiplying
    and the final image is rounded and clipped.
    """
    # Since n would be odd, we take an integer divide of the first n/2 values
    kernel = {'size': n, 'values': [-1/(n**2)]*((n**2)//2) +
              [2 - 1/(n**2)] + [-1/(n**2)]*((n**2)//2)}
    sharp_image = multiply(image, kernel)
    round_and_clip_image(sharp_image)
    return sharp_image


def edges(image):
    """
    We take an image (image), multiply it with two kernels (kernel_x and kernel_y),
    and then take the root of the sum of the squares of each output image from 
    the kernels and use that as the final pixel values in the final image. We 
    then round and clip the final values before returning it.
    """
    kernel_x = {'size': 3, 'values': [-1, 0, 1, -2, 0, 2, -1, 0, 1]}
    kernel_y = {'size': 3, 'values': [-1, -2, -1, 0, 0, 0, 1, 2, 1]}
    o_x = multiply(image, kernel_x)
    o_y = multiply(image, kernel_y)
    pixels = []
    for pos in range(len(o_x['pixels'])):
        pixels.append(((o_x['pixels'][pos])**2 + (o_y['pixels'][pos])**2)**0.5)
    image_edges = {'height': image['height'],
                   'width': image['width'], 'pixels': pixels}
    round_and_clip_image(image_edges)
    return image_edges

##############################
### Color Filter Functions ###
##############################


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_function(color_image):  # We must define a function inside to return
        r_pixels = []  # We create a separate pixel list for r
        g_pixels = []  # We create a separate pixel list for g
        b_pixels = []  # We create a separate pixel list for b
        # We add the corresponding color to that list
        for tup in color_image['pixels']:
            r_pixels.append(tup[0])
            g_pixels.append(tup[1])
            b_pixels.append(tup[2])
        # We then create greyscale images for each color pixel
        r_image = {'height': color_image['height'],
                   'width': color_image['width'], 'pixels': r_pixels}
        g_image = {'height': color_image['height'],
                   'width': color_image['width'], 'pixels': g_pixels}
        b_image = {'height': color_image['height'],
                   'width': color_image['width'], 'pixels': b_pixels}
        # We apply the filter to each image
        new_r = filt(r_image)
        new_g = filt(g_image)
        new_b = filt(b_image)
        # We create another pixel list for the image to return after the filter is applied
        new_pixels = []
        for i in range(len(new_r['pixels'])):
            tup = (new_r['pixels'][i], new_g['pixels'][i], new_b['pixels'][i])
            # We add the corresponding tuples for all the colors to one pixel list to return
            new_pixels.append(tup)
        # We create the filtered color dictionary
        new_color_image = {
            'height': color_image['height'], 'width': color_image['width'], 'pixels': new_pixels}
        return new_color_image  # It is returned from the function
    return color_function  # We return the created function


def make_blur_filter(n):
    def new_blur(image):  # We start defining a function to return
        # We use the same blurred function as above, now using the n inputted from the overall function
        return blurred(image, n)
    return new_blur  # Now, we return the created function


def make_sharpen_filter(n):
    def new_sharpen(image):  # We start defining a function to return
        # We use the same sharpened function as above, now using the n inputted from the overall function
        return sharpened(image, n)
    return new_sharpen  # Now, we return the created function


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def filters_applied(image):  # We start defining a function to return
        old_image = image
        for filt in filters:  # Now we run through filters and apply them in order
            new_image = filt(old_image)
            old_image = new_image
        return new_image  # For the created function, we return the image created
    return filters_applied  # Now we return the function created


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    image_trial = image
    # For each trial, we run the same thing... we run through the functions as described
    for i in range(ncols):
        grey_im = greyscale_image_from_color_image(image_trial)
        energy = compute_energy(grey_im)
        cum_energy_map = cumulative_energy_map(energy)
        min_energy_seam = minimum_energy_seam(cum_energy_map)
        # We redo the process on the new image formed
        image_trial = image_without_seam(image_trial, min_energy_seam)
    return image_trial  # We return the last one formed


def darken_color_image(image):
    """
    This function takes in a color image and, with each implementation, darkens the image
    """
    new_pixels = []
    old_pixels = image['pixels']
    for pix in old_pixels:
        r_pixel = pix[0]
        g_pixel = pix[1]
        b_pixel = pix[2]
        r_new = (g_pixel + b_pixel)/2
        g_new = (b_pixel + r_pixel)/2  # We average out the values
        b_new = (r_pixel + g_pixel)/2
        # We round the float values
        new_pixel = (round(r_new), round(g_new), round(b_new))
        new_pixels.append(new_pixel)
    new_image = {'height': image['height'],
                 'width': image['width'], 'pixels': new_pixels}
    return new_image


def lighten_color_image(image):
    """
    This function takes in a color image and, with each implementation, lightens the image
    """
    new_pixels = []
    old_pixels = image['pixels']
    for pix in old_pixels:
        r_pixel = pix[0]
        g_pixel = pix[1]
        b_pixel = pix[2]
        r_new = g_pixel + b_pixel - r_pixel
        # This simply does the exact reverse of the previous function
        g_new = b_pixel + r_pixel - g_pixel
        b_new = r_pixel + g_pixel - b_pixel
        # We round the float values
        new_pixel = (round(r_new), round(g_new), round(b_new))
        new_pixels.append(new_pixel)
    new_image = {'height': image['height'],
                 'width': image['width'], 'pixels': new_pixels}
    return new_image


def make_one_color(image, color):
    """
    This takes in an image and either 'r' for red, 'g' for green, or 'b' for blue
    and converts the color image into the color inputted
    """
    new_pixels = []
    old_pixels = image['pixels']
    for pix in old_pixels:
        r_pixel = pix[0]
        g_pixel = pix[1]
        b_pixel = pix[2]
        val = (r_pixel + g_pixel + b_pixel)/3
        if color == 'r':
            new_pixel = (round(val), 0, 0)
            new_pixels.append(new_pixel)
        elif color == 'g':
            new_pixel = (0, round(val), 0)
            new_pixels.append(new_pixel)
        elif color == 'b':
            new_pixel = (0, 0, round(val))
            new_pixels.append(new_pixel)
    new_image = {'height': image['height'],
                 'width': image['width'], 'pixels': new_pixels}
    return new_image


if __name__ == '__main__':
    bluegill = load_greyscale_image('test_images/bluegill.png')
    save_greyscale_image(inverted(bluegill), 'image_results/bluegill_inverted.png')

    pigbird = load_greyscale_image('test_images/pigbird.png')
    pigbird_kernel = {'size': 9, 'values': [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            1, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, ]
                      }
    pigbird_test = multiply(pigbird, pigbird_kernel)
    round_and_clip_image(pigbird_test)
    save_greyscale_image(pigbird_test, 'image_results/pigbird_multiply.png')

    cat = load_greyscale_image('test_images/cat.png')
    save_greyscale_image(blurred(cat, 5), 'image_results/blurred_cat.png')

    python = load_greyscale_image('test_images/python.png')
    save_greyscale_image(sharpened(python, 11),
                         'image_results/sharpened_python.png')

    construct = load_greyscale_image('test_images/construct.png')
    save_greyscale_image(edges(construct), 'image_results/edges_construct.png')
    
    chess = load_greyscale_image('test_images/chess.png')
    save_greyscale_image(edges(chess), 'image_results/edges_chess.png')

    cat = load_color_image('test_images/cat.png')
    inverted_color = color_filter_from_greyscale_filter(inverted)
    cat_inverted = inverted_color(cat)
    save_color_image(cat_inverted, 'image_results/inverted_cat.png')

    python = load_color_image('test_images/python.png')
    blurry9 = make_blur_filter(9)
    blurry9_color = color_filter_from_greyscale_filter(blurry9)
    blurry_python = blurry9_color(python)
    save_color_image(blurry_python, 'image_results/blurry_python.png')

    chick = load_color_image('test_images/sparrowchick.png')
    sharp7 = make_sharpen_filter(7)
    sharp7_color = color_filter_from_greyscale_filter(sharp7)
    sharp_chick = sharp7_color(chick)
    save_color_image(sharp_chick, 'image_results/sharpened_sparrowchick.png')

    frog = load_color_image('test_images/frog.png')
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    filt = filter_cascade([filter1, filter1, filter2, filter1])
    filtered_frog = filt(frog)
    save_color_image(filtered_frog, 'image_results/filtered_frog.png')

    twocats = load_color_image('test_images/twocats.png')
    carved_twocats = seam_carving(twocats, 100)
    save_color_image(carved_twocats, 'image_results/carved_twocats.png')

    bluegill = load_color_image('test_images/bluegill.png')
    bluegill_darkened1 = darken_color_image(bluegill)
    bluegill_darkened2 = darken_color_image(bluegill_darkened1)
    bluegill_darkened3 = darken_color_image(bluegill_darkened2)
    save_color_image(bluegill_darkened1,
                     'image_results/bluegill_darkended1.png')
    save_color_image(bluegill_darkened2,
                     'image_results/bluegill_darkended2.png')
    save_color_image(bluegill_darkened3,
                     'image_results/bluegill_darkended3.png')

    bluegill_lightened1 = lighten_color_image(bluegill)
    bluegill_lightened2 = lighten_color_image(bluegill_lightened1)
    bluegill_lightened3 = lighten_color_image(bluegill_lightened2)
    save_color_image(bluegill_lightened1,
                     'image_results/bluegill_lightended1.png')
    save_color_image(bluegill_lightened2,
                     'image_results/bluegill_lightended2.png')
    save_color_image(bluegill_lightened3,
                     'image_results/bluegill_lightended3.png')

    red_bluegill = make_one_color(bluegill, 'r')
    green_bluegill = make_one_color(bluegill, 'g')
    blue_bluegill = make_one_color(bluegill, 'b')
    save_color_image(red_bluegill, 'image_results/bluegill_turned_red.png')
    save_color_image(green_bluegill, 'image_results/bluegill_turned_green.png')
    save_color_image(blue_bluegill, 'image_results/bluegill_turned_blue.png')

    pass
