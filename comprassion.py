import os
import numpy as np
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree


def quantize(block, component):
    """The method implements image quantization.
           Each of the coefficients in each of the 8x8 matrices is divided by a certain number.
           If the image quality after all its modifications you will not reduce more,
           then the divider should be one.If the memory occupied by this photo is more important to you,
           then the divisor will be greater than 1, and the quotient will be rounded.
           :param block: 8x8 pixel matrix
           :param component: type of color components (luma or chrominance)
           :return:8x8 pixel matrix with changed coefficients
            """
    q = quantization_table(component)
    return (block / q).round().astype(np.int32)


def block_to_zigzag(block):
    """The method implements a zig-zag matrix pass
            :param matrix: 8x8 matrix
            :return: array of coefficients
            """
    return np.array([block[point] for point in zigzag_traversal(*block.shape)])


def dct_2d(block):
    """The method implements the Discrete-cosine transform for an 8x8 pixel block.
           DCT turns the block into a spectrum, and where the readings change dramatically,
           the coefficient becomes positive. Where the coefficient is higher,
           the picture shows clear transitions in color and brightness,
           where it is lower - weak (smooth) changes in the values of the YCbCr components in the block.
            :param block: 8x8 pixel matrix
            :return:8x8 pixel matrix with changed coefficients
            """
    return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')


def run_length_encode(arr):
    """The method implements run length encoding compression for AC coefficients.
    :param arr: array of AC coefficient
    :return: Number of zeros before AC, AC length (in bits)
    """
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


def write_to_file(filepath, dc, ac, blocks_count, tables):
    """The method implements writing encoded data to a file.
           - 16 bits for 'table_size'
            for dc:
               - 4 bits for the 'category'
               - 4 bits for 'code_length'
               - 'code_length' bits for 'huffman_code'
            for ac:
               - 4 bits for 'run_length'
               - 4 bits for 'size'
               - 8 bits for 'code_length'
               - 'code_length' bits for 'huffman_code'
            :param - filepath: path to write file
                   - blocks_count: blocks count
                   - tables: for huffman tables 'dc_y', 'dc_c', 'ac_y', 'ac_c'
            :return: None
                    """
    try:
        f = open(filepath, 'w')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e

    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bits for 'table_size'
        f.write(uint_to_binstr(len(tables[table_name]), 16))

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # 4 bits for the 'category'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                # 4 bits for 'run_length'
                # 4 bits for 'size'
                # 8 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)

    # 32 bits for 'blocks_count'
    f.write(uint_to_binstr(blocks_count, 32))

    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])

            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))

            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()


def main(input_file, output_file):

    image = Image.open(input_file)


    ycbcr = image.convert('YCbCr')

    npmat = np.array(ycbcr, dtype=np.uint8)

    rows, cols = npmat.shape[0], npmat.shape[1]

    # block size: 8x8
    if rows % 8 == cols % 8 == 0:
        blocks_count = rows // 8 * cols // 8
    else:
        raise ValueError(("the width and height of the image "
                          "should both be mutiples of 8"))

    # dc is the top-left cell of the block, ac are all the other cells
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            for k in range(3):
                # split 8x8 block and center the data range on zero
                block = npmat[i:i+8, j:j+8, k] - 128
                # discrete cosine block transform
                dct_matrix = dct_2d(block)
                # block quantization
                quant_matrix = quantize(dct_matrix,
                                        'lum' if k == 0 else 'chrom')
                # get an array of coefficients
                zz = zigzag_traversal(block_to_zigzag(quant_matrix))


                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]
    # сreate huffman trees separately for 'dc_y', 'ac_y', 'dc_c', 'ac_c'
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
            flatten(run_length_encode(ac[i, :, 0])[0]
                    for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
            flatten(run_length_encode(ac[i, :, j])[0]
                    for i in range(blocks_count) for j in [1, 2]))
    # final tables for blocks
    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}

    write_to_file(output_file, dc, ac, blocks_count, tables)

def quantization_table(component):

    if component == 'lum':

        q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [16, 11, 10, 16, 24, 40, 51, 61]])
    elif component == 'chrom':
        q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [17, 18, 24, 47, 99, 99, 99, 99]])
    else:
        raise ValueError((
            "component should be either 'lum' or 'chrom', "
            "but '{comp}' was found").format(comp=component))

    return q






def bits_required(n):
    """Helper method for counting the required number of bits."""
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result

def zigzag_traversal(rows, cols):
    """The method implements a zig-zag matrix pass
            :param matrix: 8x8 matrix
            :return: array of coefficients
            """
    # constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # move the point in different directions
    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # True when moving up-right, False when moving down-left
    move_up = True

    for i in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)

def binstr_flip(binstr):
    """Helper method for check if binstr is a binary string."""
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)


def flatten(lst):
    return [item for sublist in lst for item in sublist]



if __name__ == "__main__":
    image_path = ""
    output_file = ""
    main(image_path, output_file)
