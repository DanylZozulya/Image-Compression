from PIL import Image
import numpy as np
import scipy




def resample(img):
   for i in img:
       for j in range(0, len(i), 4):
           count = int((i[j][2] + i[j+1][2] + i[j+2][2] + i[j+3][2]) / 4)
           i[j][2] = count
           i[j+1][2] = count
           i[j+2][2] = count
           i[j+3][2] = count

def dct2(img):
    return scipy.fftpack.dct( scipy.fftpack.dct( img, axis=0, norm='ortho' ), axis=1, norm='ortho' )


#DCT-mattrix for transformation
DCT = [[.353553, .353553, .353553, .353553, .353553, .353553, .353553, .353553],
      [.490393, .415818, .277992, .097887, -.097106, -.277329, -.415375, -.490246],
      [.461978, .191618, -.190882, -.461673, -.462282, -.192353, .190145, .461366],
      [.414818, -.097106, -.490246, -.278653, .276667, .490710, .099448, -.414486],
      [.353694, -.353131, -.354256,  .352567, .354819, -.352001, -.355378, .351435],
      [.277992, -.490246, .096324, .416700, -.414486, -.100228, .491013, -.274673],
      [.191618, -.462282, .461366, -.189409, -.193822, .463187, -.460440, .187195],
      [.097887, -.278653, .416700, -.490862, .489771, -.413593, .274008, -.092414]]

def dct(block):
    TMP = block * DCT
    block = TMP * DCT


def quantization(block, q):
    Q = [[0.0 for y in range(8)] for x in range(8)]
    for i in range(0,8):
        for j in range(0,8):
            Q[i][j] = 1 + ((1 + i + j) * q)

    for i in range(0,8):
        for j in range(0,8):
            block[i][j] = int(block[i][j] / Q[i][j])
    return block

def zig_zag_traversal(matrix):
    rows = 8
    columns = 8
    solution = [[] for i in range(rows + columns - 1)]

    for i in range(rows):
        for j in range(columns):
            sum = i + j
            if (sum % 2 == 0):

                # add at beginning
                solution[sum].insert(0, matrix[i][j])
            else:

                # add at end of the list
                solution[sum].append(matrix[i][j])
    res = []
    for i in solution:
        for j in i:
            res.append(j)
    return res

def comprassion(block):
    c = 0
    flag = False
    for i in block:
        for j in i:
            if j == 0:
                c += 1
        if c >= 2:
            flag = True
    return flag

def encode_str(sequence, table):
    str = ''
    for i in sequence:
        str += table[i]
    return str

def main(image):
    img = Image.open(image)
    #convert to YCbCr
    ycbcr = img.convert('YCbCr')
    npmat = np.array(ycbcr, dtype=np.uint8)
    rows, cols = npmat.shape[0], npmat.shape[1]
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
                # [0, 255] --> [-128, 127]
                block = npmat[i:i + 8, j:j + 8, k] - 128
                dct(block)
                quant_matrix = quantization(block, 20)
                zz = zig_zag_traversal(quant_matrix)

                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]
                #Після етапу квантування за алгоритмом jpeg іде кодумання алгоритмом Хаффмана
                #Агоритм Хаффмана я реалізува у huffman.py
                #Проблема полягає в тому,як цей закодований рядок записати у файл,щоб збереглось зображення.












if __name__ == "__main__":
    main('C:\\Users\\zozul\\Desktop\\bigdata\\unnamed.png')