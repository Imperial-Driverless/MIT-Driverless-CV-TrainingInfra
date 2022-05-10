from PIL import Image
import numpy

"""def crop_image(target_path, x0, y0, x1, y1):
    img = Image.open(target_path)
    # area = (x0, y0, x1, y1)
    # return img.crop(area)
    pix = numpy.array(img)
    crop = pix[x0:x1, y0:y1]
    PIL_image = Image.fromarray(crop.astype('uint8'), 'RGB')
    return PIL_image
"""
