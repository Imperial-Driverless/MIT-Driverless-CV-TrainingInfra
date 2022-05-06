
def crop_image(img_with_boxes, x0, y0, x1, y1):
    cropped_img = img_with_boxes[y0:y1, x0:x1]
    return cropped_img