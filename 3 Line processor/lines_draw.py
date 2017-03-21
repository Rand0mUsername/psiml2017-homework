import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def cv_plot(T):
    TRES = [cv.resize(t, (400, 700)) for t in T]
    img = np.concatenate(TRES, axis=1)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def count_lines(img):
    """Return the number of lines in the document."""
    # convert to black and white
    cv_plot([img])
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_bw = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV, 25, 29)
    # do a strong opening to remove noise and find bounding box
    img_bw_denoise = cv.morphologyEx(img_bw.copy(), cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white_pixels = np.where(img_bw_denoise == 255)
    coords = np.column_stack((white_pixels[1], white_pixels[0]))
    rect = cv.minAreaRect(coords)
    box = np.int0(np.around(cv.boxPoints(rect)))
    cv.drawContours(img_bw_denoise,[box],0,255,5)
    cv_plot([img_gray, img_bw, img_bw_denoise])
    # create a rotation matrix to fix skew
    ang = -(90 + rect[2]) if rect[2] < -45 else -rect[2]
    (height, width) = img_bw_denoise.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv.getRotationMatrix2D(center, -ang, 1.0)
    # do a weaker opening and rotate the image
    img_bw_curr = cv.morphologyEx(img_bw, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    box_rotated = cv.transform(np.array([box]), matrix)
    img_bw_rot = cv.warpAffine(img_bw_curr, matrix, (width, height),
                               flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    img_bw_rot2 = img_bw_rot.copy()
    cv.drawContours(img_bw_rot2,[box_rotated],0,255,5)
    cv_plot([img_bw_curr, img_bw_rot2])
    # crop the image, remove parts outside the bounding box
    top_left = np.min(np.min(box_rotated, axis=1), axis=0)
    bot_right = np.max(np.max(box_rotated, axis=1), axis=0)
    img_cropped = img_bw_rot[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
    # do a vertical erosion to separate lines
    img_final = cv.erode(img_cropped, np.ones((15, 1), np.uint8), iterations=1)
    cv_plot([img_cropped, img_final])
    # extract the average of lightest 4 pixels from every row
    # this way short rows don't have a disadvantage but we are less
    # resistant to noise, so we have to do good noise removal
    vals = []
    for row in img_final:
        ind = np.argpartition(row, -4)[-4:]
        avg_good = np.average(row[ind])
        vals.append(int(np.around(avg_good)))
    plt.plot(vals)
    plt.show()
    # do some trivial smoothing
    vals_smooth = []
    for i, _ in enumerate(vals):
        left = max(0, i-3)
        right = min(len(vals), i+3+1)
        vals_smooth.append(np.average(vals[left:right]))
    # count rising edges and return the result
    plt.plot(vals_smooth)
    cnt = 0
    for i, val_i in enumerate(vals_smooth):
        if val_i > 0 and (i == 0 or vals_smooth[i-1] == 0):
            cnt += 1
            plt.scatter(i, val_i, marker='o')
    plt.xlabel('solution = ' + str(cnt))
    plt.show()
    return cnt

if __name__ == "__main__":
    in_file = raw_input()
    img = cv.imread(in_file)
    print(count_lines(img))
