import cv2 as cv

if __name__ == "__main__":
    in_file = raw_input()
    img = cv.imread(in_file)
    pixel = img[0, 0]
    print("Red: {0}\n".format(pixel[2]))
    print("Green: {0}\n".format(pixel[1]))
    print("Blue: {0}".format(pixel[0]))