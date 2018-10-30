"""
Reference : https://waset.org/publications/11683/extracting-human-body-based-on-background-estimation-in-modified-hls-color-space
"""



import cv2
import numpy as np
from scipy import stats

bg_img_path = r"images/Page-2-Image-1.png"
fg_img_path = r"images/Page-2-Image-2.png"

def removeBackground(img):

    #
    pass

def grabCut(img):

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (150, 152, 418, 453)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    return img

def main():
    img_bg = cv2.imread(bg_img_path)
    img_fg = cv2.imread(fg_img_path)

    hsv_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2HSV)
    hsv_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGR2HSV)

    hls_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2HLS)
    hls_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGR2HLS)

    sigma_l_bg = np.var(hls_bg[:,:,1].flatten())
    sigma_s_bg = np.var(hls_bg[:,:,2].flatten())

    mode_h = stats.mode(hsv_bg[:,:,0].flatten())

    grabCutImg = grabCut(img_fg)
    #for i in range
    print (mode_h)
    mode_h = mode_h[0][0]

    mode_img = img_bg* 0

    #mode_img = (mode_h == hsv_bg[:,:,0]) * 255
    #mode_img = mode_img.astype(np.uint8)

    # gray
    gray_img = cv2.cvtColor(img_fg, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_img, 30, 120)
    canny_dilated = cv2.dilate(canny, (11, 11))

    hue_range = list(range(mode_h-10, mode_h+10))
    mask = gray_img * 0
    for i,row in enumerate(mode_img):
        for j,col in enumerate(row):

            intensity_diff =  0
            for k in range(2):
                intensity_diff += abs(int(hsv_fg[i][j][k]) - int(hsv_bg[i][j][k]))
            intensity_diff /= 3
            #print (intensity_diff)
            if hsv_fg[i][j][0] in hue_range or intensity_diff < 10:
                mode_img[i][j] = (0,0,0)
                mask[i][j] = 0
            else:
                mode_img[i][j] = img_fg[i][j]
                mask[i][j] = 255


    canny_filtered = cv2.bitwise_and(canny_dilated,mask)


    rect = (150, 152, 418, 453)
    cv2.rectangle(img_fg,rect[0:2],rect[2:],(255,0,0),3)

    cv2.imshow("mode_img", mode_img)
    #cv2.imshow("img_bg",img_bg)
    cv2.imshow("img_fg",img_bg)
    cv2.imshow("canny_dilated",canny_dilated)
    cv2.imshow("canny_filtered",canny_filtered)

    cv2.waitKey()
    cv2.destroyAllWindows()


main()
