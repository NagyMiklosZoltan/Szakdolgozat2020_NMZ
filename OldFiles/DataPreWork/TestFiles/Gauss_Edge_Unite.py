import cv2
from matplotlib import pyplot as plt

# Gauss Blur kernel size
dim = 5
k_size = (dim, dim)

# Edge Detecition Thresholds
thre_1 = 100
thre_2 = 200


# Read in GrayScale
img = cv2.imread('image_017.jpg', 0)

gaus = cv2.GaussianBlur(img, k_size, cv2.BORDER_DEFAULT)

edges = cv2.Canny(gaus, thre_1, thre_2)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


plt.show()