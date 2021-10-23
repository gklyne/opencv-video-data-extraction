import cv2

print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread("clouds.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
cv2.imshow("Over the Clouds", img)
cv2.imshow("Over the Clouds - gray", gray)

# NOTE: click in displayed image window to quit
cv2.waitKey(0)
cv2.destroyAllWindows()


