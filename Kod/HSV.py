import cv2
import numpy as np

cap=cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("HSV Trackbar")
cv2.createTrackbar("L-H","HSV Trackbar",0,179,nothing)
cv2.createTrackbar("L-S","HSV Trackbar",0,255,nothing)
cv2.createTrackbar("L-V","HSV Trackbar",0,255,nothing)
cv2.createTrackbar("U-H","HSV Trackbar",179,179,nothing)
cv2.createTrackbar("U-S","HSV Trackbar",255,255,nothing)
cv2.createTrackbar("U-V","HSV Trackbar",255,255,nothing)

while True:
    _,frame =cap.read()
    frame = cv2.resize(frame, (712, 400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    l_h=cv2.getTrackbarPos("L-H","HSV Trackbar")
    l_s=cv2.getTrackbarPos("L-S","HSV Trackbar")
    l_v=cv2.getTrackbarPos("L-V","HSV Trackbar")
    h_h=cv2.getTrackbarPos("U-H","HSV Trackbar")
    h_s=cv2.getTrackbarPos("U-S","HSV Trackbar")
    h_v=cv2.getTrackbarPos("U-V","HSV Trackbar")
   
    low=np.array([l_h,l_s,l_v])
    high=np.array([h_h,h_s,h_v])

    mask=cv2.inRange(hsv,low,high) 
    result=cv2.bitwise_and(frame,frame,mask=mask)    
    cv2.imshow("result",result)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        thearray = [[l_h,l_s,l_v],[h_h, h_s, h_v]]
        np.save('marker_values',thearray)
        break
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



