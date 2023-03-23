import cv2
import numpy as np
import time

class drawingCanvas():
    def __init__(self):
        self.marker_values = np.load('marker_values.npy')
        self.cap = cv2.VideoCapture(0)
        self.canvas = None
        self.x1,self.y1=0,0
        self.val=1
        self.draw()

    def draw(self):
        while True:
            _, self.frame = self.cap.read()
            self.frame = cv2.flip( self.frame, 1 )
            self.frame = cv2.resize(self.frame, (712, 400))
            if self.canvas is None:
                self.canvas = np.zeros_like(self.frame)
            
            mask=self.CreateMask()
            contours=self.Contours(mask)
            self.drawLine(contours)
            self.shapes = self.ShapeDetection(self.canvas, self.frame)
            self.display()

            k = cv2.waitKey(1) & 0xFF
            self.takeAction(k)
            if k == 27:
                break     

    def CreateMask(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV) 
        lower_range = self.marker_values[0]
        upper_range = self.marker_values[1]
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask
      
    def Contours(self,mask):
        contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def drawLine(self,contours):
        if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > 100:  
            c = max(contours, key = cv2.contourArea)
            x2,y2,w,h = cv2.boundingRect(c)

            # Find center point of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(self.frame, (cX, cY), 10, (0, 0, 255), 2)

            if self.x1 == 0 and self.y1 == 0:
                self.x1,self.y1= x2,y2
            else:
                self.canvas = cv2.line(self.canvas, (self.x1,self.y1),(x2,y2), [255*self.val,255,255], 10)
            self.x1,self.y1= x2,y2
        else:
            self.x1,self.y1 =0,0        
    

    def ShapeDetection(self, image, frame):
        thresh_image = cv2.inRange(image, np.array([250, 250, 250]), np.array([255, 255, 255]))
        thresh_image = cv2.GaussianBlur(thresh_image, (5, 5), 0)
        contours, _ = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = np.zeros_like(image)

        for i, contour in enumerate(contours):
            epsilon = 0.06*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h= cv2.boundingRect(approx)
            x_mid = int(x + (w/3)) 
            y_mid = int(y + (h/1.5)) 
            coords = (x_mid, y_mid)
            colour = (255, 0, 0)
            font = cv2.FONT_HERSHEY_DUPLEX

            if len(approx) == 3:
                cv2.putText(result_image, "Triangle", coords, font, 1, colour, 1) 
            elif len(approx) == 4:
                cv2.putText(result_image, "Rectangle", coords, font, 1, colour, 1)
        
        return result_image
    
    def display(self):
        self.frame = cv2.add(self.frame,self.canvas)    
        cv2.imshow('frame',self.frame)
        cv2.imshow('canvas',self.canvas)
        cv2.imshow('shapes',self.shapes)
        
    def takeAction(self,k):
        if k == ord('c'):
            self.canvas = None
        if k==ord('e'):
            self.val= int(not self.val)
                   
if __name__ == '__main__':
    drawingCanvas()
    
cv2.waitKey(0)
cv2.destroyAllWindows()