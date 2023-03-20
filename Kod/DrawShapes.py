import cv2
import numpy as np
import time

class drawingCanvas():
    def __init__(self):
        self.penrange = np.load('penrange.npy')
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
            contours=self.ContourDetect(mask)
            self.drawLine(contours)
            self.detect_Shape(self.canvas, self.frame)
            self.display()
            k = cv2.waitKey(1) & 0xFF
            self.takeAction(k)
            if k == 27:
                break     
               

    def CreateMask(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV) 
        lower_range = self.penrange[0]
        upper_range = self.penrange[1]
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask
      
    def ContourDetect(self,mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    

    def detect_Shape(self, image, frame):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        _, thresh_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        contours, h = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            if i == 0:
                continue
            epsilon = 0.03*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h= cv2.boundingRect(approx)
            x_mid = int(x + (w/3)) 
            y_mid = int(y + (h/1.5)) 
            coords = (x_mid, y_mid)
            colour = (255, 0, 0)
            font = cv2.FONT_HERSHEY_DUPLEX

            #Examples of shapes, need to match drawing position

            if len(approx) == 3:
                cv2.putText(image, "Triangle", coords, font, 1, colour, 1) 
                p1 = (100, 200)
                p2 = (50, 50)
                p3 = (300, 100)
                cv2.line(image, p1, p2, (0, 255, 0), 5)
                cv2.line(image, p2, p3, (0, 255, 0), 5)
                cv2.line(image, p1, p3, (0, 255, 0), 5)
                break
            elif len(approx) == 4:
                cv2.putText(image, "Rectangle", coords, font, 1, colour, 1)
                cv2.rectangle(image, (30, 30), (300, 200), (0, 255, 0), 5)  
                approx = np.zeros_like(frame)
                break
            else:
                cv2.putText(image, "Circle", coords, font, 1, colour, 1)
                cv2.circle(image, (coords), 80, (0, 255, 0), 5)
                approx = np.zeros_like(frame)
                break
                
    def display(self):
        self.frame = cv2.add(self.frame,self.canvas)    
        cv2.imshow('frame',self.frame)
        cv2.imshow('canvas',self.canvas)
        
    def takeAction(self,k):
        if k == ord('c'):
            self.canvas = None
        if k==ord('e'):
            self.val= int(not self.val)
                   
if __name__ == '__main__':
    drawingCanvas()
    
cv2.waitKey(0)
cv2.destroyAllWindows()