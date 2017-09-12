#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

##class trackerPoint(object):
##    def __init__(self,x,y,size,frame):
##        self.tracker = cv2.TrackerMIL_create()
##        self.bbox = (x-size/2,y-size/2,size,size)
##        ok = self.tracker.init(frame,self.bbox)
##        self.x = 0
##        self.y = 0
##        self.size = size
##    def update(self,frame):
##        ok, self.bbox = self.tracker.update(frame)
##        if ok:
##            self.x = int(self.bbox[0] + self.size/2)
##            self.y = int(self.bbox[1] + self.size/2)
##            p1 = (self.x-2,self.y-2)
##            p2 = (self.x+2,self.y+2)
##            cv2.rectangle(frame, p1, p2, (0,0,255),-1)
##
##def findMiddle(x0,x1,y0,y1):
##    x = int((x0+x1)/2)
##    y = int((y0+y1)/2)
##    return x,y
##
##def drawMiddle(x,y,frame):
##    p1 = (x-2,y-2)
##    p2 = (x+2,y+2)  
##    cv2.rectangle(frame, p1, p2, (255,0,0),-1)
##
##def Dist(p1,p2):
##    x1=p1[0]
##    y1=p1[1]
##    x2=p2[0]
##    y2=p2[1]
##    return math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2))
##
##def calcAngle(d):
##    a = d[0]
##    b = d[1]
##    c = d[2]
##    angRad = math.acos(((a*a)+(b*b)-(c*c))/(2*a*b))
##    return math.degrees(angRad)
##        
##
##cap = cv2.VideoCapture('Video.mp4')
##fourcc = cv2.VideoWriter_fourcc(*'XVID')
##out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280,720))
##
##kernel = np.ones((3,3),np.uint8)
##
##ret, frame = cap.read()
##thresh = cv2.inRange(frame,(150,150,150),(255,255,255))
##frame = cv2.erode(thresh,kernel,iterations = 1)
##        
##
##TrackersList = [(548,389),(574,454),(613,374),(699,396),(564,274),(605,252)]
##
##Trackers = []
##for t in TrackersList:
##    Trackers.append(trackerPoint(t[0],t[1],40,frame))
##
##while(cap.isOpened()):
##    ret, frame = cap.read()
##    if(frame is None):
##        break
##    thresh = cv2.inRange(frame,(150,150,150),(255,255,255))
##    frame = cv2.erode(thresh,kernel,iterations = 1)
##    for t in Trackers:
##        t.update(frame)
##    j = 0
##    m = []
##    for i in range(int(len(Trackers)/2)):
##        x0 = Trackers[j].x
##        x1 = Trackers[j+1].x
##        y0 = Trackers[j].y
##        y1 = Trackers[j+1].y
##        j += 2
##        m.append(findMiddle(x0,x1,y0,y1))
##
##    for i in range(len(m)-1):
##        p1 = m[i]
##        p2 = m[i+1]
##        cv2.line(frame,p1,p2,(255,0,255),2)
##
##    d = []
##    j = 0
##    for i in range(len(m)):
##        p1 = m[j]
##        if(j == len(m)-1):
##            p2 = m[0]
##        else:
##            p2 = m[j+1]
##        d.append(Dist(p1,p2))
##        j += 1
##
##    for i in m:
##        drawMiddle(i[0],i[1],frame)
##
##    ang = calcAngle(d)
##    strAng = "Angle = %2.2f degrees" % ang
##
##    cv2.putText(frame,strAng,(800,100),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255))
##    
##    cv2.imshow('frame',frame)
##    out.write(frame)
##    
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##
##cap.release()
##out.release()
##cv2.destroyAllWindows()


def main():
    kernel = np.ones((3,3),np.uint8)
    cap = cv2.VideoCapture('Video.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frame is None):
            break
        frame = frame[0:550,0:1280,:]
        #thresh1 = cv2.inRange(frame,(150,150,150),(255,255,255))
        #thresh1 = cv2.erode(thresh1,kernel,iterations = 5)
        #thresh1 = cv2.dilate(thresh1,kernel,iterations = 1)

        edges = cv2.Canny(frame,50,150,apertureSize = 3)
        #edges = cv2.dilate(edges,kernel,iterations = 3)

        lines = cv2.HoughLines(edges,1,np.pi/180,70)
        for i in range(len(lines)):
            for rho,theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
