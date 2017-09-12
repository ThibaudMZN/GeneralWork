#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

class trackerPoint(object):
   def __init__(self,x,y,size,frame):
       self.tracker = cv2.TrackerKCF_create()
       self.bbox = (x-size/2,y-size/2,size,size)
       self.tracker.init(frame,self.bbox)
       self.x = x
       self.y = y
       self.size = size
       self.ptsize = 4
   def update(self,frame,frameToDrawOn):
       ok, self.bbox = self.tracker.update(frame)
       if ok:
           self.x = int(self.bbox[0] + self.size/2)
           self.y = int(self.bbox[1] + self.size/2)
           p1 = (self.x-self.ptsize,self.y-self.ptsize)
           p2 = (self.x+self.ptsize,self.y+self.ptsize)
           cv2.rectangle(frameToDrawOn, p1, p2, (0,0,255),-1)

def Dist(p1,p2):
   x1=p1.x
   y1=p1.y
   x2=p2.x
   y2=p2.y
   return math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2))

def calcAngle(pTrack):
    a = Dist(pTrack[0],pTrack[1])
    b = Dist(pTrack[1],pTrack[2])
    c = Dist(pTrack[2],pTrack[0])
    angRad = math.acos(((a*a)+(b*b)-(c*c))/(2*a*b))
    return math.degrees(angRad)

def drawLine(frame,pTrack):
    cv2.line(frame,(pTrack[0].x,pTrack[0].y),(pTrack[1].x,pTrack[1].y),(255,0,255),2)
    cv2.line(frame,(pTrack[1].x,pTrack[1].y),(pTrack[2].x,pTrack[2].y),(255,0,255),2)

def main():
    kernel = np.ones((3,3),np.uint8)
    cap = cv2.VideoCapture('Video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280,720))
    ret, frame = cap.read()
    pList = [(561,421),(656,385),(584,263)]
    pTrack = []

    for pt in pList:
        pTrack.append(trackerPoint(pt[0],pt[1],80,frame))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frame is None):
            break
        thresh1 = cv2.inRange(frame,(170,170,170),(255,255,255))
        thresh1 = cv2.erode(thresh1,kernel,iterations = 3)
        thresh1 = cv2.dilate(thresh1,kernel,iterations = 3)

        res = cv2.bitwise_and(frame,frame,mask=thresh1)
        for p in pTrack:
            p.update(res, frame)

        drawLine(frame,pTrack)
        ang = calcAngle(pTrack)
        strAng = "%2.2f deg" % ang
        cv2.putText(frame,strAng,(pTrack[1].x+40,pTrack[1].y),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255))

        cv2.imshow('frame',frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
