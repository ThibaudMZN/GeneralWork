#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Tracking 3 points on arm to calulate angle
'''

import math
import cv2
import numpy as np


class TrackerPoint(object):
    '''
    Allow to tracker input point with KCF algorithm
    '''

    def __init__(self, x, y, size, frame):
        # KCF tracker init
        self.tracker = cv2.TrackerKCF_create()
        self.bbox = (x - size / 2, y - size / 2, size, size)
        self.tracker.init(frame, self.bbox)
        self.x_pos = x
        self.y_pos = y
        self.size = size
        self.ptsize = 4

    def update(self, frame, frame_draw):
        '''
        Update the tracker position
        '''
        valid, self.bbox = self.tracker.update(frame)
        if valid:
            self.draw(frame_draw)

    def draw(self, frame_draw):
        '''
        Draw the updated point on frame
        '''
        self.x_pos = int(self.bbox[0] + self.size / 2)
        self.y_pos = int(self.bbox[1] + self.size / 2)
        p_1 = (self.x_pos - self.ptsize, self.y_pos - self.ptsize)
        p_2 = (self.x_pos + self.ptsize, self.y_pos + self.ptsize)
        cv2.rectangle(frame_draw, p_1, p_2, (0, 0, 255), -1)


def dist(p_1, p_2):
    '''
    Compute distance between two points
    '''
    x_1 = p_1.x_pos
    y_1 = p_1.y_pos
    x_2 = p_2.x_pos
    y_2 = p_2.y_pos
    return math.sqrt(math.pow((x_2 - x_1), 2) + math.pow((y_2 - y_1), 2))


def calc_angle(trackers):
    '''
    Calculate angle give 3 distance of ABC triangle
    '''
    dist_a = dist(trackers[0], trackers[1])
    dist_b = dist(trackers[1], trackers[2])
    dist_c = dist(trackers[2], trackers[0])
    ang_rad = math.acos(((dist_a * dist_a) + (dist_b * dist_b) -
                         (dist_c * dist_c)) / (2 * dist_a * dist_b))
    return math.degrees(ang_rad)


def draw_line(frame, trackers):
    '''
    Draw 2 lines between the 3 arm point
    '''
    cv2.line(frame, (trackers[0].x_pos, trackers[0].y_pos),
             (trackers[1].x_pos, trackers[1].y_pos), (255, 0, 255), 2)
    cv2.line(frame, (trackers[1].x_pos, trackers[1].y_pos),
             (trackers[2].x_pos, trackers[2].y_pos), (255, 0, 255), 2)


def main():
    '''
    Main program execution :
        - Open the Video
        - Initiate trackers on first frame
        - Update trackers on every frame,
                calculate angle and plot
    '''
    # Init kernel for erode / dilate
    kernel = np.ones((3, 3), np.uint8)
    # Init media in/out
    cap = cv2.VideoCapture('Video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))
    # Read first frame for trackers
    _, frame = cap.read()
    # Instantiate trackers at known positions
    tracker_list = [(561, 421), (656, 385), (584, 263)]
    trackers = []
    for pts in tracker_list:
        trackers.append(TrackerPoint(pts[0], pts[1], 80, frame))

    while cap.isOpened():
        # Read new frame
        _, frame = cap.read()
        if frame is None:
            break
        # Thresholde / Erosion / Dilatation for arm detection
        thresh1 = cv2.inRange(frame, (170, 170, 170), (255, 255, 255))
        thresh1 = cv2.erode(thresh1, kernel, iterations=3)
        thresh1 = cv2.dilate(thresh1, kernel, iterations=3)
        # Mask
        res = cv2.bitwise_and(frame, frame, mask=thresh1)
        # Update trackers
        for tracker in trackers:
            tracker.update(res, frame)
        draw_line(frame, trackers)
        # Calculate angle between points
        ang = calc_angle(trackers)
        string_ang = "%2.2f deg" % ang
        # Display it
        cv2.putText(frame, string_ang, (trackers[1].x_pos + 40, trackers[1].y_pos),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        # Show image
        cv2.imshow('frame', frame)
        # Write to output video
        out.write(frame)
        # "q" key to escape
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
