
import tkinter as tk
import numpy as np
import time

class trianglePoly(object):

    def __init__(self,canvas,x,y,size,angle):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.size = size
        self.angle = angle

    def rotate(self,x,y):
        th = self.angle * np.pi /180
        s = np.sin(th)
        c = np.cos(th)
        x -= self.x
        y -= self.y
        xn = x * c - y * s
        yn = x * s + y * c
        xn += self.x
        yn += self.y
        return xn, yn

    def animate(self):
        self.angle += 1        

    @property
    def points(self):
        x = list(range(3))
        y = list(range(3))
        x[0] = self.x
        y[0] = self.y + self.size
        x[1] = self.x + self.size/2
        y[1] = self.y - self.size
        x[2] = self.x - self.size/2
        y[2] = self.y - self.size

        for i in range(3):
            x[i], y[i] = self.rotate(x[i], y[i])
        
        return x[0],y[0],x[1],y[1],x[2],y[2]
        



if __name__ == "__main__":

    root = tk.Tk()

    c = tk.Canvas(root,width=400,height=400)
    c.pack()

    t = trianglePoly(c,200,200,50,0)
    for i in range(360):
        c.delete("all")
        t.animate()
        poly = c.create_polygon(t.points)
        c.update()
        time.sleep(0.01)
    
    root.mainloop()
