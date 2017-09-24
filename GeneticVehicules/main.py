
import tkinter as tk
import numpy as np

from population import Population
from dna import DNA


class RootWindow(object):
    def __init__(self):
        self.win = tk.Tk()
        self.configWin()
        self.spawnTarget()
        self.spawnObst()
        self.pop = Population(100, self.canv, self.targetPos, self.obstPos)
        self.LIFESPAN = DNA().LIFESPAN
        self.cnt = 0
        self.win.mainloop()

    def configWin(self):
        col = '#%02x%02x%02x' % (51, 51, 51)
        self.fps = int(1000 / 60)
        self.canv = tk.Canvas(self.win, width=800,
                              height=400, background=col)
        self.canv.pack()
        self.win.after(self.fps, self.update)

    def update(self):
        # DO ALL THE STUFF HERE
        self.pop.run(self.cnt)
        self.cnt += 1
        if(self.cnt == self.LIFESPAN):
            self.pop.evaluate()
            self.pop.selection()
            self.cnt = 0

        self.canv.update()
        self.win.after(self.fps, self.update)

    def spawnObst(self):
        l = ([300, 200], [500, 200], [600, 50], [600, 350])
        self.obstPos = np.array(l)
        for p in l:
            x0 = p[0] - 20
            x1 = p[0] + 20
            if(p[0] == 500):
                y0 = p[1] - 50
                y1 = p[1] + 50
            else:
                y0 = p[1] - 100
                y1 = p[1] + 100
            self.obstID = self.canv.create_rectangle(
                x0, y0, x1, y1, fill="red", outline="red")

    def spawnTarget(self):
        self.targetPos = np.array([750, 200])
        x0 = self.targetPos[0] - 10
        x1 = self.targetPos[0] + 10
        y0 = self.targetPos[1] - 10
        y1 = self.targetPos[1] + 10
        self.targetId = self.canv.create_oval(
            x0, y0, x1, y1, fill="green", outline="green")


if __name__ == "__main__":
    w = RootWindow()
