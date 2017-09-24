
import numpy as np

from dna import DNA


class Vehicule(object):
    def __init__(self, canvas, dna=None):
        self.pos = np.array([50, 200]).astype(np.float)
        self.vel = np.array([0, 0]).astype(np.float)
        self.acc = np.array([0, 0]).astype(np.float)
        self.completed = False
        self.crashed = False
        if(dna is None):
            self.dna = DNA()
        else:
            self.dna = dna
        self.fitness = 0
        self.size = 5
        self.canv = canvas
        self.idLine = self.canv.create_line(
            self.pointsLine, fill="white")
        self.id = self.canv.create_oval(
            self.points, fill="white", outline="white")

    def applyForce(self, force):
        self.acc += force

    def update(self, cnt, targetPos, obstPos):
        self.edges(targetPos, obstPos)
        if(not self.crashed and not self.completed):
            self.applyForce(self.dna.genes[cnt, :])
            self.vel += self.acc
            self.pos += self.vel
            self.acc *= 0

    def calcFitness(self, targetPos):
        d = self.distFromTarget(targetPos)
        if(d == 0):
            self.fitness = 1
        else:
            self.fitness = 1 / d
        if(self.crashed):
            self.fitness /= 10
        if(self.completed):
            self.fitness = 10

    def distFromTarget(self, targetPos):
        x0 = self.pos[0]
        y0 = self.pos[1]
        x1 = targetPos[0]
        y1 = targetPos[1]
        m0 = (x1 - x0) * (x1 - x0)
        m1 = (y1 - y0) * (y1 - y0)
        d = np.sqrt(m0 + m1)
        return d

    def edges(self, targetPos, obstPos):
        self.isInObst(obstPos)
        x0 = self.pos[0] - self.size
        x1 = self.pos[0] + self.size
        y0 = self.pos[1] - self.size
        y1 = self.pos[1] + self.size
        if(x0 < 0 or x1 > 800 or y0 < 0 or y1 > 400):
            self.crashed = True
        d = self.distFromTarget(targetPos)
        if(d < (self.size + 20)):
            self.completed = True
            self.pos = targetPos.copy()

    def isInObst(self, obPos):
        for p in obPos:
            xP = self.pos[0]
            yP = self.pos[1]
            x0 = p[0] - 20
            x1 = p[0] + 20
            if(p[0] == 500):
                y0 = p[1] - 50
                y1 = p[1] + 50
            else:
                y0 = p[1] - 100
                y1 = p[1] + 100
            if(xP > x0 and xP < x1 and yP > y0 and yP < y1):
                self.crashed = True

    def show(self):
        self.canv.coords(self.idLine, self.pointsLine)
        self.canv.coords(self.id, self.points)

    @property
    def pointsLine(self):
        x0 = self.pos[0]
        y0 = self.pos[1]
        x1 = self.pos[0] + self.vel[0] * 4
        y1 = self.pos[1] + self.vel[1] * 4
        return x0, y0, x1, y1

    @property
    def points(self):
        x0 = self.pos[0] - self.size
        y0 = self.pos[1] - self.size
        x1 = self.pos[0] + self.size
        y1 = self.pos[1] + self.size
        return x0, y0, x1, y1
