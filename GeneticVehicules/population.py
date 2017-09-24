
import numpy as np
import random

from vehicule import Vehicule


class Population(object):
    def __init__(self, popSize, canvas, targetPos, obstPos):
        self.vehicules = []
        self.popSize = popSize
        self.canvas = canvas
        self.targetPos = targetPos
        self.obstPos = obstPos
        for i in range(popSize):
            self.vehicules.append(Vehicule(self.canvas))

    def run(self, cnt):
        for i in range(len(self.vehicules)):
            self.vehicules[i].update(cnt, self.targetPos, self.obstPos)
            self.vehicules[i].show()

    def evaluate(self):
        maxFit = -1
        for i in range(len(self.vehicules)):
            self.vehicules[i].calcFitness(self.targetPos)
            if(self.vehicules[i].fitness > maxFit):
                maxfit = self.vehicules[i].fitness

        for i in range(len(self.vehicules)):
            self.vehicules[i].fitness /= maxfit

        self.matingpool = []
        for i in range(len(self.vehicules)):
            n = self.vehicules[i].fitness * 100
            for j in range(int(n)):
                self.matingpool.append(self.vehicules[i])

    def selection(self):
        newVehic = []
        for i in range(self.popSize):
            self.canvas.delete(self.vehicules[i].id)
            self.canvas.delete(self.vehicules[i].idLine)
            parentA = random.choice(self.matingpool).dna
            parentB = random.choice(self.matingpool).dna
            child = parentA.crossover(parentB)
            child.mutation()
            newVehic.append(Vehicule(self.canvas, child))
        self.vehicules = newVehic
