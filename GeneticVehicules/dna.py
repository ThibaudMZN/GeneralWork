import numpy as np


class DNA(object):
    def __init__(self, genes=None):
        self.LIFESPAN = 300
        self.maxSpeed = 1
        if(genes is not None):
            self.genes = genes
        else:
            self.genes = np.random.rand(
                self.LIFESPAN, 2) * self.maxSpeed - self.maxSpeed / 2

    def crossover(self, partner):
        newgenes = np.zeros((self.LIFESPAN, 2))
        for i in range(self.genes.shape[0]):
            r = np.random.rand(1)
            if(r > 0.5):
                newgenes[i, :] = self.genes[i, :]
            else:
                newgenes[i, :] = partner.genes[i, :]
        return DNA(newgenes)

    def mutation(self):
        for i in range(self.genes.shape[0]):
            r = np.random.rand(1)
            if(r < 0.01):
                self.genes[i, :] = np.random.rand(
                    1, 2) * self.maxSpeed - self.maxSpeed / 2
