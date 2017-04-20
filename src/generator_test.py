import generator
import maze_solver
import numpy as np

np.random.seed(5)


gen = generator.Prim(100, 100)
maze = maze_solver.pad_maze(gen.generate(), 3)
path = maze_solver.solver(maze)
print(len(path))