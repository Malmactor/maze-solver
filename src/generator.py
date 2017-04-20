"""Randomized Prim's Minimum Spanning Tree Algorithm for Maze Generation."""


from __future__ import print_function
import sys
import numpy as np


class Point:

    def __init__(self, r, c, parent=None):
        self.r = r
        self.c = c
        self.parent = parent

    def opposite(self):
        if self.r > self.parent.r:
            return Point(self.r+1, self.c, self)
        if self.r < self.parent.r:
            return Point(self.r-1, self.c, self)
        if self.c > self.parent.c:
            return Point(self.r, self.c+1, self)
        if self.c < self.parent.c:
            return Point(self.r, self.c-1, self)
        return None


class Prim:
    def __init__(self, r, c):
        self.r = r
        self.c = c

    def generate(self):
        """
        Generate the maze map.
        0 - road
        1 - wall
        2 - lava
        3 - start point
        4 - end point
        """

        maze_r = self.r-2
        maze_c = self.c-2

        # build maze and initialize with only walls and lavas
        maze = np.random.randint(2, size=(maze_r, maze_c))+1

        # select random point and open as start node
        start_x = np.random.random_integers(maze_r)-1
        start_y = np.random.random_integers(maze_c)-1
        maze[start_x][start_y] = 3
        start = Point(start_x, start_y)

        # iterate through direct neighbors of node
        frontier = list()
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x*y == 0 and x+y != 0:
                    if start.r+x >= 0 and start.r+x < maze_r and start.c+y >= 0 and start.c+y < maze_c:
                        frontier.append(Point(start.r+x, start.c+y, start))

        last = None
        while frontier:
            # pick current node at random
            current = frontier.pop(np.random.random_integers(len(frontier))-1)
            opposite = current.opposite()

            # if both node and its opposite are walls
            if opposite.r >= 0 and opposite.r < maze_r and opposite.c >= 0 and opposite.c < maze_c:
                if maze[current.r][current.c] in [1, 2] and maze[opposite.r][opposite.c] in [1, 2]:
                    # open path between the nodes
                    maze[current.r][current.c] = 0
                    maze[opposite.r][opposite.c] = 0

                    # store last node in order to mark it later
                    last = opposite
                    # iterate through direct neighbors of node, same as earlier
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if x*y == 0 and x+y != 0:
                                if opposite.r+x >= 0 and opposite.r+x < maze_r and opposite.c+y >= 0 and opposite.c+y < maze_c:
                                    frontier.append(Point(opposite.r+x, opposite.c+y, opposite))

        # if algorithm has resolved, mark end node
        if last:
            maze[last.r][last.c] = 4
            # add the surrounding obstacles
            out_maze = np.random.randint(2, size=(self.r, self.c))+1
            for x in range(maze_r):
                for y in range(maze_c):
                    out_maze[x+1][y+1] = maze[x][y]
            self.maze = out_maze
        else:
            self.maze = None

        return self.maze


def print_maze(maze):
    # print support
    printdict = {0: ' ', 1: '*', 2: '@', 3: 'S', 4: 'E', 5: '-'}
    r, c = maze.shape
    for x in range(r):
        for y in range(c):
            print(printdict[maze[x][y]], end='')
        print('', end='\n')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        num_rows, num_cols = [int(x) for x in sys.argv[1:]]
        prim = Prim(num_rows, num_cols)
        prim.generate()
        print_maze(prim.maze)
    else:
        print("Usage: prim.py num_rows num_cols")
        sys.exit()
