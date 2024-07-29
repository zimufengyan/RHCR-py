from basic_graph import BasicGraph
import time
import numpy as np

class BeeGraph(BasicGraph):
    def __init__(self):
        super().__init__()
        self.flowers = []
        self.flower_demands = []
        self.flower_costs = []
        self.flower_time_windows = []
        self.initial_locations = []
        self.bee_capacity = 0
        self.entrance = 0
        self.num_of_bees = 0
        self.max_timestep = 0
        self.move_cost = 0
        self.wait_cost = 0
        self.loading_time = 0
        self.preprocessing_time = 0

    def load_map(self,fname):
        try:
            with open(fname, 'r') as file:
                start_time = time.time()
                self.map_name = fname.split('.')[0]


                # Read size
                file.readline()
                self.rows = int(file.readline().strip())
                self.cols = self.rows

                # Moves
                self.move[0] = 1
                self.move[1] = -self.cols
                self.move[2] = -1
                self.move[3] = self.cols

                # Number of obstacles
                file.readline()
                num_of_obstacles = int(file.readline().strip())

                # Number of flowers
                file.readline()
                num_of_flowers = int(file.readline().strip())
                self.flowers = [0] * num_of_flowers
                self.flower_demands = [0] * num_of_flowers
                self.flower_costs = [0] * num_of_flowers
                self.flower_time_windows = [0] * num_of_flowers

                # Number of bees
                file.readline()
                self.num_of_bees = int(file.readline().strip())

                # Initial locations
                file.readline()
                num_of_initial_locations = int(file.readline().strip())
                self.initial_locations = [0] * num_of_initial_locations

                # Max timestep
                file.readline()
                self.max_timestep = int(file.readline().strip())

                # Bee capacity
                file.readline()
                self.bee_capacity = int(file.readline().strip())

                # Flower demands
                file.readline()
                for i in range(num_of_flowers):
                    self.flower_demands[i] = int(file.readline().strip())

                # Wait cost
                file.readline()
                self.wait_cost = int(file.readline().strip())

                # Move cost
                file.readline()
                self.move_cost = int(file.readline().strip())

                # Flower costs
                file.readline()
                for i in range(num_of_flowers):
                    self.flower_costs[i] = int(file.readline().strip())

                # Flower locations
                file.readline()
                for i in range(num_of_flowers):
                    self.flowers[i] = int(file.readline().strip()) - 1

                # Entrance location
                file.readline()
                self.entrance = int(file.readline().strip()) - 1

                # Initial locations
                file.readline()
                for i in range(num_of_initial_locations):
                    self.initial_locations[i] = int(file.readline().strip()) - 1

                # Read map and weights
                self.types = [""] * (self.rows * self.cols)
                self.weights = np.full((self.rows * self.cols, 5), float('inf'))

                for i in range(self.rows):
                    line = file.readline().strip()
                    for j in range(self.cols):
                        id = self.cols * i + j
                        if line[j] == '.':
                            self.types[id] = "Travel"
                        else:
                            self.types[id] = "Obstacle"

                ##
                self.types[entrance] = "Magic"

                for i in range(self.rows * self.cols):
                    if self.types[i] == "Obstacle":
                        continue
                    elif self.types[i] == "Magic":
                        self.weights[i][4] = wait_cost
                    else:
                        self.weights[i][4] = wait_cost

                    for dir in range(4):
                        if 0 <= i + self.move[dir] < self.rows * self.cols and \
                                self.get_manhattan_distance(i, i + self.move[dir]) <= 1 and \
                                self.types[i + self.move[dir]] != "Obstacle":
                            self.weights[i][dir] = move_cost

                self.loading_time = time.time() - start_time
                #print(f"Map size: {self.rows}x{self.cols}")
                #print(f"Done! ({self.loading_time} s)")
                return True
        except FileNotFoundError:
            print(f"Parameter file {fname} does not exist.")
            return False

    def load_Nathan_map(self, fname):
        try:
            with open(fname, 'r') as myfile:
                print("*** Loading map ***")
                start_time = time.time()
                pos = fname.rfind('.')
                self.map_name = fname[:pos]

                # Skip first line
                myfile.readline()

                # Read number of rows
                line = myfile.readline().strip()
                parts = line.split()
                self.rows = int(parts[1])

                # Read number of cols
                line = myfile.readline().strip()
                parts = line.split()
                self.cols = int(parts[1])

                self.move[0] = 1
                self.move[1] = -self.cols
                self.move[2] = -1
                self.move[3] = self.cols

                # Skip "map" line
                myfile.readline()

                # Read types and edge weights
                self.types = [""] * (self.rows * self.cols)
                self.weights = [[float('inf')] * 5 for _ in range(self.rows * self.cols)]

                for i in range(self.rows):
                    line = myfile.readline().strip()
                    for j in range(self.cols):
                        id = self.cols * i + j
                        if line[j] == '.':
                            self.types[id] = "Travel"
                        else:
                            self.types[id] = "Obstacle"

                for i in range(self.rows * self.cols):
                    if self.types[i] == "Obstacle":
                        continue
                    for dir in range(4):
                        if 0 <= i + self.move[dir] < self.rows * self.cols and \
                                self.get_manhattan_distance(i, i + self.move[dir]) <= 1 and \
                                self.types[i + self.move[dir]] != "Obstacle":
                            self.weights[i][dir] = 1

                runtime = time.time() - start_time
                print(f"Map size: {self.rows}x{self.cols}")
                print(f"Done! ({runtime} s)")
                return True
        except FileNotFoundError:
            print(f"Map file {fname} does not exist.")
            return False


    ##额外自己加的
    def get_manhattan_distance(self, start, end):
        start_row, start_col = divmod(start, self.cols)
        end_row, end_col = divmod(end, self.cols)
        return abs(start_row - end_row) + abs(start_col - end_col)

    def preprocessing(self, fname, consider_rotation):
        start_time = time.time()
        self.consider_rotation = consider_rotation
        self.heuristics[self.entrance] = self.compute_heuristics(self.entrance)

        #这一块不太懂，不知道对不对

        with open(fname, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                for id in map(int, parts[2:]):
                    id -= 1
                    if id not in self.heuristics:
                        self.heuristics[id] = self.compute_heuristics(id)

        self.preprocessing_time = time.time() - start_time
        #print(f"Done! ({self.preprocessing_time} s)")



