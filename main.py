import random
from math import exp
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# History for plotting line graph
twl_history = []
temperature_history = []

class Net:
    def __init__(self):
        self.rows = defaultdict(int)
        self.columns = defaultdict(int)
        self.wire_length = 0

def isFull(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != -1

def printGrid(grid):
    column_width = 5  # Set the desired width for the columns
    for i in range(ny):
        for j in range(nx):
            if not isFull(grid, i, j):
                print('--'.ljust(column_width), end=' ')
            else:
                print(f"{grid[i][j]:02}".ljust(column_width), end=' ')
        print()
    print()

def printGridBinary(grid):
    for i in range(ny):
        for j in range(nx):
            if isFull(grid, i, j):
                print("0", end=' ')
            else:
                print("1", end=' ')
        print()
    print()

def updateAffectedNets(cell_id, row, col, cell_net_mapping, nets):
    global total_wire_length

    for net in cell_net_mapping[cell_id]:
        current_wire_length = nets[net].wire_length

        nets[net].rows[row] += 1
        nets[net].columns[col] += 1

        new_min_row = min(nets[net].rows.keys())
        new_min_col = min(nets[net].columns.keys())
        new_max_row = max(nets[net].rows.keys())
        new_max_col = max(nets[net].columns.keys())

        new_wire_length = new_max_row - new_min_row + new_max_col - new_min_col

        nets[net].wire_length = new_wire_length

        total_wire_length += new_wire_length - current_wire_length

def create_cell_net_mapping(file, cell_net_mapping):
    for i in range(num_nets):
        line = file.readline().strip().split()
        for component in map(int, line[1:]):
            cell_net_mapping[component].append(i)

def findEmptySpace(grid):
    while True:
        row = random.randint(0, ny - 1)
        col = random.randint(0, nx - 1)
        if not isFull(grid, row, col):
            return row, col

def intitialRandomPlacement(grid, cell_net_mapping, nets):
    for cell_id in range(num_cells):
        row, col = findEmptySpace(grid)
        grid[row][col] = cell_id
        updateAffectedNets(cell_id, row, col, cell_net_mapping, nets)

def swap(row1, col1, row2, col2, grid, cell_net_mapping, nets):
    delta = 0
    cell1 = grid[row1][col1]
    cell2 = grid[row2][col2]

    # If the cells are full, update the affected nets for first cell
    if isFull(grid, row1, col1):
        for net in cell_net_mapping[cell1]:
            current_wire_length = nets[net].wire_length

            nets[net].rows[row1] -= 1
            if nets[net].rows[row1] == 0:
                del nets[net].rows[row1]

            nets[net].columns[col1] -= 1
            if nets[net].columns[col1] == 0:
                del nets[net].columns[col1]

            nets[net].rows[row2] += 1
            nets[net].columns[col2] += 1

            new_min_row = min(nets[net].rows.keys())
            new_min_col = min(nets[net].columns.keys())
            new_max_row = max(nets[net].rows.keys())
            new_max_col = max(nets[net].columns.keys())
            
            new_wire_length = new_max_row - new_min_row + new_max_col - new_min_col

            nets[net].wire_length = new_wire_length

            delta += new_wire_length - current_wire_length

    # Update the affected nets for the second cell
    if isFull(grid, row2, col2):
        for net_idx in cell_net_mapping[cell2]:
            current_wire_length = nets[net_idx].wire_length

            nets[net_idx].rows[row2] -= 1
            if nets[net_idx].rows[row2] == 0:
                del nets[net_idx].rows[row2]

            nets[net_idx].columns[col2] -= 1
            if nets[net_idx].columns[col2] == 0:
                del nets[net_idx].columns[col2]

            nets[net_idx].rows[row1] += 1
            nets[net_idx].columns[col1] += 1

            new_min_row = min(nets[net_idx].rows.keys())
            new_min_col = min(nets[net_idx].columns.keys())
            new_max_row = max(nets[net_idx].rows.keys())
            new_max_col = max(nets[net_idx].columns.keys())
            new_wire_length = new_max_row - new_min_row + new_max_col - new_min_col

            nets[net_idx].wire_length = new_wire_length

            delta += new_wire_length - current_wire_length

    return delta

def simulatedAnnealing(file, cooling_rate):
    global num_cells, num_nets, ny, nx, total_wire_length, cell_net_mapping, grid, nets, twl_history, temperature_history

    random.seed(42)

    # Reset global variables
    twl_history = []
    temperature_history = []
    
    nets = [Net() for _ in range(num_nets)]
    cell_net_mapping = [[] for _ in range(num_cells)]
    grid = [[-1 for _ in range(nx)] for _ in range(ny)]

    create_cell_net_mapping(file, cell_net_mapping)

    total_wire_length = 0
    intitialRandomPlacement(grid, cell_net_mapping, nets)

    initial_total_wire_length = total_wire_length
    print("Initial placement: \n")
    printGrid(grid)
    print(f"Initial total wire length: {total_wire_length}")

    initial_temp = 500.0 * initial_total_wire_length
    final_temp = 5e-6 * initial_total_wire_length / num_nets
    curr_temp = initial_temp

    # Store initial values for plotting
    twl_history.append(total_wire_length)
    temperature_history.append(curr_temp)

    # Simulated Annealing
    while curr_temp > final_temp:
        # Randomly swap cells
        for _ in range(10 * num_cells):
            row1, col1, row2, col2 = None, None, None, None
            row1, col1 = random.randint(0, ny - 1), random.randint(0, nx - 1)
            row2, col2 = random.randint(0, ny - 1), random.randint(0, nx - 1)

            # Swap the cells and calculate the change in wire length
            delta = swap(row1, col1, row2, col2, grid, cell_net_mapping, nets)
            grid[row1][col1], grid[row2][col2] = grid[row2][col2], grid[row1][col1]
            total_wire_length += delta

            if delta >= 0:
                rand_val = random.random()
                threshold = 1 - exp(-delta / curr_temp)
                if rand_val < threshold:
                    delta = swap(row1, col1, row2, col2, grid, cell_net_mapping, nets)
                    grid[row1][col1], grid[row2][col2] = grid[row2][col2], grid[row1][col1]
                    total_wire_length += delta

        # Reduce temperature
        curr_temp *= cooling_rate

        # Store current values for plotting
        twl_history.append(total_wire_length)
        temperature_history.append(curr_temp)

    print("Final placement: ")
    printGrid(grid)
    print(f"Total wire length: {total_wire_length}")
    return total_wire_length

def plot_temperature_vs_wire_length():
    plt.figure()
    plt.plot(temperature_history, twl_history, marker='o')
    plt.xlabel('Temperature (log2 scale)')
    plt.ylabel('Total Wire Length (TWL)')
    plt.title('Temperature vs Total Wire Length')
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.gca().invert_xaxis()
    plt.show()

def plot_twl_v_cooling_rate(twls):
    cooling_rates = [0.75, 0.80, 0.85, 0.90, 0.95]
    plt.figure()
    plt.plot(cooling_rates, twls, marker='o', linestyle='-')
    plt.xlabel('Cooling Rate')
    plt.ylabel('Final Total Wire Length (TWL)')
    plt.title('Effect of Cooling Rate on Final Total Wire Length')
    plt.grid(True)
    plt.show()

def run_experiments(filepath):
    cooling_rates = [0.75, 0.80, 0.85, 0.90, 0.95]
    final_wire_lengths = []

    for cooling_rate in cooling_rates:
        with open(filepath, 'r') as file:
            global num_cells, num_nets, ny, nx
            num_cells, num_nets, ny, nx = map(int, file.readline().strip().split())
            final_wire_length = simulatedAnnealing(file, cooling_rate)
            final_wire_lengths.append(final_wire_length)
    return final_wire_lengths

if __name__ == "__main__":
    filename = input("Enter file name: ")

    try:
        with open(filename, 'r') as file:
            num_cells, num_nets, ny, nx = map(int, file.readline().strip().split())
            cooling_rate = 0.95
            start = time.time()
            simulatedAnnealing(file, cooling_rate)
            end = time.time()
            duration = end - start

            print(f"Time Elapsed: {duration:.6f} s.")
            plot_temperature_vs_wire_length()
            
            final_wire_lengths = run_experiments(filename)
            plot_twl_v_cooling_rate(final_wire_lengths)

    except FileNotFoundError:
        print("File not found.")
