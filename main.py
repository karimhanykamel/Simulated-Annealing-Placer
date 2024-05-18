import random
from math import exp
import time


# We created this function first, to parse a netlist given in a file (Part A)
def parse_netlist(filepath):
    # First we open the file in read mode
    with open(filepath, 'r') as file:
        # Then we read all lines from the file
        lines = file.readlines()

    # We had to strip the whitespace first from the first line and split the values into components
    # header = first line in the file
    header = lines[0].strip().split()

    # After splitting the header, the first value is the number of cells
    num_cells = int(header[0])
    # The second value is the number of nets
    num_nets = int(header[1])
    # This is to get the number of rows
    ny = int(header[2])
    # This is to get the number of cols
    nx = int(header[3])

    # After this, we initialize an empty list to store the nets, from the remaining lines in the file
    nets = []
    # We loop through each remaining line in the file, starting from the second line
    # lines[1:] skips the header line and starts from the second line
    for line in lines[1:]:
        parts = line.strip().split()
        # The first value indicates the size of the net
        net_size = int(parts[0])

        # The next values are the components of the net
        # Loops till net_size + 1, to include the last value in the line
        # We then add the list of components to the nets list
        components = [int(x) for x in parts[1:net_size + 1]]
        nets.append(components)

    # Finally we return the parsed values
    return num_cells, num_nets, ny, nx, nets


# Then we created this function, to randomly place the cells on the grid (Part B)
def initial_placement(num_cells, ny, nx):
    # First, we initialize the grid with -1 to indicate empty cells
    grid = [[-1 for _ in range(nx)] for _ in range(ny)]

    # Then we created a list, available_positions, to hold all possible positions on the grid, row by row
    # for example: [(0, 0), (0, 1), (1, 0), (1, 1)]
    available_positions = [(y, x) for y in range(ny) for x in range(nx)]

    # Then we shuffle the list to randomize the placement positions
    random.shuffle(available_positions)

    # After that, we assign a cell number to the random positions on the grid
    # For the number of cells we have:
    for cell_id in range(num_cells):
        # If available_positions is not empty
        if available_positions:
            # We get a position from the shuffled list
            y, x = available_positions.pop()
            # And place the cell number at the popped position on the grid
            grid[y][x] = cell_id

    # Finally, we return the grid with the initial placement of cells
    return grid


def calculate_wire_length(grid, nets):
    # Function to find the position (y, x) of a cell_id in the grid
    def position(cell_id):
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == cell_id:
                    return y, x
        return None

    wire_length = 0  # Initialize total wire length

    # Iterate over each net to calculate the wire length
    for net in nets:
        # Get the positions of all cells in the current net
        positions = [position(cell_id) for cell_id in net if position(cell_id) is not None]

        # If there are positions (the cells exist in the grid)
        if positions:
            # Find the minimum and maximum y and x coordinates
            min_y = min(pos[0] for pos in positions)
            max_y = max(pos[0] for pos in positions)
            min_x = min(pos[1] for pos in positions)
            max_x = max(pos[1] for pos in positions)

            # Calculate the half-perimeter wire length for this net
            wire_length += (max_y - min_y) + (max_x - min_x)

    return wire_length  # Return the total wire length


# As for the main function, to simulate the annealing placer, we added the following function (Part C and D)
def simulated_annealing(grid, nets, num_cells, cooling_rate=0.95):
    # We first calculate the initial cost (wire length) of the initial placement
    initial_cost = calculate_wire_length(grid, nets)
    # Then we set the initial temperature based on the initial cost
    initial_temp = 500 * initial_cost
    # And we set the final temperature based on the initial cost and number of nets
    final_temp = 5e-6 * initial_cost / len(nets)
    # After, we set the number of iterations (Moves per Temperature) based on the number of cells
    num_iterations = 20 * num_cells

    # After that, we initialize the current temperature and wire length
    current_temp = initial_temp
    current_wire_length = initial_cost

    # Then we set the initial grid and wire length as the best solution so for, which will be updated
    best_grid = [row[:] for row in grid]
    best_wire_length = current_wire_length

    # We needed to a function to keep updating the temperature, which will be needed for cooling down
    def schedule_temp(current_temp):
        return current_temp * cooling_rate

    # So as long as the current_temp is greater than the final temp, this will happen:
    while current_temp > final_temp:
        # For each iteration, which we got based on the number of cells:
        for _ in range(num_iterations):

            # We first pick two random positions
            y1, x1 = random.randint(0, ny - 1), random.randint(0, nx - 1)
            y2, x2 = random.randint(0, ny - 1), random.randint(0, nx - 1)

            # Then on the grid, we swap these positions, which is basically swapping 2 cells
            grid[y1][x1], grid[y2][x2] = grid[y2][x2], grid[y1][x1]

            # Then we calculate the new wire length after the swap
            new_wire_length = calculate_wire_length(grid, nets)

            # We then calculate the change in wire length which is delta L
            delta_L = new_wire_length - current_wire_length

            # We then have to decide if we will accept the new configuration
            if delta_L < 0 or random.random() < exp(-delta_L / current_temp):
                # If the change in wire length is negative, the new configuration is better, so we accept it
                # Or, if the new configuration is worse (delta_L > 0), we accept it with a probability
                # based on the exponential function. This probability decreases as delta_L increases
                # or as the temperature decreases, allowing it to escape local minima early on.
                current_wire_length = new_wire_length
                # If the new configuration is the best we have seen so far, we update the best wire length and grid
                if new_wire_length < best_wire_length:
                    best_wire_length = new_wire_length
                    best_grid = [row[:] for row in grid]
            else:
                # If the new configuration is not accepted, revert the swap to restore the previous state
                grid[y1][x1], grid[y2][x2] = grid[y2][x2], grid[y1][x1]

        # Cool down
        current_temp = schedule_temp(current_temp)

    return best_grid, best_wire_length


def grid_to_binary(grid):
    binary_grid = []
    for row in grid:
        binary_row = ''.join('0' if cell_id != -1 else '1' for cell_id in row)
        binary_grid.append(binary_row)
    return binary_grid


if __name__ == "__main__":

    filepath = 'd1.txt'
    num_cells, num_nets, ny, nx, nets = parse_netlist(filepath)
    grid = initial_placement(num_cells, ny, nx)
    initial_wire_length = calculate_wire_length(grid, nets)

    start_time = time.time()
    final_grid, final_wire_length = simulated_annealing(grid, nets, num_cells)
    end_time = time.time()
    duration = end_time - start_time

    print("\n")
    print("Parsing the netlist:")
    print("Number of cells:", num_cells)
    print("Number of nets:", num_nets)
    print("Grid dimensions:", ny, "x", nx)
    print("Nets:", nets)

    print("\n")
    print("Initial Random Placement:")
    for row in grid:
        print(' '.join(f"{cell_id:02}" if cell_id != -1 else "--" for cell_id in row))

    print("Initial Random Placement Wire Length:", initial_wire_length)

    print("\n")
    print("Final Placement:")
    for row in final_grid:
        print(' '.join(f"{cell_id:02}" if cell_id != -1 else "--" for cell_id in row))
    print("Final Wire Length:", final_wire_length)

    print("\nFinal Placement (Binary Format):")
    final_binary_grid = grid_to_binary(final_grid)
    for row in final_binary_grid:
        print(row)

    print("\n")
    print("Time taken for simulated annealing: {:.2f} seconds".format(duration))
