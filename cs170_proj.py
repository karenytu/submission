import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import SetCoverPy.setcover as setcover
import utils
from student_utils_sp18 import *
import networkx as nx
# import importlib
# python_christofides = importlib.import_module("C:\\python\\python36-32\\lib\\site-packages\\python-christofides")
from pyChristofides import christofides
from random import shuffle
import os
import sys

# n is the number of kingdoms
# c is the amount of connectivity; it must be between 0 and 1
def input_generator(n, graph, c = 0.5):
	# each vertex has a name, be boring and have each vertex_name = number Ex: '0', '1', etc.
	vertex_names = range(n)
	vertex_start = np.random.randint(n)

	# adjacency matrix starts with all 'x' (no vertices are connected), must be symmetric
	adjacency_matrix = [["x" for i in vertex_names] for j in vertex_names]
	coordinate_locations = point_generator(n, graph)

	def distance(p1_i, p2_i):
		if p1_i == p2_i:
			return 0
		point1 = coordinate_locations[p1_i]
		point2 = coordinate_locations[p2_i]

		return np.round(np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2),4)


	complete_graph_matrix = [[distance(row,col) for col in range(n)] for row in range(n)]
	mst_adjacency_matrix = sp.csgraph.minimum_spanning_tree(complete_graph_matrix).toarray()

	# updating adjacency matrix to have the MST
	for j in range(n):
		for i in range(n):
			if mst_adjacency_matrix[j][i] != 0:
				adjacency_matrix[j][i] = mst_adjacency_matrix[j][i]
				adjacency_matrix[i][j] = mst_adjacency_matrix[j][i]
				p1 = coordinate_locations[i]
				p2 = coordinate_locations[j]
				plt.plot([p1[0],p2[0]],[p1[1],p2[1]])

	# then add more edges
	# randomly?? how many?? TODO
	# total_edges = np.sum([[1 if e != 'x' else 0 for e in row] for row in adjacency_matrix])/2
	# only an MST has been added, so total_edges = n - 1
	# print("edges: ", total_edges)
	# print("to add: ", ((n-1)+total_edges.astype(int))//2)
	# for i in range((total_edges.astype(int))//2):
	for i in range(n):
		row = np.random.randint(n)
		col = np.random.randint(n)

		while(row == col or adjacency_matrix[row][col] != "x"):
			row = np.random.randint(n)
			col = np.random.randint(n)
		if adjacency_matrix[row][col] == 'x':
			adjacency_matrix[row][col] = distance(row,col)
			adjacency_matrix[col][row] = distance(row,col)
			p1 = coordinate_locations[row]
			p2 = coordinate_locations[col]
			plt.plot([p1[0],p2[0]],[p1[1],p2[1]])

	if (graph == True):
		plt.show()

	adjacency_matrix_formatted = np.array([[0 if e == 'x' else e for e in row] for row in adjacency_matrix])

	visualize(adjacency_matrix_formatted)

	# then add conquering costs
	for u in range(n):
		# how to determine how much it costs to conquer? TODO
		adjacency_matrix[u][u] = np.random.randint(1,500)

	print_input(n, vertex_names, vertex_start, adjacency_matrix)
	filename = "inputs\\"+str(n) + ".in"
	write_to_input(filename, n, vertex_names, vertex_start, adjacency_matrix)
	return

def print_input(n, vertex_names, vertex_start, adjacency_matrix):
	print(n) # number of kingdoms
	# print kingdom names
	for name in vertex_names:
		print(name, "", end='')
	print("")
	#print starting vertex 
	print(vertex_start)
	# print adjacency matrix
	for row in adjacency_matrix:
		for element in row:
			print(element, "", end='')
		print("")

def write_to_input(filename, n, vertex_names, vertex_start, adjacency_matrix):
	print("writing to file: " + filename)
	f=open(filename,"w")
	f.write(str(n) + "\n")
	for name in vertex_names:
		f.write(str(name) + " "),
	f.write("\n")
	f.write(str(vertex_start) + "\n")
	for row in adjacency_matrix:
		for element in row:
			f.write(str(element))
			f.write(" ")
		f.write("\n")
	f.close()
	return

def gadget_input_generator(n):
	vertex_names = range(n)
	vertex_start = np.random.randint(n)
	nodes_remaining = n
	gadget_size = n/10 # TODO: maybe change later to have all gadgets be different sizes...

	# UNFINISHED
	# when generating gadgets, half of the time optimal_circle = True
	while nodes_remaining > 0:
		if nodes_remaining >= 5:
			# do something
			print("UNFINISHED")
	print_input(n, vertex_names, vertex_start, adjacency_matrix)
	filename = "gadget_inputs\\"+str(n) + ".in"
	write_to_input(filename, n, vertex_names, vertex_start, adjacency_matrix)
	return

def solver1(list_of_kingdom_names, starting_kingdom, adjacency_matrix):
	# takes in an input file
	# finds some kind of tour and it's cost
	# because the graph is metric, if two kingdoms are neighbors, direct path shortest path
		# so if neighboring kingdoms are both conquered, they should be visited consecutively
	# it needs to actually make every kingdom surrender

	# use set cover to figure out which kingdoms to conquer
	# input_data = utils.read_file("inputs\\" + input_file + ".in")
	n = len(list_of_kingdom_names)
	# n, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)
	start_index = list_of_kingdom_names.index(starting_kingdom)

	cost = []

	for i in range(n):
		cost.append(adjacency_matrix[i][i]) # cost = conquering cost
		# cost.append(-np.sum(np.array([0 if x == "x" else 1 for x in adjacency_matrix[i]])))# cost = degree of vertices
		# adjacency_matrix[i][i] = 0

	boolean_matrix = np.array([[False if e=="x" else True for e in row] for row in adjacency_matrix])

	g = setcover.SetCover(boolean_matrix, cost)
	solution, time_used = g.SolveSCP()
	optimal_columns = g.s # boolean array; index i is true if vertex i is being conquered
	conquering_cost = g.total_cost
	# print(optimal_columns)

	# TSP to find tour through all vertices
	# adjacency_matrix_formatted = np.array([[0 if e=="x" else e for e in row] for row in adjacency_matrix])
	# print(adjacency_matrix_formatted)
	G = adjacency_matrix_to_graph(adjacency_matrix)
	
	# try to get a good ordering of all kingdoms
	adjacency_matrix_formatted = np.array([[0 if e=="x" else e for e in row] for row in adjacency_matrix])

	path = []
	for i in range(n):
		if optimal_columns[i] == True:
			path.append(i)
	conquered_kingdoms = list(path)
	# print("unshuffled kingdoms: ", path)

	T = nx.minimum_spanning_tree(G)
	# construct a path that goes through the MST
	mst_tour = []
	mst_edges_dict = {(s,e): False for (s,e) in T.edges}

	# current_edge = None
	current_vertex = start_index
	while mst_edges_dict: # while there are not fully traversed edges
		# find an edge from current_vertex to some other vertex
		current_edge = None
		keys = list(mst_edges_dict.keys())
		end = False
		i = 0
		while end == False and i < len(keys):
			e = keys[i]
			# normal case: an untraversed edge in the tree is found from the current vertex
			if mst_edges_dict[e] == False: # not traversed at all yet!
				if e[1] == current_vertex :
					current_edge = e
					mst_edges_dict.pop(e)
					mst_edges_dict[(e[1],e[0])] = True
					current_vertex = e[0]
					# print("traverse 1")
					# print(current_vertex)
					end = True

				elif e[0] == current_vertex:
					current_edge = e
					mst_edges_dict[e] = True
					current_vertex = e[1]
					# print("traverse 2")
					# print(current_vertex)
					end = True
			i += 1

		# if there are none, take the return edge if there are no more edges..
		removed = None
		if current_edge == None:
			keys = list(mst_edges_dict.keys())
			end = False
			i = 0
			while end == False and i < len(keys):
				e = keys[i]
				if mst_edges_dict[e] == True and e[1] == current_vertex:
					mst_edges_dict.pop(e) # this edge has been taken twice, cannot be traversed anymore
					removed = e
					current_vertex = e[0]
					# print("return")
					# print(removed)
					end = True
				i += 1
			# print("failed to remove??")
			# print(e)
			# print(current_vertex)
			# print(keys)

		mst_tour.append(current_vertex)

		if removed == None and current_edge == None:
			print("whaaaat")
			mst_edges_dict = False
	# print(mst_tour)
	mst_path = []
	for kingdom in mst_tour:
		if kingdom in conquered_kingdoms:
			if len(mst_path) >= 1:
				if kingdom != mst_path[-1]:
					mst_path.append(kingdom)
			else:
				mst_path.append(kingdom)
	# print("ALTERNATE kingdom order: ")
	# print(mst_path)
	path = mst_path

	# # OR...put the kingdoms in a random order! hmm... this seems to work better
	# # np.random.shuffle(path)
	# # print("shuffled?: ", path)

	def shift(seq, shift=1):
		return seq[-shift:]+ seq[:-shift]

	# make sure the path starts and ends with the start node
	if start_index in path:
		shift_num = len(path) - path.index(start_index)
		path = shift(path, shift_num)
	else:
		path.insert(0, start_index)
	path.append(start_index)

	# but the list of start node + kingdoms to conquer + start node is not necessarily connected
	index = 1
	while index < len(path):
		current_kingdom = path[index-1] # the index of the current kingdom
		next_kingdom = path[index] # the index of the next kingdom
		if adjacency_matrix[current_kingdom][next_kingdom] == 'x':
			inter_path = nx.dijkstra_path(G,source=current_kingdom,target=next_kingdom)
			if index == len(path) - 1:
				path = path[:index-1] + inter_path
			else:
				path = path[:index-1] + inter_path + path[index+1:]
			index += len(inter_path) - 1
		else:
			index += 1

	# # print("processed path (TOUR)")
	# # print(path)
	tour = path
	travel_cost = 0
	for i in range(len(tour) - 1):
		travel_cost += float(adjacency_matrix[tour[i]][tour[i+1]])
	conquering_cost = 0
	for i in conquered_kingdoms:
		conquering_cost += float(adjacency_matrix[i][i])
	cost = conquering_cost + travel_cost

	a = np.array(list_of_kingdom_names)
	tour = a[path]
	conquered_kingdoms = a[conquered_kingdoms]

	# print("conquered kingdoms: ", conquered_kingdoms)
	# print("entire path through graph: ", tour)
	# print("cost: ", cost)
	return tour, conquered_kingdoms, cost

	# if optimal_cost == 0 or cost < optimal_cost:
	# 	filename = output_folder + "\\"+input_file + ".out"
	# 	print("writing to file: " + filename)
	# 	f=open(filename,"w")

	# 	for kingdom_i in tour:
	# 		f.write(str(list_of_kingdom_names[kingdom_i]) + " "),
		
	# 	f.write("\n")

	# 	for kingdom_i in conquered_kingdoms:
	# 		f.write(str(list_of_kingdom_names[kingdom_i]) + " ")
	# 	f.close()


	# modify tour so that it keeps the same order of visiting kingdoms, but only visit the ones necessary

	# OR...construct a new matrix with only the conquered kingdoms, run TSP on this new matrix
		# create a dictionary to reindex kingdoms to be conquered
			# map new index to original index
		# create a new matrix
			# for all cells, calculate shortest paths
		# run TSP on new matrix

	# for later: randomly swap the order of kingdoms to see if it improves the cost
	# randomly swap out kingdoms to conquer to see if it improves the cost

def solver2(input_file):
	input_data = utils.read_file(input_file)
	n, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)

	# 1. Select the start city.
	# 2. Find the most valuable neighboring city and go there.
	# 3. Are there any unvisitied cities left? If yes, repeat step 2.
	# 4. Return to the start city.

	conquered = []

	# start
	kingdom = starting_kingdom

	# while not all kingdoms have been conquered
	while len(conquered) != n:
		print(kingdom)
		# find best neighbor
		best_neighbor_value = float("inf")
		best_neighbor = None

		for i in range(adjacency_matrix[kingdom]):
			if i != kingdom:
				edge = adjacency_matrix[kingdom][i]
				h = heuristic(i)

				if edge + h < best_neighbor_value:
					best_neighbor = i
		kingdom = best_neighbor

	# find path home
	G=nx.from_numpy_matrix(np.array(adjacency_matrix))
	shortest_path(G, kingdom, starting_kingdom)
	print(shortest_path)


	# greedy
	# heuristic: returns p (probability to conquer), h (heuristic -- smaller is more valuable)
	# def heuristic(n):
	# 	p1 = 0.4
	# 	p2 = 0.6

	# 	if conquered:
	# 		return float("inf")
	# 	else if surrendered:
	# 		all_surrendered = True
	# 		for n in neighbors:
	# 			if n not conquered and n not surrendered:
	# 				all_surrendered = False
	# 		if all_surrendered:
	# 			return float("inf")
	# 	else:
	# 		all_surrendered = True
	# 		none_surrendered = False
	# 		for n in neighbors:
	# 			if n surrendered:
	# 				none_surrendered = False
	# 			if n not surrendered:
	# 				all_surrendered = False
	# 		if all_surrendered:
	# 			return 0
	# 		if none_surrendered:
	# 			conquer with p1
	# 		conquer with p2

def point_generator(n, graph):
	points = {(1,2),(1,2)}
	generated_points = np.random.random_integers(500,size=(2,n))
	x = generated_points[0]
	y = generated_points[1]
	points = {(x[i],y[i]) for i in range(n)}
	while(len(points) != n): #in case there are duplicates
		points.add((np.random.randint(500),np.random.randint(500)))

	if graph == True:
		plt.scatter(x,y)
	print("coordinates: ")
	print(list(points))
	return list(points)

def scramble(matrix):
	# find indices
	indices = np.arange(len(matrix))
	np.random.shuffle(indices)

	# shuffle matrix
	matrix = np.array(matrix)
	matrix = matrix[indices].T
	matrix = matrix[indices].T

	return matrix

def gadget_generator(n, optimal_circle = True):
	adjacency_matrix = [["x" for i in range(n)] for j in range(n)]
	# index 0 = secret node
		# EDGES connected to ALL other nodes in the gadget = x
		# Conquer cost = d
	# index != 0: ring nodes
		# index 1 = start
		# EDGE ring node to ring node = y (1 to 2...n-2 to n-1, n-1 to 1)
		# Conquer cost of ring node = c

	# in the optimal, do NOT conquer the secret node, try to trick people into conquering the secret node
	# optimal tour: go through all ring nodes (n - 2 edges), conquer every other 3 ring nodes
		# travel cost = (n-2) * y
		# conquer cost
			# if (n-1)%3 == 0: ((n-1)/3)*c
			# floor_div((n-1)/2)*c
		# exit is NOT necessarily a neighbor of start, exit = last node conquered
		# total_cost = (n-2)*y + ((n-1)/3)*c
	# circle cost < center_cost, ring kingdoms to travel to*y + ((n-1)/3)*c < 2x+d
		# ring kingdoms to travel to*y + ((n-1)/3)*c -2x < d
		# to trick set cover, d < ((n-1)/3)*c
	# y = 20
	# c = 10
	y = np.random.randint(10, 500)
	c = y//2
	# c = np.random.randint(1, 500)
	# x value changes depending on if we want the optimal solution to include the center or not
	circle_neighbors_conquered = (n-1)/3 if (n-1)%3==0 else (n+1)//3
	start_index = 1
	exit_index = n-1 # always exit at the neighbor of the start node; start node is always conquered, exit node never conquered
	
	if optimal_circle == True:
		# y//2 < x < y
		#we need this lower bound for x; otherwise the optimal route is to pass through the center node...forcing optimal route to avoid center
		# to trick people into choosing the center, either make d smaller or c bigger
		error_offset = 500
		x = np.random.randint(3*y//2 + 1, 3*y)
		# print("x upperbound: ", (n-1)*y + circle_neighbors_conquered*c)
		# x = np.random.randint(((n-2)*y)//2,(circle_neighbors_conquered*c - 1)//2)
		# make x > y
		d=(exit_index-start_index + 1)*y + circle_neighbors_conquered*c - 2*x + error_offset #500 is the offset from the optimal, if the person decides to conquer the center
		kingdoms_conquered = []
		i = 1
		while i < n:
			kingdoms_conquered.append(i)
			i += 3
			# if i > n-1:
			# 	exit_index = i - 3
			if i == n-1: # never conquer the neighbor of the start node
				i -= 1 # have it conquer the second to last instead

		tour = [i for i in range(1, exit_index + 1)]
	else:
		# in the optimal, conquer the secret node, try to trick people into conquering the ring nodes
		# optimal tour: go from start to secret node, conquer secret node
		# travel cost = 2x, conquer cost = d, total_cost = 2x + d, circle cost > center_cost
		x = np.random.randint(3*y//2 + 1, 3*y) # make x kind of big to trick people into avoid the center
		# x = np.random.randint(y + 1, 2*y)
		d=(exit_index-start_index)*y + c - 2*x - 1
		while d <= 0:
			x = np.random.randint(3*y//2 + 1, 3*y)
			d=(exit_index-start_index)*y + c - 2*x - 1
		# d > circle_neighbors_conquered*c
		kingdoms_conquered = [0]
		tour = [1,0, exit_index]
	
	# set edges/conquering costs for each neighbor
	for row in range(1,n):
		adjacency_matrix[row][row] = c
		if row == n-1:
			adjacency_matrix[row][1] = y
			adjacency_matrix[1][row] = y
		else:
			adjacency_matrix[row][row+1] = y
			adjacency_matrix[row+1][row] = y

	adjacency_matrix[0][0] = d
	for col in range(1,n):
		adjacency_matrix[0][col] = x
		adjacency_matrix[col][0] = x

	circle_cost = (exit_index-start_index)*y + circle_neighbors_conquered*c
	center_cost = 2*x + d

	# G = adjacency_matrix_to_graph(adjacency_matrix)
	# nx.draw(G, with_labels = True)
	# plt.show()

	# helpful for debugging, comment out if these print statements are annoying
	for row in adjacency_matrix:
		for element in row:
			print(element, " ", end='')
		print("")
	print("travel between neighbors y = ", y)
	print("travel between neighbors and center x = ", x)
	print("surrounding conquer costs c = ", c)
	print("center conquer cost d = ", d)
	print("circle cost = ", circle_cost)
	print("center cost = ", center_cost)
	print("# non center kingdoms conquered = ", circle_neighbors_conquered if optimal_circle else 0)
	print("kingdoms_conquered = ", kingdoms_conquered)
	print("tour = ", tour)
	cost = circle_cost if optimal_circle == True else center_cost

	# uncomment for TESTING purposes, see if solver can find something better...OR see if tour is valid
	filename = "one_gadget_inputs\\"+str(n) + ".in" # files are put in a separate folder, to distinguish inputs made of a SINGLE gadget
	write_to_input(filename, n, range(n), start_index, adjacency_matrix)

	# store optimal solution
	# first force it to be a tour; since exit is connected to start, just append start to the end
	if optimal_circle == False:
		tour = [start_index, 0, start_index] # conquer middle, go back to start
	else:
		tour.append(start_index)

	filename = "one_gadget_outputs\\"+str(n) + ".out"
	print("writing to file: " + filename)
	f=open(filename,"w")

	for kingdom_i in tour:
		f.write(str(kingdom_i) + " "),
	
	f.write("\n")

	for kingdom_i in kingdoms_conquered:
		f.write(str(kingdom_i) + " ")
	f.close()

	# sanity check to make sure costs are correctly calculated
	check_cost = 0
	for i in kingdoms_conquered:
		check_cost += adjacency_matrix[i][i]
	for i in range(len(tour)-1):
		check_cost += adjacency_matrix[tour[i]][tour[i+1]]	
	print("check cost: ", check_cost) # sanity check
	
	return adjacency_matrix, kingdoms_conquered, tour, cost


def visualize(adjacency_matrix):
	G = nx.from_numpy_matrix(adjacency_matrix)
	# or nx.from_numpy_matrix
	# or from_scipy_sparse
	nx.draw(G, with_labels = True)
	plt.show()
	return


def solve_from_file(input_file, output_directory):
	print('Processing', input_file)
	basename, filename = os.path.split(input_file)

	try:
		input_data = utils.read_file(input_file)
		number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)
		closed_walk, conquered_kingdoms, cost = solver1(list_of_kingdom_names, starting_kingdom, adjacency_matrix)


		utils.write_to_file("karen_outputs\\costs.out", '\n', append=True)
		utils.write_to_file("karen_outputs\\costs.out", filename + " " + str(cost), append=True)
		output_filename = utils.input_to_output(filename)
		output_file = f'{output_directory}/{output_filename}'
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)
		utils.write_data_to_file(output_file, closed_walk, ' ')
		utils.write_to_file(output_file, '\n', append=True)
		utils.write_data_to_file(output_file, conquered_kingdoms, ' ', append=True)
	except Exception as e:
		utils.write_to_file("karen_outputs\\costs.out", '\n', append=True)
		utils.write_to_file("karen_outputs\\costs.out", filename + ": ERROR", append=True)

def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory)

solve_from_file("karen_inputs\\sub200.in", "karen_outputs")
# solve_all("karen_inputs", "karen_outputs")
# solver1("0", 0, "outputs") # the second parameter is the optimal cost seen so far
								 # the solver will only write to the output file if it beats
								 # this cost, in which case you should run the solver again
								 # with the updated new cost to beat