import networkx as nx
import matplotlib.pyplot as plt
import math


class astar:
	def __init__(self, _row, _col, _start,_end):
		self.graph = nx.Graph()
		self.rows = _row
		self.cols = _col
		self.start = _start
		self.end = _end
		self._build_graph()

	def _build_graph(self):
		# NetworkX graph with movement in all 8 directions
		directions = [
			(0, 1),   # Right
			(1, 0),    # Down
			(0, -1),   # Left
			(-1, 0),   # Up
			(1, 1),    # Down-Right
			(1, -1),   # Down-Left
			(-1, 1),   # Up-Right
			(-1, -1)   # Up-Left
		]
		
		for x in range(self.cols):
			for y in range(self.rows):
				for dx, dy in directions:
					nx, ny = x + dx, y + dy
					if 0 <= nx < self.cols and 0 <= ny < self.rows:
						weight = 1 if abs(dx) + abs(dy) == 1 else 1.414
						self.graph.add_edge((x, y), (nx, ny), weight=weight)


	def visualize_graph(self, show_weights=False):
		# Visualize the grid graph with optional edge weights
		plt.figure(figsize=(8, 6))
		
		# Position nodes in grid layout
		pos = {(x, y): (x, -y) for x in range(self.cols) for y in range(self.rows)}
		
		# Add nodes, edges and labels
		nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color='lightblue')
		nx.draw_networkx_edges(self.graph, pos, edge_color='gray', width=1.5)
		nx.draw_networkx_labels(self.graph, pos, font_size=12)
		
		if show_weights:
			edge_labels = nx.get_edge_attributes(self.graph, 'weight')
			rounded_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
			nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=rounded_labels)
		
		plt.title("Grid method")
		plt.grid(True)
		plt.axis('equal')
		plt.show()

	def find_path(self):
		def heuristic(u, v):
			dx = abs(u[0] - v[0])
			dy = abs(u[1] - v[1])
			return math.sqrt(dx*dx + dy*dy)
			
		# try:
		return nx.astar_path(self.graph, self.start, self.end, heuristic=heuristic)
		# except nx.NetworkXNoPath:
		#     return None




def main():
	# no_rows = 3
	# no_col = 4
	# start = (0,2)
	# end = (3,0)

	no_rows = 4
	no_col = 8
	start = (0,3)
	end = (7,0)

	path = astar(no_rows, no_col, start, end)

	# print("Visualizing graph...")
	# path.visualize_graph(show_weights=True)  # Set False to hide weights

	print(f"the path is {path.find_path()}")


if __name__ == "__main__":
	main()