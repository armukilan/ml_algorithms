import networkx as nx
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches


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

	def visualize_path(self, path):
		"""Visualize the grid with path highlighted"""
		fig, ax = plt.subplots(figsize=(10, 8))
		
		# Create grid boxes
		for x in range(self.cols):
			for y in range(self.rows):
				# Create rectangle
				rect = patches.Rectangle(
					(x - 0.5, y - 0.5), 1, 1,
					linewidth=2, edgecolor='black', facecolor='white'
				)
				ax.add_patch(rect)
				
				# Add coordinate label
				plt.text(x, y, f"({x},{y})", 
						ha='center', va='center', fontsize=10)
		
		# Highlight path if one exists
		if path:
			# Draw connecting lines
			x_coords = [p[0] for p in path]
			y_coords = [p[1] for p in path]
			ax.plot(x_coords, y_coords, 'r-', linewidth=3, zorder=10)
			
			# Highlight path nodes
			for x, y in path:
				rect = patches.Rectangle(
					(x - 0.5, y - 0.5), 1, 1,
					linewidth=2, edgecolor='black', facecolor='red', alpha=0.7
				)
				ax.add_patch(rect)
				plt.text(x, y, f"({x},{y})", 
						ha='center', va='center', fontsize=10, color='white', weight='bold')
		
		# Mark start and end points
		start_rect = patches.Rectangle(
			(self.start[0] - 0.5, self.start[1] - 0.5), 1, 1,
			linewidth=3, edgecolor='green', facecolor='none'
		)
		end_rect = patches.Rectangle(
			(self.end[0] - 0.5, self.end[1] - 0.5), 1, 1,
			linewidth=3, edgecolor='blue', facecolor='none'
		)
		ax.add_patch(start_rect)
		ax.add_patch(end_rect)
		
		ax.set_xlim(-0.6, self.cols - 0.4)
		ax.set_ylim(-0.6, self.rows - 0.4)
		ax.set_aspect('equal')
		ax.axis('off')
		plt.title("A* Path Visualization", pad=20)
		plt.tight_layout()
		plt.show()




def main():
	no_rows = 3
	no_col = 4
	start = (0,2)
	end = (3,0)

	# no_rows = 4
	# no_col = 8
	# start = (0,3)
	# end = (7,0)

	path = astar(no_rows, no_col, start, end)

	# print("Visualizing graph...")
	# path.visualize_graph(show_weights=True)  # Set False to hide weights

	final_path = path.find_path()

	print(f"The path is {final_path}")
	path.visualize_path(final_path)


if __name__ == "__main__":
	main()