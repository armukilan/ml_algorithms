import networkx as nx
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches


class astar:
	def __init__(self, _row, _col, _start, _end, _obstacle_mat):
		self.graph = nx.Graph()
		self.rows = _row
		self.cols = _col
		self.start = _start
		self.end = _end
		self.obstacle_mat = _obstacle_mat
		self._build_graph()

	def _build_graph(self):
		# NetworkX graph with movement in all 8 directions
		directions = [
			(0, 1), (1, 0), (0, -1), (-1, 0),   # Cardinal
			(1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
		]
		
		for x in range(self.cols):
			for y in range(self.rows):
				# Skip if current cell is blocked
				if self.obstacle_mat[y][x] == 0:
					continue
					
				for dx, dy in directions:
					nx, ny = x + dx, y + dy
					if (0 <= nx < self.cols and 0 <= ny < self.rows and 
						self.obstacle_mat[ny][nx] == 1):  # Check if neighbor is free
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


	def visualize_path(self, path, obstacle_mat=None):
		"""Visualize the grid with path highlighted and obstacles marked"""
		fig, ax = plt.subplots(figsize=(10, 8))
		
		# Create grid boxes
		for x in range(self.cols):
			for y in range(self.rows):
				# Determine cell color (black for obstacles)
				if self.obstacle_mat and self.obstacle_mat[y][x] == 0:
					facecolor = 'black'
					textcolor = 'white'
				else:
					facecolor = 'white'
					textcolor = 'black'
				
				# Create rectangle
				rect = patches.Rectangle(
					(x - 0.5, y - 0.5), 1, 1,
					linewidth=2, edgecolor='black', facecolor=facecolor
				)
				ax.add_patch(rect)
				
				# Add coordinate label
				plt.text(x, y, f"({x},{y})", 
						ha='center', va='center', fontsize=10, color=textcolor)
		
		# Highlight path if one exists
		if path:
			# Draw connecting lines (only between non-obstacle nodes)
			valid_segments = []
			current_segment = []
			
			for x, y in path:
				if not obstacle_mat or obstacle_mat[y][x] != 0:
					current_segment.append((x, y))
				else:
					if current_segment:
						valid_segments.append(current_segment)
						current_segment = []
			
			if current_segment:
				valid_segments.append(current_segment)
				
			for segment in valid_segments:
				x_coords = [p[0] for p in segment]
				y_coords = [p[1] for p in segment]
				ax.plot(x_coords, y_coords, 'r-', linewidth=3, zorder=10)
			
			# Highlight path nodes (only non-obstacle ones)
			for x, y in path:
				if not obstacle_mat or obstacle_mat[y][x] != 0:
					rect = patches.Rectangle(
						(x - 0.5, y - 0.5), 1, 1,
						linewidth=2, edgecolor='black', facecolor='red', alpha=0.7
					)
					ax.add_patch(rect)
					plt.text(x, y, f"({x},{y})", 
							ha='center', va='center', fontsize=10, color='white', weight='bold')
		
		# Mark start and end points (if they're not obstacles)
		if not obstacle_mat or obstacle_mat[self.start[1]][self.start[0]] != 0:
			start_rect = patches.Rectangle(
				(self.start[0] - 0.5, self.start[1] - 0.5), 1, 1,
				linewidth=3, edgecolor='green', facecolor='none'
			)
			ax.add_patch(start_rect)
		
		if not obstacle_mat or obstacle_mat[self.end[1]][self.end[0]] != 0:
			end_rect = patches.Rectangle(
				(self.end[0] - 0.5, self.end[1] - 0.5), 1, 1,
				linewidth=3, edgecolor='blue', facecolor='none'
			)
			ax.add_patch(end_rect)
		
		ax.set_xlim(-0.6, self.cols - 0.4)
		ax.set_ylim(-0.6, self.rows - 0.4)
		ax.set_aspect('equal')
		ax.axis('off')
		plt.title("A* Path Visualization with Obstacles", pad=20)
		plt.tight_layout()
		plt.show()




def main():
	no_rows = 3
	no_col = 4
	start = (0,2)
	end = (3,0)
	mat = [[1,1,1,1], [1,0,0,1],[1,1,1,1]]

	# no_rows = 4
	# no_col = 8
	# start = (0,3)
	# end = (7,0)

	path = astar(no_rows, no_col, start, end, mat)

	# print("Visualizing graph...")
	# path.visualize_graph(show_weights=True)  # Set False to hide weights
	# exit()

	final_path = path.find_path()

	print(f"The path is {final_path}")
	path.visualize_path(final_path)


if __name__ == "__main__":
	main()