import java.util.PriorityQueue;

public class Dijkstra {

	public void calculate(Node source) {
		// take the unvisited node with minimum weight.
		// visit all its neighbours.
		// update the distances for all the neighbours
		// repeat the process till all the connected nodes are visited.

		source.minTime = 0;
		PriorityQueue<Node> queue = new PriorityQueue<Node>();
		queue.add(source);

		while (!queue.isEmpty()) {

			Node u = queue.poll();

			for (Edge neighbour : u.neighbours) {
				Double newDist = u.minTime + neighbour.weight;

				if (neighbour.target.minTime > newDist) {
					// Remove the node from the queue to update the distance value.
					queue.remove(neighbour.target);
					neighbour.target.minTime = newDist;

					// Reenter the node with new distance.
					queue.add(neighbour.target);
				}
			}
		}
	}

}
