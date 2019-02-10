import java.util.ArrayList;

public class Graph {
	private ArrayList<Node> nodes;

	public Graph(int numberNodes) {
		nodes = new ArrayList<Node>(numberNodes);
		for (int i = 0; i < numberNodes; i++) {
			nodes.add(new Node("n" + Integer.toString(i)));
		}
	}

	public void addEdge(int src, int dest, int weight) {
		Node s = nodes.get(src);
		Edge new_edge = new Edge(nodes.get(dest), weight);
		s.neighbours.add(new_edge);
	}

	public ArrayList<Node> getNodes() {
		return nodes;
	}

	public Node getNode(int node) {
		return nodes.get(node);
	}
}
