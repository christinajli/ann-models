import java.util.ArrayList;

public class Node implements Comparable<Node> {
	public final String name;
	public ArrayList<Edge> neighbours;
	public double minTime = Double.POSITIVE_INFINITY;
	//public Node previous;

	public int compareTo(Node other) {
		return Double.compare(minTime, other.minTime);
	}
	public Node(String name) {
		this.name = name;
		neighbours = new ArrayList<Edge>();
	}
	public String toString() {
		return name;
	}
}
