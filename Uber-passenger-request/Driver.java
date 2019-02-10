/*
 * Name: Christina Li
 * NetID: 15jl119
 * Project: Uber
 */
import java.util.ArrayList;

public class Driver {
	private int start_node;
	private int time_to_passenger;
	private int time_to_destination;
	// need reference to passenger to change passenger state
	private Passenger assigned;
	private String state;

	// constructor
	public Driver(int currentNode, int timeToPassenger, int timeToDestination, Passenger myPassenger,
			String currentstate) {
		this.start_node = currentNode;
		this.time_to_passenger = timeToPassenger;
		this.time_to_destination = timeToDestination;
		this.assigned = myPassenger;
		this.state = currentstate;
	}

	// getters
	public int getStartNode() {
		return start_node;
	}

	public int getTimeToPassenger() {
		return time_to_passenger;
	}

	public int getTimeToDestination() {
		return time_to_destination;
	}

	public Passenger getAssigned() {
		return assigned;
	}

	public String getState() {
		return state;
	}

	// setters
	public void setStartNode(int newNode) {
		this.start_node = newNode;
	}

	public void setTimeToPassenger(int newPassenger) {
		this.time_to_passenger = newPassenger;
	}

	public void setTimeToDestination(int newEndNode) {
		this.time_to_destination = newEndNode;
	}

	public void setState(String newstate) {
		this.state = newstate;
	}

	public void setAssigned(Passenger assigned) {
		this.assigned = assigned;
	}

	public void updatePassenger(Passenger aPassenger, String newState) {
		aPassenger.setState(newState);
	}

	public void move(ArrayList<ArrayList<Integer>> shortestTimeData) {
		// on the way to pick up passenger
		if (this.getTimeToPassenger() != 0) {
			this.setTimeToPassenger(this.getTimeToPassenger() - 1);
		}
		// picked up passenger, heading to destination
		if (this.getTimeToPassenger() == 0 && this.getAssigned().getState() == "waiting") {
			this.updatePassenger(this.getAssigned(), "onthego");
			this.setTimeToDestination(shortestTimeData.get((this.getAssigned().getStart_node()) - 1)
										.get((this.getAssigned().getEnd_node()) - 1));
			if (this.getAssigned().getEnd_node() == 50) {
				this.setStartNode(this.getAssigned().getEnd_node() - 1); // prevent array index out of bound
			} else {
				this.setStartNode(this.getAssigned().getEnd_node()); // set next start node
			}
		}
		// with passenger on the way to drop off passenger
		if (this.getTimeToDestination() != 0) {
			this.setTimeToDestination(this.getTimeToDestination() - 1);
		}
		// dropped off passenger
		if (this.getTimeToDestination() == 0 && this.getAssigned().getState() == "onthego") {
			// finished driving passenger to destination
			this.updatePassenger(this.getAssigned(), "done");
			this.setTimeToPassenger(0);
			this.setTimeToDestination(0);

			if (this.getState() == "driving") {
				this.setState("free");
			} else if (this.getState() == "booked") {
				this.setTimeToPassenger(shortestTimeData.get((this.getAssigned().getStart_node()) - 1)
										.get((this.getAssigned().getEnd_node()) - 1));
				this.setState("driving");
			}
		}
	}

}
