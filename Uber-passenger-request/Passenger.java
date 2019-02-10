/*
 * Name: Christina Li
 * NetID: 15jl119
 * Project: Uber
 */
public class Passenger {
	private int request_time;
	private int start_node;
	private int end_node;
	private String state;
	
	//constructor
	Passenger(){
		this.request_time = 0;
		this.start_node = 0;
		this.end_node = 0;
		this.state = "";
	}
     Passenger(int requestTime, int startNode, int endNode, String startstate) {
		this.request_time = requestTime;
		this.start_node = startNode;
		this.end_node = endNode;
		this.state = startstate;
	}
	//getters
	public int getRequest_time() {
		return request_time;
	}
	public int getStart_node() {
		return start_node;
	}
	public int getEnd_node() {
		return end_node;
	}
	public String getState() {
		return state;
	}
	//setters
	public void setRequest_time(int request) {
		this.request_time = request;
	}
	public void setStart_node(int start) {
		 this.start_node = start;
	}
	public void setEnd_node(int end) {
		 this.end_node = end;
	}
	public void setState(String newstate) {
		this.state = newstate;
	}
}
