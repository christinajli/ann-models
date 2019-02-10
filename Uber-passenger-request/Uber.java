/*
 * Name: Christina Li
 * NetID: 15jl119
 * Project: Uber
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays; //Use when want to print out and check data
import java.util.Collections;
import java.util.ArrayList;
import java.util.Scanner;

public class Uber {
	public static ArrayList<ArrayList<Integer>> readData(String name) {
		String fileName = name;
		File tempFile = new File(fileName);
		String tempString = "";

		Scanner scannerInput;
		try {
			scannerInput = new Scanner(tempFile);
			while (scannerInput.hasNext()) {
				String buff = scannerInput.next();
				tempString = tempString + buff + "\n";
			}
			scannerInput.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// convert string to 2D array
		String[] rows = tempString.split("\n");
		int rowNum = (rows.length);
		String[][] tempData = new String[rowNum][];
		for (int i = 0; i < rowNum; i++) {
			tempData[i] = rows[i].split(",");
		}

		int colNum = (tempData[0].length);
		
		ArrayList<ArrayList<Integer>> someData = new ArrayList<ArrayList<Integer>>();
		for (int i = 0; i < rowNum; i++) {
			ArrayList<Integer> row = new ArrayList<Integer>();
			for (int j = 0; j < colNum; j++) {
				row.add(Integer.parseInt(tempData[i][j]));
			}
			someData.add(row);
		}
		//System.out.println(Arrays.toString(someData.toArray()).replace("], ", "]\n").replace("[[", "[").replace("]]", "]"));
		return someData;
	}// end read data
	
	
	public static void compute_waitTime(ArrayList<ArrayList<Integer>> shortestTimeData) {
		int time = 0;
		int total_waitTime = 0;
		int numberofDrivers = 2; ////CHANGE NUMBER OF DRIVERS HERE///////
		
		// Define states for the classes
		String[] passenger_state = {"noRequest","waiting","onthego","done"};
		String[] driver_state = {"free","driving"};
		
		// Initialize drivers
		Driver[] drivers = new Driver[numberofDrivers];
		for (int i = 0; i<numberofDrivers;i++) {
			drivers[i] = new Driver(0,0,0,null,driver_state[0]);
		}
		
		//Initialize passengers
		ArrayList<ArrayList<Integer>> requestsData = readData("requests.csv"); /////CHANGE DATA FILE HERE////////
		//System.out.println(Arrays.toString(requestsData.toArray()).replace("], ", "]\n").replace("[[", "[").replace("]]", "]"));
		Passenger[] passengers = new Passenger[requestsData.size()];
		for (int i = 0; i < requestsData.size();i++) {
			passengers[i] = new Passenger((int)requestsData.get(i).get(0),(int)requestsData.get(i).get(1),
						(int)requestsData.get(i).get(2),passenger_state[0]);
		}

		int flag = 1;
		// While all passengers are not at done state
		while (flag == 1) {
			time++; // increment universe time each loop
			int allDone = 0;
			for (int i = 0; i < passengers.length; i++) {
				if (passengers[i].getState() != "done") {
					allDone++;
					//System.out.println("Done passenger # "+ (i-1));
					break;
				}
			}

			// For all drivers in busy state
			for (int j = 0; j < numberofDrivers; j++) {
				if (drivers[j].getState() == "driving") {
					if (drivers[j].getAssigned().getState() == "waiting") {
						total_waitTime++; // increment total time per number of passenger waiting
					}
					drivers[j].move(shortestTimeData);
				}
			} 
			
			// For all passengers with state no request
			for (int i = 0; i < passengers.length; i++) {
				if (passengers[i].getState() == "noRequest") {
					if (time >= passengers[i].getRequest_time()) {
						// total wait time is universe time minus the time it has been waiting since requested
						total_waitTime += time-passengers[i].getRequest_time();
						passengers[i].setState("waiting");
						
						// Assign driver to passenger based on shortest time
						// compare and if no free use booked to find next
						ArrayList<Integer> compareShortest = new ArrayList<Integer>();
						for (int j = 0; j < numberofDrivers; j++) {
							//-1 because passenger request nodes go from 1 - 50 where shortest time data goes from 0 - 49
							int nextTime = shortestTimeData.get((drivers[j].getStartNode())).get((passengers[i].getStart_node()) - 1) 
									+ drivers[j].getTimeToPassenger()
									+ drivers[j].getTimeToDestination();
							compareShortest.add(nextTime);
						}
						int minIndex = compareShortest.indexOf(Collections.min(compareShortest));
						
						if (drivers[minIndex].getState() == "free") {
							drivers[minIndex].setAssigned(passengers[i]);
							drivers[minIndex].setState("driving");
							drivers[minIndex].setTimeToPassenger(shortestTimeData.get((passengers[i].getStart_node()) - 1).get((passengers[i].getEnd_node()) - 1));
						} else if (drivers[minIndex].getState() == "driving") {
							// reset passenger state because did not assign a driver yet
							drivers[minIndex].updatePassenger(passengers[i],"noRequest");
						} 
					} 
				} 
			} 
			
			if (allDone == 0) {
				flag = 0; // exit while true
			}
			
			// For detailed information for each time interval
			//System.out.println("Time " + time);
			//System.out.println("Driver 0: "+drivers[0].getState() + " " + drivers[0].getTimeToPassenger() + " " + drivers[0].getTimeToDestination());
			//System.out.println("Driver 1: "+drivers[1].getState() + " " + drivers[1].getTimeToPassenger() + " " + drivers[1].getTimeToDestination());
			//System.out.println("Current total wait time: " + total_waitTime);
		} 
		
		System.out.println("Overall total wait time: " + total_waitTime);
	}

	public static void main(String[] arg) {
		// Add the required data
		ArrayList<ArrayList<Integer>> networkData = readData("network.csv");
		ArrayList<ArrayList<Integer>> shortestTimeData = new ArrayList<ArrayList<Integer>>();
		
		Dijkstra obj = new Dijkstra();
		// Calculate and store Dijkstra shortest time between any two nodes
		for (int k = 0; k < networkData.size(); k++) {
			// Create a new graph
			Graph g = new Graph(networkData.size());
			for (int i = 0; i < networkData.size(); i++) {
				for (int j = 0; j < networkData.get(i).size(); j++) {
					if (networkData.get(i).get(j) != 0) {
						g.addEdge(i, j, networkData.get(i).get(j));
					}
				}
			}
			obj.calculate(g.getNode(k));

			ArrayList<Integer> row = new ArrayList<Integer>();
			for (int j = 0; j < g.getNodes().size(); j++) {
				row.add((int) g.getNodes().get(j).minTime);
			}
			shortestTimeData.add(row);
		}
		System.out.println(Arrays.toString(shortestTimeData.toArray()).replace("], ", "]\n").replace("[[", "[").replace("]]", "]"));

		// Call compute wait times
		compute_waitTime(shortestTimeData);
	}
}
