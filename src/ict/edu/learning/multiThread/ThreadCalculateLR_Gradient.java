package ict.edu.learning.multiThread;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.Vector;

public class ThreadCalculateLR_Gradient implements Callable {

	private List<PartialPairList> ppll;	
	HashMap<String, Integer> hp;
	Matrix V;
	int q_index;
	Vector w;
	public ThreadCalculateLR_Gradient(List<PartialPairList> ppll, int q_index, Vector w){
//		this.rll = rll;		
		this.ppll = ppll;		
		this.q_index = q_index;
		this.w = w;
	}
	public Vector execute(){
		double total = 0;
		Vector gradient = new Vector();
		for(int j = 0; j < ppll.get(q_index).size(); j++){
			Vector x_ijq = new Vector(ppll.get(q_index).get(j).getPartialFVals());
			double index_E = Vector.dotProduct(w,x_ijq);
			double coefficient = 1/(1+Math.exp(index_E));
			Vector v=Vector.multiply(-coefficient, x_ijq);
			gradient = Vector.addition(gradient,v);
		}		
		return gradient;		
	}
	
	@Override
	public Object call() throws Exception {
		// TODO Auto-generated method stub
		Vector w = execute();
		return w;
	}

}
