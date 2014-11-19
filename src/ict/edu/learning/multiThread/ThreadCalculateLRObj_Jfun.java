package ict.edu.learning.multiThread;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.Vector;

public class ThreadCalculateLRObj_Jfun implements Callable {

	private List<PartialPairList> ppll;	
	HashMap<String, Integer> hp;
	Matrix V;
	int q_index;
	Vector w;
	public ThreadCalculateLRObj_Jfun(List<PartialPairList> ppll, int q_index, Vector w){
//		this.rll = rll;		
		this.ppll = ppll;		
		this.q_index = q_index;
		this.w = w;
	}
	public double execute(){
		double total = 0;
		for(int j = 0; j < ppll.get(q_index).size(); j++){
			double[] vals = ppll.get(q_index).get(j).getPartialFVals();
			Vector x_ijq= new Vector(vals);
			double index_E = Vector.dotProduct(w,x_ijq);
			if (index_E>=20) {
				total +=0;
			}
			else if(index_E<=-20){
				total += (-index_E);
			}
			else{
				total += Math.log(1+Math.exp(-index_E));
			}
		}		
		return total;		
	}
	
	@Override
	public Object call() throws Exception {
		// TODO Auto-generated method stub
		double subJ_value = execute();
		return subJ_value;
	}

}