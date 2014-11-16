package ict.edu.learning.multiThread;


import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.PartialPairList;

public class ThreadCalculateObj_Jfun implements Callable {

	private List<PartialPairList> ppll;	
	HashMap<String, Integer> hp;
	Matrix V;
	int q_index;
	public ThreadCalculateObj_Jfun(List<PartialPairList> ppll, int q_index, Matrix V, HashMap<String, Integer> hp){
//		this.rll = rll;
		this.hp = hp;
		this.ppll = ppll;
		this.V = V;
		this.q_index = q_index;
	}
	public double execute(){
		double J_value = 0;
		for(int i = 0; i < ppll.get(q_index).size(); i++){
			double index_E = 0f;// we can calculate index_E parallelly , need to accomplish
			for (int k = 0; k < ppll.size(); k++) {
				for (int l = 0; l < ppll.get(k).size(); l++) {
					//for a given partialPair X_ijq=ppll.get(i).get(j), we need to compute the  
					/*if(ppll.get(k).size()==0)
						continue;*/
					String queryID = ppll.get(k).get(l).getQueryID();
					String largeDocID = ppll.get(k).get(l).getLargeDocID();
					String smallDocID = ppll.get(k).get(l).getSmallDocID();
					int V_iq = hp.get(queryID+"-"+largeDocID).intValue();
					int V_jq = hp.get(queryID+"-"+smallDocID).intValue();
					double innerProduct_V = V.getInnerProduct(V_iq, V_jq);
					double innerProduct_partialPair = ppll.get(q_index).get(i).dotProduct(ppll.get(k).get(l));
					index_E += innerProduct_V * innerProduct_partialPair;
				}
			}
			if(index_E>10)
				J_value += 0;
			else if(index_E<-10)
				J_value += (-index_E);
			else					
				J_value += Math.log(1+Math.exp(-index_E));
		}
		System.out.println("query "+q_index +" is over");
		return J_value;
		
	}
	
	@Override
	public Object call() throws Exception {
		// TODO Auto-generated method stub
		double subJ_value = execute();
		return subJ_value;
	}

}
