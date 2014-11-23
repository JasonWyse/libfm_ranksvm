package ict.edu.learning.multiThread;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.PartialPairList;

public class ThreadCalculate_PartsPartialPairsInOneQuery_Obj implements
		Callable {

	private List<PartialPairList> ppll;
	HashMap<String, Integer> hp;
	Matrix V;
	int q_index;
	int cpu_index;
//	int nThread;
	int remaining_pp;
	int each_CPU_load;
	public ThreadCalculate_PartsPartialPairsInOneQuery_Obj(
			List<PartialPairList> ppll, int q_index, Matrix V,
			HashMap<String, Integer> hp, int cpu_index, int each_CPU_load,
			int remaining_pp) {
		// this.rll = rll;
		this.hp = hp;
		this.ppll = ppll;
		this.V = V;
		this.q_index = q_index;
		this.cpu_index = cpu_index;
//		this.nThread = nThread;
		this.each_CPU_load = each_CPU_load;
		this.remaining_pp = remaining_pp;
	}

	public double execute() {
		double J_value = 0;
		int start_position = cpu_index * each_CPU_load;
		int numberOfPartialPair =  each_CPU_load + remaining_pp;
		if (cpu_index<0) {
			start_position = 0;
			numberOfPartialPair = remaining_pp;
		}
		for (int i = start_position; i < (start_position + numberOfPartialPair); i++) {
			double index_E = 0f;// we can calculate index_E parallelly , need to
								// accomplish
			for (int k = 0; k < ppll.size(); k++) {
				for (int l = 0; l < ppll.get(k).size(); l++) {
					// for a given partialPair X_ijq=ppll.get(i).get(j), we need
					// to compute the
					/*
					 * if(ppll.get(k).size()==0) continue;
					 */
					String queryID = ppll.get(k).get(l).getQueryID();
					String largeDocID = ppll.get(k).get(l).getLargeDocID();
					String smallDocID = ppll.get(k).get(l).getSmallDocID();
					int V_iq = hp.get(queryID + "-" + largeDocID).intValue();
					int V_jq = hp.get(queryID + "-" + smallDocID).intValue();
					double innerProduct_V = V.getInnerProduct(V_iq, V_jq);
					double innerProduct_partialPair = ppll.get(q_index).get(i)
							.dotProduct(ppll.get(k).get(l));
					index_E += innerProduct_V * innerProduct_partialPair;
				}
			}
			if (index_E > 10)
				J_value += 0;
			else if (index_E < -10)
				J_value += (-index_E);
			else
				J_value += Math.log(1 + Math.exp(-index_E));
		}
		return J_value;

	}

	@Override
	public Object call() throws Exception {
		// TODO Auto-generated method stub
		double subJ_value = execute();
		return subJ_value;
	}

}
