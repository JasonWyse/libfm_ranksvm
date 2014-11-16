package ict.edu.learning.multiThread;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.PartialPair;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.RankList;

public class ThreadUpdateVMatrix implements Callable{

	PartialPair pp;
	int query_index;
	List<RankList> rll;
	List<PartialPairList> ppll;
	HashMap<String, Integer> hp;
	Matrix V_old;
	int V_ac;
	double factor1;
//	Matrix V_new;
	double learningRate;
	public ThreadUpdateVMatrix(double factor1,int query_index, PartialPair pp, int V_ac,List<PartialPairList> ppll,HashMap<String, Integer> hp,Matrix V_old, double learningRate){
		this.pp = pp;
	//	this.rll = rll;
		this.ppll = ppll;
		this.hp = hp;
		this.V_old = V_old;
		this.query_index = query_index;
		this.V_ac = V_ac;
		this.factor1 = factor1;
//		this.V_new = V_new;
		this.learningRate = learningRate;
	}
	
	public double[] execute(){
//		System.out.println(V_ac + "row is going to be updated");
		Matrix V_new = new Matrix(V_old);
		double [] factor2 = new double[Matrix.ColsOfVMatrix];	
		
		for (int j2 = 0; j2 < ppll.get(query_index).size(); j2++) {
			double [] temp = new double[Matrix.ColsOfVMatrix];
			
			PartialPair ite_pp = ppll.get(query_index).get(j2);
			String qid_largeDoc = ite_pp.getQueryID() + "-" + ite_pp.getLargeDocID();
			String qid_smallDoc = ite_pp.getQueryID() + "-" + ite_pp.getSmallDocID();
			if (V_ac == hp.get(qid_largeDoc)) {
				int docID_associatedWithV_ac = hp.get(qid_smallDoc);
				double multiplier =pp.dotProduct(ite_pp);
				// parameter factor2, stores the result of multiplication
				V_old.multiplyRowVector(docID_associatedWithV_ac, multiplier, temp);
				Matrix.RowVectorAddition(factor2, temp);				
				
			}
			else if(V_ac == hp.get(qid_smallDoc)){
				int docID_associatedWithV_ac = hp.get(qid_largeDoc);
				double multiplier = pp.dotProduct(ite_pp);
				V_old.multiplyRowVector(docID_associatedWithV_ac, multiplier, temp);
				Matrix.RowVectorAddition(factor2, temp);				
			}
		}
		double[] gradient = Matrix.multiplyRowVector(-factor1, factor2);
		Matrix.RowVectorAddition(V_new.getV()[V_ac], Matrix.multiplyRowVector(-learningRate, gradient));//negative direction of the gradient
//		System.out.println(V_ac + " row of matrix V has been updated according to the negative direction of the gradient");
		return V_new.getV()[V_ac];
	}
	

	@Override
	public Object call() throws Exception {
		// TODO Auto-generated method stub
		double[] m = execute();
		return m;
	}

}
