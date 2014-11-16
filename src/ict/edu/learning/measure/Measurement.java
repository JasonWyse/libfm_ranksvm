package ict.edu.learning.measure;

import java.util.ArrayList;
import java.util.List;

import ciir.umass.edu.learning.PartialPair;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.Sorter;

public class Measurement {

	public static double MAP(List<ArrayList<Double>> dll, List<RankList> rll){
		double map = 0;	
		double cumulative_ap = 0;
		double noPartialPairQueryNum = 0;
		for (int i = 0; i < rll.size(); i++) {//for each query			
			int [] docs_des_byScore = Sorter.sortDesc(dll.get(i));//calculate the permutation for documents of query i
			double p_i_allJ = 0;
			double p_i_j = 0;
			double relevantDocNum_i = 0;
			for (int j = 0; j < docs_des_byScore.length; j++) {//calculate each document's precision
				if(rll.get(i).get(docs_des_byScore[j]).getLabel()==0)
					continue;
				relevantDocNum_i++;
				double relevance =0;				
				for (int k = 0; k <=j; k++) {//iterate all the document k before document j
					int index = docs_des_byScore[k];					
					if (rll.get(i).get(index).getLabel() > 0) {
						relevance ++;
					}					
				}				
				p_i_j = (relevance/(j+1));
				p_i_allJ += p_i_j;
			}
			if(p_i_allJ==0){//which means there is no one relevant document in query i, all the label value is zeros.
				noPartialPairQueryNum++;
			}
			else{
				cumulative_ap += p_i_allJ/relevantDocNum_i;
			}			
			
		}
		map = cumulative_ap/(rll.size()-noPartialPairQueryNum);		
//		map = cumulative_ap/(rll.size());
		return map;		
	}
	public static List<Double> getRelevantDegreeForOneQuery(RankList rl)//rl holds all documents for one query 
	{
		List<Double> label_list = new ArrayList<Double>();
		for (int i = 0; i < rl.size(); i++) {
			label_list.add((double) rl.get(i).getLabel());
		}
		return label_list;
		
	}
	public static double NDCG(List<ArrayList<Double>> dll, List<RankList> rll, int N){
		double dcg_all_i = 0;
		double idcg_all_i = 0;
		double ndcg_all_i = 0;
		double noPartialPairQueryNum = 0;
		for (int i = 0; i < rll.size(); i++) {//iterate each query
			double dcg_i = 0;
			double idcg_i = 0;
			double ndcg_i = 0;
			int [] docs_des_byScore = Sorter.sortDesc(dll.get(i));//calculate the permutation for documents of query i
			List<Double>  label_list = getRelevantDegreeForOneQuery(rll.get(i));
			int [] idcg_array = Sorter.sortDesc(label_list);
			for (int j = 0; j < rll.get(i).size(); j++) {
				if(j==N)
					break;
				float relevant_degree = rll.get(i).get(docs_des_byScore[j]).getLabel();
				dcg_i += (Math.pow(2, relevant_degree)-1)/(Math.log(2+j)/Math.log(2));
				
				double relevant_degree2 = label_list.get(idcg_array[j]);
				idcg_i += (Math.pow(2, relevant_degree2)-1)/(Math.log(1+j+1)/Math.log(2));
			}
			if(idcg_i==0){
				noPartialPairQueryNum++;
				continue;
			}
				
			dcg_all_i += dcg_i;
			idcg_all_i += idcg_i;
			ndcg_i = dcg_all_i/idcg_all_i;
			ndcg_all_i += ndcg_i;
		}
//		ndcg_all_i = ndcg_all_i/rll.size();
		ndcg_all_i = ndcg_all_i/(rll.size()-noPartialPairQueryNum);
		return ndcg_all_i;
	}
	public static double precision(List<ArrayList<Double>> dll, List<RankList> rll, int N){
		double p_all_i = 0;
		for (int i = 0; i < rll.size(); i++) {
			double p_i = 0;
			int relevantDocsNum_i = 0;
			int [] docs_des_byScore = Sorter.sortDesc(dll.get(i));
			for (int j = 0; j < rll.get(i).size(); j++) {
				if(j==N)
					break;
				float relevant_degree = rll.get(i).get(docs_des_byScore[j]).getLabel();
				if(relevant_degree > 0)
					relevantDocsNum_i++;
			}
			p_i = relevantDocsNum_i/N;
			p_all_i = p_i/rll.size();
		}
		return p_all_i;
	}
}
