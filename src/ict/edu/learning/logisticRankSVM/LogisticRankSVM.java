package ict.edu.learning.logisticRankSVM;

import ict.edu.learning.metric.ResultClass;
import ict.edu.learning.multiThread.ThreadCalculateObj_Jfun;
import ict.edu.learning.multiThread.ThreadCalculate_PartsPartialPairsInOneQuery_Obj;
import ict.edu.learning.multiThread.ThreadUpdateVMatrix;
import ict.edu.learning.utilities.FileUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ciir.umass.edu.features.FeatureManager;
import ciir.umass.edu.features.Normalizer;
import ciir.umass.edu.features.SumNormalizor;
import ciir.umass.edu.features.ZScoreNormalizor;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.PartialPair;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.RANKER_TYPE;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.Vector;
import ciir.umass.edu.metric.ERRScorer;

public class LogisticRankSVM extends Ranker {

	/**
	 * @param args
	 */
	public static boolean letor = false;
	public static boolean mustHaveRelDoc = false;
	public static boolean normalize = false;
	public static Normalizer nml = new SumNormalizor();
	public static int partialPairTotalNum = 100;
	public static int RowsOfVMatrix = 100;
	public static int ColsOfVMatrix = 5;
	public static int ROW_INCREASE = 20;
	public static int V_size = 0;
	public static double epsilon = 0.00000000001f;
	public static int nThread = 16;
	public static double writeMatrixVInterval = 2;
	public static double learningRate = 0.005;
	public static double maxIterations = 500;
	public static int learningRateAttenuationTime = 5;
	public static int NDCG_para = 10;
	public static String allFile_prefix ="";
	public static HashMap<String, Integer> hp_V = null;
	static String fold_n = null;
	public static double output_interval = 3;

	public static void main(String[] args) throws InterruptedException,
			Exception {
		// TODO Auto-generated method stub
		String[] rType = new String[] { "MART", "RankNet", "RankBoost",
				"AdaRank", "Coordinate Ascent", "LambdaRank", "LambdaMART",
				"ListNet", "Random Forests", "Logistic RanKSVM" };
		RANKER_TYPE[] rType2 = new RANKER_TYPE[] { RANKER_TYPE.MART,
				RANKER_TYPE.RANKNET, RANKER_TYPE.RANKBOOST,
				RANKER_TYPE.ADARANK, RANKER_TYPE.COOR_ASCENT,
				RANKER_TYPE.LAMBDARANK, RANKER_TYPE.LAMBDAMART,
				RANKER_TYPE.LISTNET, RANKER_TYPE.RANDOM_FOREST,
				RANKER_TYPE.LOGISTIC_RANKSVM };

		String trainFile = "";
		String featureDescriptionFile = "";
		double ttSplit = 0.0;// train-test split
		double tvSplit = 0.0;// train-validation split
		int foldCV = -1;
		String validationFile = "";
		String testFile = "";
		int rankerType = 10;// our own logistic ranksvm
		String trainMetric = "ERR@10";
		String testMetric = "";

		String savedModelFile = "";
		String rankFile = "";
		boolean printIndividual = false;

		// for my personal use
		String indriRankingFile = "";
		String scoreFile = "";
		if (args.length < 2) {

			System.out.println("not enough parameter");
			return;
		}

		for (int i = 0; i < args.length; i++) {
			if (args[i].compareTo("-train") == 0)
				trainFile = args[++i];
			else if (args[i].compareTo("-ranker") == 0)
				rankerType = Integer.parseInt(args[++i]);
			else if (args[i].compareTo("-feature") == 0)
				featureDescriptionFile = args[++i];
			else if (args[i].compareTo("-metric2t") == 0)
				trainMetric = args[++i];
			else if (args[i].compareTo("-metric2T") == 0)
				testMetric = args[++i];
			else if (args[i].compareTo("-nThread") == 0)
				nThread = Integer.parseInt(args[++i]);
			else if (args[i].compareTo("-learningRate") == 0)
				learningRate = Double.parseDouble(args[++i]);
			else if (args[i].compareTo("-maxIterations") == 0)
				maxIterations = Double.parseDouble(args[++i]);
			else if (args[i].compareTo("-writeMatrixVInterval") == 0)
				writeMatrixVInterval = Double.parseDouble(args[++i]);
			else if (args[i].compareTo("-output_interval") == 0)
				output_interval = Double.parseDouble(args[++i]);
			
			else if (args[i].compareTo("-epsilon") == 0)
				epsilon = Double.parseDouble(args[++i]);
			else if (args[i].compareTo("-gmax") == 0)
				ERRScorer.MAX = Math.pow(2, Double.parseDouble(args[++i]));
			else if (args[i].compareTo("-tts") == 0)
				ttSplit = Double.parseDouble(args[++i]);
			else if (args[i].compareTo("-tvs") == 0)
				tvSplit = Double.parseDouble(args[++i]);
			else if (args[i].compareTo("-allFile_prefix") == 0)
				allFile_prefix = (args[++i] + "-");
			else if (args[i].compareTo("-kcv") == 0)
				foldCV = Integer.parseInt(args[++i]);
			else if (args[i].compareTo("-validate") == 0)
				validationFile = args[++i];
			else if (args[i].compareTo("-test") == 0)
				testFile = args[++i];
			else if (args[i].compareTo("-norm") == 0) {

				String n = args[++i];
				if (n.compareTo("sum") == 0) {
					nml = new SumNormalizor();
					normalize = true;
				} else if (n.compareTo("zscore") == 0) {
					nml = new ZScoreNormalizor();
					normalize = true;
				} else {
					System.out.println("Unknown normalizor: " + n);
					System.out.println("System will now exit.");
					System.exit(1);
				}
			} else {
				System.out
						.println("Unknown command-line parameter: " + args[i]);
				System.out.println("System will now exit.");
				System.exit(1);
			}
		}

		LogisticRankSVM logi_rankSvm = new LogisticRankSVM();
		long startTime = System.currentTimeMillis();
		System.out.println("program starts");
		logi_rankSvm.evaluate(trainFile, validationFile, testFile, "");
		long endTime = System.currentTimeMillis();
		System.out.println("past time:" + (endTime - startTime) / 1000 + "s");

	}

	public List<RankList> readInput(String inputFile) {
		FeatureManager fm = new FeatureManager();
		List<RankList> samples = fm.read3(inputFile);// read3(String fn) is
														// defined myself for
														// sake of my own
														// experiment
		return samples;
	}

	public int[] readFeature(String featureDefFile) {
		FeatureManager fm = new FeatureManager();
		int[] features = fm.getFeatureIDFromFile(featureDefFile);
		return features;
	}

	public void normalize(List<RankList> samples, int[] fids) {
		for (int i = 0; i < samples.size(); i++)
			nml.normalize(samples.get(i), fids);
	}

	public int[] getFeatureFromSampleVector(List<RankList> samples) {
		DataPoint dp = samples.get(0).get(0);
		int fc = dp.getFeatureCount();
		int[] features = new int[fc];
		for (int i = 0; i < fc; i++)
			features[i] = i + 1;
		return features;
	}

	public List<PartialPairList> getPartialPairForAllQueries(List<RankList> rll) {
		List<PartialPairList> ppll = new ArrayList<PartialPairList>();
		// int num=0;
		for (int i = 0; i < rll.size(); i++) {
			PartialPairList tem = getPartialPairForOneQuery(rll.get(i));
			ppll.add(tem);
			// num++;
		}
		// System.out.println(num);
		return ppll;
	}

	// convert labeled documents from one query into partialPair for the same
	// query
	public PartialPairList getPartialPairForOneQuery(RankList rl)// rl holds all
																	// documents
																	// for one
																	// query
	{
		PartialPairList ppl = new PartialPairList();
		for (int i = 0; i < rl.size(); i++) {
			for (int j = i + 1; j < rl.size(); j++) {
				if (rl.get(i).getLabel() != (rl.get(j).getLabel())) {
					ppl.add(new PartialPair(rl.get(i), rl.get(j)));
				}
			}
		}
		return ppl;

	}

	public List<String> getAllPartialPairID(List<PartialPairList> ppll) {
		List<String> strl = new ArrayList<String>();
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				strl.add(ppll.get(i).get(j).getPartialPairID());
			}
		}
		return strl;
	}

	public List<List<String>> getVRowsID(List<RankList> rll) {
		List<List<String>> sll = new ArrayList<List<String>>();
		for (int i = 0; i < rll.size(); i++) {
			List<String> sl = new ArrayList<String>();
			for (int j = 0; j < rll.get(i).size(); j++) {
				// put ith query's relative document id into a list
				sl.add(rll.get(i).get(j).getDocID());
			}
			sll.add(sl);
		}
		return sll;
	}

	public int RowSize_V(List<RankList> rll) {
		int total = 0;

		for (int i = 0; i < rll.size(); i++) {
			total += rll.get(i).size();
		}
		return total;
	}

	public HashMap<String, Integer> getRowIDofVMatrix(List<RankList> rll) {

		HashMap<String, Integer> hp = new HashMap<String, Integer>();
		int index = 0;
		for (int i = 0; i < rll.size(); i++) {
			for (int j = 0; j < rll.get(i).size(); j++) {
				String key = rll.get(i).get(j).getID() + "-"
						+ rll.get(i).get(j).getDocID();

				hp.put(key, index);
				index++;
			}
		}
		return hp;

	}

	public Matrix updateVMatrix(Matrix V_pre, List<PartialPairList> ppll,
			List<RankList> rll) {
		HashMap<String, Integer> hp = getRowIDofVMatrix(rll);
		// get the derivative of V_ac
		for (int i = 0; i < rll.size(); i++) {
			for (int j = 0; j < rll.get(i).size(); j++) {
				// iterate every vector V_ac

			}
		}
		//
		return null;
	}

	public Matrix parallel_sgd_random_JFun(PartialPair pp, Matrix V_old,
			List<PartialPairList> ppll, List<RankList> rll, int nThread)
			throws InterruptedException {
		HashMap<String, Integer> hp = hp_V;
		// Matrix V_new = new Matrix(V_old);
		// double eta = Math.pow(10, -3);
		Matrix V_new = new Matrix(V_old);
		// we use V_iq and V_jq to stand for the row id of the corresponding
		// documents related to partialPair pp
		int V_iq = hp.get(pp.getQueryID() + "-" + pp.getLargeDocID())
				.intValue();
		int V_jq = hp.get(pp.getQueryID() + "-" + pp.getSmallDocID())
				.intValue();
		double index_E = 0;
		double factor1 = 0f;
		for (int k = 0; k < ppll.size(); k++) {
			for (int l = 0; l < ppll.get(k).size(); l++) {
				// for a given partialPair X_ijq=ppll.get(i).get(j), we need to
				// compute the
				double innerProduct_V = V_old.getInnerProduct(V_iq, V_jq);
				double innerProduct_partialPair = pp.dotProduct(ppll.get(k)
						.get(l));
				index_E += innerProduct_V * innerProduct_partialPair;
			}
		}
		if (index_E > 20)
			factor1 = 0;
		else if (index_E < -20)
			factor1 = 1;
		else
			factor1 = 1 / (1 + Math.exp(index_E));
		if (factor1 == 0) {
			System.out.println("the gradient is 0 for partialPair "
					+ pp.getPartialPairID());
			return null;
		}
		// we parallelize the calculation
		ExecutorService es = Executors.newFixedThreadPool(nThread);
		List<Future<double[]>> resultList = new ArrayList<Future<double[]>>();
		// next we calculate the gradients of matrix V,ie all the elements in
		// matrix V

		for (int i = 0; i < rll.size(); i++) {// iterate all the queries
			for (int j = 0; j < rll.get(i).size(); j++) {// iterate all the
															// documents of
															// query i
				// find out the partialPairs which dataPoint=rll.get(i).get(j)
				// involves,
				int V_ac = hp_V.get(rll.get(i).get(j).getID() + "-"
						+ rll.get(i).get(j).getDocID());
				Future<double[]> fu = es.submit(new ThreadUpdateVMatrix(
						factor1, i, pp, V_ac, ppll, hp, V_old, learningRate));
				resultList.add(fu);

			}// end of iterating documents under the same query
		}// end of iterating queries

		es.shutdown();
		while (!es.awaitTermination(1, TimeUnit.SECONDS))
			;
		for (int i = 0; i < resultList.size(); i++) {
			try {
				V_new.setRowVector(resultList.get(i).get(), i);
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		/*
		 * for (int i = 0; i < rll.size(); i++) { for (int j = 0; j
		 * <rll.get(i).size() ; j++) { int V_ac =
		 * hp_V.get(rll.get(i).get(j).getID() + "-" +
		 * rll.get(i).get(j).getDocID()); double [] factor2 = new
		 * double[Matrix.ColsOfVMatrix]; for (int j2 = 0; j2 <
		 * ppll.get(i).size(); j2++) { double [] temp = new
		 * double[Matrix.ColsOfVMatrix]; PartialPair ite_pp =
		 * ppll.get(i).get(j2); String qid_largeDoc = ite_pp.getQueryID() + "-"
		 * + ite_pp.getLargeDocID(); String qid_smallDoc = ite_pp.getQueryID() +
		 * "-" + ite_pp.getSmallDocID(); if (V_ac == hp.get(qid_largeDoc)) { int
		 * docID_associatedWithV_ac = hp.get(qid_smallDoc); double multiplier
		 * =pp.dotProduct(ite_pp); // parameter factor2, stores the result of
		 * multiplication V_old.multiplyRowVector(docID_associatedWithV_ac,
		 * multiplier, temp); Matrix.RowVectorAddition(factor2, temp);
		 * 
		 * } else if(V_ac == hp.get(qid_smallDoc)){ int docID_associatedWithV_ac
		 * = hp.get(qid_largeDoc); double multiplier = pp.dotProduct(ite_pp);
		 * V_old.multiplyRowVector(docID_associatedWithV_ac, multiplier, temp);
		 * Matrix.RowVectorAddition(factor2, temp); } } double[] gradient =
		 * Matrix.multiplyRowVector(-factor1, factor2);
		 * Matrix.RowVectorAddition(V_new.getV()[V_ac],
		 * Matrix.multiplyRowVector(-learningRate, gradient));//negative
		 * direction of the gradient } }
		 */
		return V_new;
	}

	public Matrix sgd_random_JFun(PartialPair pp, Matrix V_old,
			List<PartialPairList> ppll, List<RankList> rll) {
		HashMap<String, Integer> hp = hp_V;
		Matrix V_new = new Matrix(V_old);
		double eta = Math.pow(10, -3);
		// we use V_iq and V_jq to stand for the row id of the corresponding
		// documents related to partialPair pp
		int V_iq = hp.get(pp.getQueryID() + "-" + pp.getLargeDocID())
				.intValue();
		int V_jq = hp.get(pp.getQueryID() + "-" + pp.getSmallDocID())
				.intValue();
		double index_E = 0;
		double factor1 = 0f;
		for (int k = 0; k < ppll.size(); k++) {
			for (int l = 0; l < ppll.get(k).size(); l++) {
				// for a given partialPair X_ijq=ppll.get(i).get(j), we need to
				// compute the
				double innerProduct_V = V_old.getInnerProduct(V_iq, V_jq);
				double innerProduct_partialPair = pp.dotProduct(ppll.get(k)
						.get(l));
				index_E += innerProduct_V * innerProduct_partialPair;
			}
		}
		if (index_E > 20)
			factor1 = 0;
		else if (index_E < -20)
			factor1 = 1;
		else
			factor1 = 1 / (1 + Math.exp(index_E));

		// next we calculate the gradients of matrix V,ie all the elements in
		// matrix V
		for (int i = 0; i < rll.size(); i++) {// iterate all the queries
			for (int j = 0; j < rll.get(i).size(); j++) {// iterate all the
															// documents of
															// query i
				// find out the partialPairs which dataPoint=rll.get(i).get(j)
				// involves,
				double[] factor2 = new double[Matrix.ColsOfVMatrix];
				int V_ac = hp_V.get(rll.get(i).get(j).getID() + "-"
						+ rll.get(i).get(j).getDocID());
				for (int j2 = 0; j2 < ppll.get(i).size(); j2++) {
					double[] temp = new double[Matrix.ColsOfVMatrix];

					PartialPair ite_pp = ppll.get(i).get(j2);
					String qid_largeDoc = ite_pp.getQueryID() + "-"
							+ ite_pp.getLargeDocID();
					String qid_smallDoc = ite_pp.getQueryID() + "-"
							+ ite_pp.getSmallDocID();
					if (V_ac == hp.get(qid_largeDoc)) {
						int docID_associatedWithV_ac = hp.get(qid_smallDoc);
						double multiplier = pp.dotProduct(ite_pp);
						// parameter factor2, stores the result of
						// multiplication
						V_old.multiplyRowVector(docID_associatedWithV_ac,
								multiplier, temp);
						Matrix.RowVectorAddition(factor2, temp);
						// flag = true;
					} else if (V_ac == hp.get(qid_smallDoc)) {
						int docID_associatedWithV_ac = hp.get(qid_largeDoc);
						double multiplier = pp.dotProduct(ite_pp);
						V_old.multiplyRowVector(docID_associatedWithV_ac,
								multiplier, temp);
						Matrix.RowVectorAddition(factor2, temp);
						// flag = true;
					}
				}
				double[] gradient = Matrix.multiplyRowVector(-factor1, factor2);
				Matrix.RowVectorAddition(V_new.getV()[V_ac],
						Matrix.multiplyRowVector(eta, gradient));
			}// end of iterating documents under the same query
		}// end of iterating queries
		return V_new;

	}

	public double[] derivative_JFun(String V_rowID, Matrix V_old,
			List<PartialPairList> ppll, List<RankList> rll) {
		HashMap<String, Integer> hp = hp_V;
		// the next two for loop to iterate every partialPair
		double[] der_value = new double[Matrix.ColsOfVMatrix];
		double[] total_der_value = new double[Matrix.ColsOfVMatrix];
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				// every single partialPair, we need to compute factor1 and
				// factor2, get the result of factor1*factor2
				double factor1 = 0f;
				/*
				 * double factor1_numerator = 0; double factor1_denominator =0;
				 */
				double index_E = 0;
				double[] derivative = new double[Matrix.ColsOfVMatrix];
				for (int kk = 0; kk < ppll.size(); kk++) {
					for (int ll = 0; ll < ppll.get(kk).size(); ll++) {
						// for a given partialPair X_ijq=ppll.get(i).get(j), we
						// need to compute the

						String queryID = ppll.get(kk).get(ll).getQueryID();
						String largeDocID = ppll.get(kk).get(ll)
								.getLargeDocID();
						String smallDocID = ppll.get(kk).get(ll)
								.getSmallDocID();
						int V_iq = hp.get(queryID + "-" + largeDocID)
								.intValue();
						int V_jq = hp.get(queryID + "-" + smallDocID)
								.intValue();
						double innerProduct_V = V_old.getInnerProduct(V_iq,
								V_jq);
						double innerProduct_partialPair = ppll.get(i).get(j)
								.dotProduct(ppll.get(kk).get(ll));
						index_E += innerProduct_V * innerProduct_partialPair;
					}
				}
				/*
				 * factor1_numerator = Math.exp(-index_E); factor1_denominator =
				 * 1+Math.exp(-index_E);
				 */
				// factor1 = Math.exp(-index_E)/(1+Math.exp(-index_E));
				if (index_E > 20)
					factor1 = 0;
				else if (index_E < -20)
					factor1 = 1;
				else
					factor1 = 1 / (1 + Math.exp(index_E));
				double[] factor2 = new double[Matrix.ColsOfVMatrix];
				String s = V_rowID.substring(0, V_rowID.indexOf("-")).trim();
				for (int k = 0; k < ppll.size(); k++) {
					// we only focus on the query which the derivated document
					// vector related to
					// ppll.get(k).size()>0 , which ensures there're at least
					// one partialPair under a query
					// ppll.get(k).get(0).getQueryID().equals(s), which ensures
					// we have found the query which the
					// derivated document belongs to
					if (ppll.get(k).size() > 0
							&& ppll.get(k).get(0).getQueryID().equals(s)) {
						// when we find the query derivated document belonging
						// to, we find another document related
						// to the derivated document under the same query
						for (int l = 0; l < ppll.get(k).size(); l++) {
							// boolean flag = false;
							double[] temp = new double[Matrix.ColsOfVMatrix];
							int V_ac = hp_V.get(V_rowID);
							PartialPair curr_pp = ppll.get(k).get(l);
							String qid_largeDoc = curr_pp.getQueryID() + "-"
									+ curr_pp.getLargeDocID();
							String qid_smallDoc = curr_pp.getQueryID() + "-"
									+ curr_pp.getSmallDocID();
							if (V_ac == hp.get(qid_largeDoc)) {
								int docID_associatedWithV_ac = hp
										.get(qid_smallDoc);
								double multiplier = ppll.get(i).get(j)
										.dotProduct(curr_pp);
								// parameter factor2, stores the result of
								// multiplication
								V_old.multiplyRowVector(
										docID_associatedWithV_ac, multiplier,
										temp);
								Matrix.RowVectorAddition(factor2, temp);
								// flag = true;
							} else if (V_ac == hp.get(qid_smallDoc)) {
								int docID_associatedWithV_ac = hp
										.get(qid_largeDoc);
								double multiplier = ppll.get(i).get(j)
										.dotProduct(curr_pp);
								V_old.multiplyRowVector(
										docID_associatedWithV_ac, multiplier,
										temp);
								Matrix.RowVectorAddition(factor2, temp);
								// flag = true;
							}
							/*
							 * if(flag == true){
							 * 
							 * Matrix.SetRowVector(der_value,
							 * Matrix.multiplyRowVector(-factor1, factor2)); //
							 * der_value = Matrix.multiplyRowVector(-factor1,
							 * factor2);
							 * Matrix.RowVectorAddition(total_der_value,
							 * der_value); }
							 */

						}
						break;
					}

				}
				Matrix.SetRowVector(der_value,
						Matrix.multiplyRowVector(-factor1, factor2));
				// der_value = Matrix.multiplyRowVector(-factor1, factor2);
				Matrix.RowVectorAddition(total_der_value, der_value);

			}
			System.out.println(i);
		}
		return total_der_value;
	}

	public double calculateObj_Jfun(List<PartialPairList> ppll, Matrix V) {
		HashMap<String, Integer> hp = hp_V;
		// the next two for loop to iterate every partialPair
		double J_value = 0f;
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				// every single partialPair, we need to compute ln(....)
				double index_E = 0f;
				for (int k = 0; k < ppll.size(); k++) {
					for (int l = 0; l < ppll.get(k).size(); l++) {
						// for a given partialPair X_ijq=ppll.get(i).get(j), we
						// need to compute the
						/*
						 * if(ppll.get(k).size()==0) continue;
						 */
						String queryID = ppll.get(k).get(l).getQueryID();
						String largeDocID = ppll.get(k).get(l).getLargeDocID();
						String smallDocID = ppll.get(k).get(l).getSmallDocID();
						int V_iq = hp.get(queryID + "-" + largeDocID)
								.intValue();
						int V_jq = hp.get(queryID + "-" + smallDocID)
								.intValue();
						double innerProduct_V = V.getInnerProduct(V_iq, V_jq);
						double innerProduct_partialPair = ppll.get(i).get(j)
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
			System.out.println(i);
		}
		return J_value;
	}

	public double parallelCalculateObj_Jfun(List<PartialPairList> ppll,
			Matrix V, int nThread) throws InterruptedException, Exception {
		HashMap<String, Integer> hp = hp_V;
		// the next two for loop to iterate every partialPair
		double J_value = 0f;
		ExecutorService es = Executors.newFixedThreadPool(nThread);
		List<Future<Double>> resultList = new ArrayList<Future<Double>>();
		for (int i = 0; i < ppll.size(); i++) {
			Future<Double> fu = es.submit(new ThreadCalculateObj_Jfun(ppll, i,
					V, hp));
			resultList.add(fu);
		}
		es.shutdown();
		while (!es.awaitTermination(1, TimeUnit.SECONDS))
			;
		for (Future<Double> future : resultList) {
			J_value += future.get().doubleValue();
		}

		return J_value;
	}

	public double parallelFullCPU_CalculateObj_Jfun(List<PartialPairList> ppll,
			Matrix V, int nThread) throws InterruptedException, Exception {
		HashMap<String, Integer> hp = hp_V;
		// the next two for loop to iterate every partialPair
		double J_value = 0f;
		for (int i = 0; i < ppll.size(); i++) {
			double subJ_oneQuery = parallelFullCPU_CalculateSubObj_Jfun(ppll,
					i, V, nThread);
			System.out.println("query " + i + "is over, its value is "
					+ subJ_oneQuery);
			J_value += subJ_oneQuery;
		}
		return J_value;
	}

	public double parallelFullCPU_CalculateSubObj_Jfun(
			List<PartialPairList> ppll, int q_index, Matrix V, int nThread)
			throws InterruptedException, ExecutionException {
		HashMap<String, Integer> hp = hp_V;
		double subJ_value = 0;
		PartialPairList ppl = ppll.get(q_index);
		ExecutorService es = Executors.newFixedThreadPool(nThread);
		List<Future<Double>> resultList = new ArrayList<Future<Double>>();
		int each_CPU_load = ppl.size() / nThread;
		int remaining = ppl.size() % nThread;
		if (each_CPU_load == 0) {
			if (remaining == 0) {
				return 0;
			} else {
				@SuppressWarnings("unchecked")
				Future<Double> fu = es
						.submit(new ThreadCalculate_PartsPartialPairsInOneQuery_Obj(
								ppll, q_index, V, hp, -1, each_CPU_load,
								remaining));
				// -1 is a special case for each_CPU_load==0, and remaining>0,
				// which means
				// the partialPair quantity under this q_index is very small
				resultList.add(fu);
				es.shutdown();
				while (!es.awaitTermination(1, TimeUnit.SECONDS))
					;
			}
		} else {

			for (int cpu_index = 0; cpu_index < nThread - 1; cpu_index++) {
				int remaining_pp = 0;
				@SuppressWarnings("unchecked")
				Future<Double> fu = es
						.submit(new ThreadCalculate_PartsPartialPairsInOneQuery_Obj(
								ppll, q_index, V, hp, cpu_index, each_CPU_load,
								remaining_pp));
				resultList.add(fu);
			}
			int remaining_pp = ppl.size() % nThread; // add the remaining pp to
														// the last CPU
			@SuppressWarnings("unchecked")
			Future<Double> fu = es
					.submit(new ThreadCalculate_PartsPartialPairsInOneQuery_Obj(
							ppll, q_index, V, hp, nThread - 1, each_CPU_load,
							remaining_pp));
			resultList.add(fu);
			es.shutdown();
			while (!es.awaitTermination(1, TimeUnit.SECONDS))
				;
		}

		for (Future<Double> future : resultList) {
			subJ_value += future.get().doubleValue();
		}

		return subJ_value;
	}

	public Boolean isConverge(double[][] V1, double[][] V2, int rows, int cols,
			double epsilon) {
		double error = 0f;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				error += Math.abs(V1[i][j] - V2[i][j]);
			}
		}
		if (error <= epsilon) {
			return true;
		} else {
			return false;
		}
	}

	public PartialPair getPP_RandomQuery(List<PartialPairList> ppll) {
		Random rand = new Random();
		//
		boolean flag = true;
		int query_index = 0;
		int pp_query_index = 0;
		while (flag) {
			query_index = rand.nextInt(ppll.size());
			int pp_num = ppll.get(query_index).size();
			if (pp_num != 0) {
				flag = false;
				pp_query_index = rand.nextInt(pp_num);
			}
		}
		return ppll.get(query_index).get(pp_query_index);

	}

	public List<ArrayList<Double>> getScoreByFun(List<RankList> rll, Vector w) {
		List<ArrayList<Double>> dll = new ArrayList<ArrayList<Double>>();
		// List<PartialPairList> ppll = getPartialPairForAllQueries(rll);
		// Vector w = getW(rll, matrixV);
		for (int i = 0; i < rll.size(); i++) {
			ArrayList<Double> dl = new ArrayList<Double>();
			for (int j = 0; j < rll.get(i).size(); j++) {
				Vector x_ij = new Vector(rll.get(i).get(j).getFeatureVector());
				double scoreByFun = Vector.dotProduct(w, x_ij) - w.getVec()[0]
						* x_ij.getVec()[0];
				dl.add(scoreByFun);
			}
			dll.add(dl);
		}
		return dll;
	}

	public Vector getW(List<RankList> rll, Matrix matrixV) {
		List<PartialPairList> ppll = getPartialPairForAllQueries(rll);
		Vector w = new Vector(DataPoint.featureCount + 1);// feature 0 is
															// reserved for
															// use,so we extend
															// the dimension.
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				PartialPair pp = ppll.get(i).get(j);
				String qid = pp.getQueryID();
				String largeDocID = qid + "-" + pp.getLargeDocID();
				String smallDocID = qid + "-" + pp.getSmallDocID();
				int v_iq = hp_V.get(largeDocID);
				int v_jq = hp_V.get(smallDocID);
				double factor = matrixV.getInnerProduct(v_iq, v_jq);
				double[] temp = Matrix.multiplyRowVector(factor,
						pp.getPartialFVals());
				temp[0] = 0;// position 0 is undefined for
				w = Vector.addition(w, new Vector(temp));
			}
		}
		return w;
	}

	public String makeDir(String tail) {
		String[] sub = tail.split("/");
		File dir = new File(".");
		for (int i = 0; i < sub.length; i++) {
			if (!dir.exists()) {
				dir.mkdir();
			}
			File dir2 = new File(dir + File.separator + sub[i]);
			if (!dir2.exists()) {
				dir2.mkdir();
			}
			dir = dir2;
		}
		return dir.toString();
	}
	public void calculatePredictionScore(List<RankList> rll_train,
			List<RankList> rll_validation,List<RankList> rll_test,Vector w,String timeStamp,String valid_round){
		if (!valid_round.equals("")) {
			valid_round = "-" + valid_round;
		}
		String prediction_dir = "output_data/factorizedLR/prediction/" + fold_n;
		makeDir(prediction_dir);
		String prediction_filename = null;
		// get the prediction score of train set
		prediction_filename = prediction_dir + "/" + allFile_prefix + timeStamp 
				+ "prediction_train" + valid_round + ".txt";
		List<ArrayList<Double>> dll_train1 = getScoreByFun(rll_train, w);
		FileUtils.write2File(prediction_filename, dll_train1, "");
		// get the prediction score of validation set
		prediction_filename = prediction_dir + "/" + allFile_prefix + timeStamp
				+ "prediction_validation" + valid_round + ".txt";
		List<ArrayList<Double>> dll_vali1 = getScoreByFun(rll_validation, w);
		FileUtils.write2File(prediction_filename, dll_vali1, "");
		// get the prediction score of test set
		List<ArrayList<Double>> dll_test1 = getScoreByFun(rll_test, w);
		prediction_filename = prediction_dir + "/" + allFile_prefix + timeStamp
				+ "prediction_test" + valid_round + ".txt";
		FileUtils.write2File(prediction_filename, dll_test1, "");
	}
	public void invokePerlScript(String date_timeStamp,String valid_round){
		String perlResult_dir = "perlEvaluate/factorizedLR/" + fold_n ;
		makeDir(perlResult_dir);
		if(!valid_round.equals(null))
			valid_round = "-" + valid_round;
		String perlResult_filename = perlResult_dir	+ "/" + allFile_prefix + 
				date_timeStamp + "test" + valid_round + ".txt";
		String[] perl_cmd = {
				"perl",
				"perlEvaluate/eval-score-mslr.pl",
				"data/OHSUMED/OHSUMED/QueryLevelNorm/" + fold_n + "/test.txt",
				"output_data/factorizedLR/prediction/" + fold_n + "/" + allFile_prefix
						+ date_timeStamp + "prediction_test" + valid_round+ ".txt", 
				perlResult_filename, 
				"0" };
		Process proc =null;
		try{
		proc = Runtime.getRuntime().exec(perl_cmd);		
		}catch(Exception e){
			System.out.println("error executing perl_cmd");
			int exitValue = proc.exitValue();
			System.out.println("exitValue: " + exitValue);
		}		
	}
	public void evaluate(String trainFile, String validationFile,
			String testFile, String featureDefFile)
			throws InterruptedException, Exception {
		List<RankList> rll_train = readInput(trainFile);// read input
		hp_V = getRowIDofVMatrix(rll_train);
		List<RankList> rll_validation = null;
		if (validationFile.compareTo("") != 0)
			rll_validation = readInput(validationFile);
		List<RankList> rll_test = null;
		if (testFile.compareTo("") != 0)
			rll_test = readInput(testFile);
		int[] features = readFeature(featureDefFile);// read features
		if (features == null)// no features specified ==> use all features in
								// the training file
			features = getFeatureFromSampleVector(rll_train);
		if (normalize) {
			normalize(rll_train, features);
			if (rll_validation != null)
				normalize(rll_validation, features);
			if (rll_test != null)
				normalize(rll_test, features);
		}
		// get all partialPairs sorted by different queries
		fold_n = (String) trainFile.subSequence(trainFile.indexOf("Fold"),
				trainFile.indexOf("Fold") + 5);

		Matrix.setRowsOfVMatrix(rll_train.size());		
		ResultClass rc = learn(rll_train, rll_validation , rll_test);//each learn process has its own timeStamp,
											//in order to distinguish each learn process from each other
		Matrix v = rc.getMatrix();
		String date_timeStamp = rc.getTimeStamp()+ "-"; 
//		Matrix v = learn(rll_train);
		// Matrix v= new Matrix();		
		String dir_final_matrix = "output_data/factorizedLR/final_matrixV/" + fold_n;
		makeDir(dir_final_matrix);
		String filename = dir_final_matrix + "/" + allFile_prefix + date_timeStamp + "matrixV.txt";
		FileUtils.write2File(filename, v, date_timeStamp);
//		Matrix v2 = FileUtils.readFromFileGetLatestMatrix(filename);
		Vector w = getW(rll_train, v);
		String dir_final_w = "output_data/factorizedLR/final_w/" + fold_n;
		makeDir(dir_final_w);
		filename = dir_final_w + "/" + allFile_prefix + date_timeStamp + "w.txt";
		FileUtils.write2File(filename, w, date_timeStamp);
		
		// -------------we calculate the prediction score of each data set-------------
		/*String prediction_dir = "output_data/factorizedLR/prediction/" + fold_n;
		makeDir(prediction_dir);
		String prediction_filename = null;
		// get the prediction score of train set
		prediction_filename = prediction_dir + "/" + allFile_prefix + date_timeStamp
				+ "prediction_train.txt";
		List<ArrayList<Double>> dll_train1 = getScoreByFun(rll_train, w);
		FileUtils.write2File(prediction_filename, dll_train1, "");
		// get the prediction score of validation set
		prediction_filename = prediction_dir + "/" + allFile_prefix + date_timeStamp
				+ "prediction_validation.txt";
		List<ArrayList<Double>> dll_vali1 = getScoreByFun(rll_validation, w);
		FileUtils.write2File(prediction_filename, dll_vali1, "");
		// get the prediction score of test set
		List<ArrayList<Double>> dll_test1 = getScoreByFun(rll_test, w);
		prediction_filename = prediction_dir + "/" + allFile_prefix + date_timeStamp
				+ "prediction_test.txt";
		FileUtils.write2File(prediction_filename, dll_test1, "");*/
		// -----------------------------------------------------------------------------
		calculatePredictionScore(rll_train, rll_validation, rll_test, w, date_timeStamp,"");
		//--------------- invoke perl script -------------------------------------------		
		/*String perlResult_dir = "perlEvaluate/factorizedLR/" + fold_n ;
		makeDir(perlResult_dir);
		String perlResult_filename = perlResult_dir	+ "/" + allFile_prefix + date_timeStamp + "test.txt";
		String[] perl_cmd = {
				"perl",
				"perlEvaluate/eval-score-mslr.pl",
				"data/OHSUMED/OHSUMED/QueryLevelNorm/" + fold_n + "/test.txt",
				"output_data/factorizedLR/prediction/" + fold_n + "/" + allFile_prefix
						+ date_timeStamp + "prediction_test.txt", 
				perlResult_filename, 
				"0" };
		Process proc =null;
		try{
		proc = Runtime.getRuntime().exec(perl_cmd);		
		}catch(Exception e){
			System.out.println("error executing perl_cmd");
			int exitValue = proc.exitValue();
			System.out.println("exitValue: " + exitValue);
		}		*/
		//------------------------------------------------------------------------------
		invokePerlScript(date_timeStamp,"");
		System.out.println("evaluate process over");
	}

	/**
	 * HAVE TO BE OVER-RIDDEN IN SUB-CLASSES
	 */
	public void init() {

	}

	public ResultClass learn(List<RankList> rll_train,List<RankList> rll_validation ,List<RankList> rll_test) throws InterruptedException,
			Exception {
		List<PartialPairList> ppll = getPartialPairForAllQueries(rll_train);
		// System.out.println(getAllPartialPairID(ppll).size());
		List<RankList> rll = rll_train;
		System.out.println("total partialPair of all query:"
				+ getAllPartialPairID(ppll).size());
		/* List<String> rowID_V = getRowIDofVMatrix(train); */
		Matrix.RowsOfVMatrix = RowSize_V(rll_train);
		Matrix V_0 = new Matrix();
		V_0.randomize();
		Matrix V_temp = new Matrix(V_0);
		Matrix V = new Matrix(V_0);
		double startTime = 0;
		double endTime = 0;
		double Jfun_pre = 0;
		double Jfun_new = 0;
		int validCount = 0;
		SimpleDateFormat sdf = new SimpleDateFormat("yy-MM-dd-HH-mm");
		String date_timeStamp = sdf.format(new Date()) + "-";//this timeStamp distinguish this learn process from other learning process
		startTime = System.currentTimeMillis(); // start the time
		// System.out.println(new Date());
		/*
		 * double full_cpu_Jfun_new = parallelFullCPU_CalculateObj_Jfun(ppll, V,
		 * nThread); endTime = System.currentTimeMillis();
		 * System.out.println("full_cpu_Jfun_new = " + full_cpu_Jfun_new);
		 * System.out
		 * .println("the time of calculating full_cpu_Jfun_new in hours: " +
		 * (endTime - startTime) / 1000 / 60 / 60 + " h");
		 */
		startTime = System.currentTimeMillis();
		// Jfun_new = parallelCalculateObj_Jfun(ppll, V, nThread);
		   Jfun_new = parallelFullCPU_CalculateObj_Jfun(ppll, V,nThread);
		// System.out.println(new Date());
		endTime = System.currentTimeMillis();
		System.out.println("the time of calculating Jfun_new in hours: "
				+ (endTime - startTime) / 1000 / 60 / 60 + " h");
		PartialPair pp = null;
		boolean isAmplifyLearningRate = false;
		do {

			Jfun_pre = Jfun_new;
			startTime = System.currentTimeMillis(); // start the time
			if (isAmplifyLearningRate) {
				this.learningRate *= 1.05;
				isAmplifyLearningRate = false;
			}
			System.out.println(new Date());
			do {
				pp = getPP_RandomQuery(ppll);
				V_temp = parallel_sgd_random_JFun(pp, V, ppll, rll, nThread);
			} while (V_temp == null);
			endTime = System.currentTimeMillis(); // end the time
			System.out.println(new Date());
			System.out
					.println("the time of updating V with a random PartialPair in seconds: "
							+ (endTime - startTime) / 1000 + " s");
			// Jfun_new = parallelCalculateObj_Jfun(ppll, V_temp, nThread);
		    Jfun_new = parallelFullCPU_CalculateObj_Jfun(ppll, V_temp,nThread);
	//		Jfun_new = -1;
			String dir_inLearning_matrixV = "output_data/factorizedLR/inLearning_matrixV/"
					+ fold_n;
			String dir_inLearning_w = "output_data/factorizedLR/inLearning_w/" + fold_n;
			String dir_final_w = "output_data/factorizedLR/final_w/" + fold_n;
			String dir_final_matrixV = "output_data/factorizedLR/final_matrixV/"
					+ fold_n;
			makeDir(dir_inLearning_matrixV);
			makeDir(dir_inLearning_w);
			makeDir(dir_final_w);
			makeDir(dir_final_matrixV);
			if (Jfun_new < Jfun_pre) {
				V = V_temp;
				Vector w = getW(rll_train, V);
				validCount++;
				if (validCount % 1 == 0) {
					String description = "current learningRate is:"
							+ learningRate + ",after " + validCount
							+ "rounds , the V_new Matrix is:";
					String fileName = null;
					
					fileName = dir_inLearning_matrixV + "/" + allFile_prefix + date_timeStamp + "matrixV.txt";
					FileUtils.write2File(fileName, V, description);

					description = "current learningRate is:" + learningRate
							+ ",after" + validCount + "rounds , the w is:";
					fileName = dir_inLearning_w + "/" + allFile_prefix + date_timeStamp + "w.txt";
					FileUtils.write2File(fileName, w, "");
					System.out.println("Jfun_pre = " + Jfun_pre);
					System.out.println("Jfun_new = " + Jfun_new);
					System.out.println("round " + validCount
							+ ", the difference is " + (Jfun_pre - Jfun_new));
					isAmplifyLearningRate = true;
					if(validCount%output_interval==0){
						String validCount_str = String.valueOf(validCount);
						makeDir(dir_final_w);
						fileName = dir_final_w + "/" + allFile_prefix + date_timeStamp + "final_w-" + validCount_str + ".txt";
						makeDir(dir_final_matrixV);
						FileUtils.write2File(fileName, w, fileName);
						fileName = dir_final_matrixV + "/" + allFile_prefix + date_timeStamp + "final_matrixV-" + validCount_str + ".txt";
						FileUtils.write2File(fileName, w, fileName);
						calculatePredictionScore(rll_train, rll_validation, rll_test, w, date_timeStamp, validCount_str);
						invokePerlScript(date_timeStamp, validCount_str);
						
					}
					
				}
			} else {
				if (learningRateAttenuationTime > 0) {
					while (Jfun_new > Jfun_pre) {
						LogisticRankSVM.learningRate /= 2;
						V_temp = parallel_sgd_random_JFun(pp, V, ppll, rll,
								nThread);
						// Jfun_new = parallelCalculateObj_Jfun(ppll,
						// V_temp,nThread);
						Jfun_new = parallelFullCPU_CalculateObj_Jfun(ppll, V_temp,
								nThread);
					}
					validCount++;
					learningRateAttenuationTime--;
					V = V_temp;
					Vector w = getW(rll_train, V);
					if (validCount % 1 == 0) {
						String description = "current learningRate is: "
								+ learningRate + ",after " + validCount
								+ "rounds , the V_new Matrix is:";
						String fileName = null;
						fileName = dir_inLearning_matrixV + "/" + allFile_prefix + date_timeStamp + "matrixV.txt";
						FileUtils.write2File(fileName, V, description);

						description = "current learningRate is:" + learningRate
								+ ",after" + validCount + "rounds , the w is:";
						fileName = dir_inLearning_w + "/" + allFile_prefix + "w.txt";
						FileUtils.write2File(fileName, w, "");
						System.out.println("Jfun_pre = " + Jfun_pre);
						System.out.println("Jfun_new = " + Jfun_new);
						System.out.println("round " + validCount
								+ ", the difference is "
								+ (Jfun_pre - Jfun_new));
						isAmplifyLearningRate = true;
						if(validCount%output_interval==0){
							String validCount_str = String.valueOf(validCount);
							makeDir(dir_final_w);
							fileName = dir_final_w + "/" + allFile_prefix + date_timeStamp + "final_w-" + validCount_str + ".txt";
							makeDir(dir_final_matrixV);
							FileUtils.write2File(fileName, w, fileName);
							fileName = dir_final_matrixV + "/" + allFile_prefix + date_timeStamp + "final_matrixV" + validCount_str + ".txt";
							FileUtils.write2File(fileName, w, fileName);
							calculatePredictionScore(rll_train, rll_validation, rll_test, w, date_timeStamp, validCount_str);
							invokePerlScript(date_timeStamp, validCount_str);
							
						}
					}
					continue;
				}

				System.out.println("Jfun_pre = " + Jfun_pre);
				System.out.println("Jfun_new = " + Jfun_new);
				System.out
						.println("Jfun_new has been larger than Jfun_pre, exit now");
				System.out.println("round " + validCount
						+ ", the difference is "
						+ Math.abs(Jfun_new - Jfun_pre));
				break;
			}

		} while (Jfun_pre - Jfun_new > epsilon && validCount < maxIterations);
		ResultClass rc = new ResultClass(V, date_timeStamp);
	//	return V;
		return rc;
	}

}
