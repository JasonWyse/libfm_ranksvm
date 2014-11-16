package ict.edu.learning.baseline;

import ict.edu.learning.measure.Measurement;
import ict.edu.learning.utilities.FileUtils;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import ciir.umass.edu.features.FeatureManager;
import ciir.umass.edu.features.Normalizer;
import ciir.umass.edu.features.SumNormalizor;
import ciir.umass.edu.features.ZScoreNormalizor;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.PartialPair;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Vector;
import ciir.umass.edu.metric.ERRScorer;

public class OriginalLogistic {
public static int w_length = 5;
public static double learningRate = 0.0001;
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
public static int nThread = 1;
public static int maxIterations=5;
public static HashMap<String, Integer> hp_V = null;
	public OriginalLogistic() {
		// TODO Auto-generated constructor stub
	}
	
	public static void main(String args[]) throws InterruptedException, Exception{
		String trainFile = "";
		String featureDescriptionFile = "";
		double ttSplit = 0.0;//train-test split
		double tvSplit = 0.0;//train-validation split
		int foldCV = -1;
		String validationFile = "";
		String testFile = "";
		int rankerType = 10;//our own logistic ranksvm
		String trainMetric = "ERR@10";
		String testMetric = "";
		
		String savedModelFile = "";
		String rankFile = "";
		boolean printIndividual = false;
		
		//for my personal use
		String indriRankingFile = "";
		String scoreFile = "";		
		if(args.length < 2)
		{
			
			System.out.println("not enough parameter");
			return;
		}
		
		for(int i=0;i<args.length;i++)
		{
			if(args[i].compareTo("-train")==0)
				trainFile = args[++i];
			else if(args[i].compareTo("-ranker")==0)
				rankerType = Integer.parseInt(args[++i]);
			else if(args[i].compareTo("-feature")==0)
				featureDescriptionFile = args[++i];
			else if(args[i].compareTo("-metric2t")==0)
				trainMetric = args[++i];
			else if(args[i].compareTo("-metric2T")==0)
				testMetric = args[++i];
			else if(args[i].compareTo("-nThread")==0)
				nThread = Integer.parseInt(args[++i]);
			else if(args[i].compareTo("-maxIterations")==0)
				nThread = Integer.parseInt(args[++i]);
			else if(args[i].compareTo("-learningRate")==0)
				learningRate = Double.parseDouble(args[++i]);
			else if(args[i].compareTo("-gmax")==0)
				ERRScorer.MAX = Math.pow(2, Double.parseDouble(args[++i]));						
			else if(args[i].compareTo("-tts")==0)
				ttSplit = Double.parseDouble(args[++i]);
			else if(args[i].compareTo("-tvs")==0)
				tvSplit = Double.parseDouble(args[++i]);
			else if(args[i].compareTo("-kcv")==0)
				foldCV = Integer.parseInt(args[++i]);
			else if(args[i].compareTo("-validate")==0)
				validationFile = args[++i];
			else if(args[i].compareTo("-test")==0)
				testFile = args[++i];
			else if(args[i].compareTo("-norm")==0)
			{
				
				String n = args[++i];
				if(n.compareTo("sum") == 0)
					{
						nml = new SumNormalizor();
						normalize = true;
					}
				else if(n.compareTo("zscore") == 0)
					{
						nml = new ZScoreNormalizor();
						normalize = true;	
					}
				else
				{
					System.out.println("Unknown normalizor: " + n);
					System.out.println("System will now exit.");
					System.exit(1);
				}
			}		
			else
			{
				System.out.println("Unknown command-line parameter: " + args[i]);
				System.out.println("System will now exit.");
				System.exit(1);
			}
		}
		
		OriginalLogistic ori_logi=new OriginalLogistic();
		long startTime=System.currentTimeMillis();   
		System.out.println("program starts");
		ori_logi.evaluate(trainFile, validationFile, testFile, "");
		long endTime=System.currentTimeMillis(); 
		System.out.println("past time:"+(endTime-startTime)/1000+"s");
		
	}//end of main()
	
	public Vector derivate_L_W(List<PartialPairList> ppll, Vector vec_w){
		
		Vector.setVectorSize(PartialPair.getFeatureCount());
		Vector gradient = new Vector();
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				double factor1=0;				
				Vector pp_features = new Vector(ppll.get(i).get(j).getPartialFVals());
				double index_E = Vector.dotProduct(vec_w, pp_features);
				if(index_E>20){
					factor1 = 0;
					continue;
				}
					
				else if(index_E<-20)
					factor1 = 1;
				else
					factor1 = 1/(1+Math.exp(index_E));
				gradient = Vector.multiply(-factor1, pp_features);
			}
		}
		return gradient;
	}
	public List<RankList> readInput(String inputFile)	
	{
		FeatureManager fm = new FeatureManager();
		List<RankList> samples = fm.read3(inputFile);//read3(String fn) is defined myself for sake of my own experiment
		return samples;
	}
	public int[] readFeature(String featureDefFile)
	{
		FeatureManager fm = new FeatureManager();
		int[] features = fm.getFeatureIDFromFile(featureDefFile);
		return features;
	}
	public int[] getFeatureFromSampleVector(List<RankList> samples)
	{
		DataPoint dp = samples.get(0).get(0);
		int fc = dp.getFeatureCount();
		int[] features = new int[fc];
		for(int i=0;i<fc;i++)
			features[i] = i+1;
		return features;
	}
	public List<PartialPairList> getPartialPairForAllQueries(List<RankList> rll)
	{
		List<PartialPairList> ppll =new ArrayList<PartialPairList>();
		//int num=0;
		for (int i = 0; i < rll.size(); i++) {
			PartialPairList tem = getPartialPairForOneQuery(rll.get(i));
			ppll.add(tem);
			//num++;
		}
		//System.out.println(num);
		return ppll;
	}
	public PartialPairList getPartialPairForOneQuery(RankList rl)//rl holds all documents for one query 
	{
		PartialPairList ppl = new PartialPairList();
		for (int i = 0; i < rl.size(); i++) {
			for (int j = i+1; j < rl.size(); j++) {
				if(rl.get(i).getLabel()!=(rl.get(j).getLabel())){
					ppl.add(new PartialPair(rl.get(i),rl.get(j)));
				}
			}
		}
		return ppl;
		
	}
	public List<ArrayList<Double>> getScoreByFun(List<RankList> rll,Vector w){
		List<ArrayList<Double>> dll = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < rll.size(); i++) {
			ArrayList<Double> dl = new ArrayList<Double>();
			for (int j = 0; j <rll.get(i).size() ; j++) {
				Vector v = new Vector(rll.get(i).get(j).getFeatureVector());
				double scoreByFun = Vector.dotProduct(w, v);
				dl.add(scoreByFun);
			}
			dll.add(dl);
		}
		return dll;
	}
	public double[] getRealLabels(int [] position_docs_Qi,List<ArrayList<Double>> dll){
		
		return null;
	}
	
	public double Obj_Jfun_originalLogistic(List<PartialPairList> ppll, Vector w){
		double total = 0;
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				double[] vals = ppll.get(i).get(j).getPartialFVals();
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
		}
		return total;
	}
	public Vector derivate_w(List<PartialPairList> ppll, Vector w){
		Vector gradient = new Vector();
		for (int i = 0; i < ppll.size(); i++) {
			for (int j = 0; j < ppll.get(i).size(); j++) {
				Vector x_ijq = new Vector(ppll.get(i).get(j).getPartialFVals());
				double index_E = Vector.dotProduct(w,x_ijq);
				double coefficient = 1/(1+Math.exp(index_E));
				Vector v=Vector.multiply(-coefficient, x_ijq);
				gradient = Vector.addition(gradient,v);
			}
		}
		return gradient;		
	}
	public void normalize(List<RankList> samples, int[] fids)
	{
		for(int i=0;i<samples.size();i++)
			nml.normalize(samples.get(i), fids);
	}
	public void evaluate(String trainFile, String validationFile, String testFile, String featureDefFile) throws InterruptedException, Exception
	{
		List<RankList> train = readInput(trainFile);//read input		
		List<RankList> validation = null;
              if(validationFile.compareTo("")!=0)
			validation = readInput(validationFile);
		List<RankList> test = null;
		if(testFile.compareTo("")!=0)
			test = readInput(testFile);
		int[] features = readFeature(featureDefFile);//read features
		if(features == null)//no features specified ==> use all features in the training file
			features = getFeatureFromSampleVector(train);
		Vector.setVectorSize(features.length);
		if(normalize)
		{
			normalize(train, features);
			if(validation != null)
				normalize(validation, features);
			if(test != null)
				normalize(test, features);
		}	
		// get all partialPairs sorted by different queries
		Vector w = learn(train);	
		List<ArrayList<Double>> dll = getScoreByFun(test,w);
		double map= Measurement.MAP(dll, test);
		double ndcg = Measurement.NDCG(dll, test, 7);
		System.out.println(map);
		System.out.println(ndcg);
		System.out.println("learning process over");
	}
	
	public Vector learn(List<RankList> train){
		List<PartialPairList> ppll = getPartialPairForAllQueries(train);
		List<RankList> rll = train;		
		long startTime = 0;
		long endTime = 0;
		System.out.println(new Date());
		startTime=System.currentTimeMillis();
		double Jfun_pre = Double.MAX_VALUE-1;
		double Jfun_new = Double.MAX_VALUE;
		Vector w = new Vector();
		w.randomize();
		Jfun_pre = Obj_Jfun_originalLogistic(ppll, w);
		Jfun_new = Jfun_pre;
		int roundCount = 0;
		int learningRateAttenuationTime = 5;
		do{			
			Jfun_pre = Jfun_new;
			Vector gradient = derivate_w(ppll,w);
			Vector tem_w = Vector.addition(w, Vector.multiply(-(this.learningRate) ,gradient));
			Jfun_new = Obj_Jfun_originalLogistic(ppll, tem_w);
			
			if(Jfun_new<Jfun_pre){
				roundCount++;				
				w = tem_w; 
				if(roundCount%3==0){
					System.out.println("Jfun_new is:" + Jfun_new);
					System.out.println("Jfun_pre is:" + Jfun_pre);
					FileUtils.write2File("Vector_W.txt", w, "");
				}
			}
			else{
				
				if(learningRateAttenuationTime>0){
					while(Jfun_new>Jfun_pre){
						this.learningRate /=2;						
						tem_w = Vector.addition(w, Vector.multiply(-(this.learningRate) ,gradient));
						Jfun_new = Obj_Jfun_originalLogistic(ppll, tem_w); 
					}			
					learningRateAttenuationTime--;
					w = tem_w;
					roundCount++;
					continue;
				}
				
				System.out.println("Jfun_new is:" + Jfun_new);
				System.out.println("Jfun_pre is:" + Jfun_pre);
				System.out.println("after round " + roundCount +",Jfun_new is greater than Jfun_pre");
				break;
				
			} 
	        
		}while(Jfun_pre-Jfun_new>epsilon && roundCount < maxIterations);
		System.out.println(new Date());
		endTime=System.currentTimeMillis();
		System.out.println("learning process is over, it costs " + (endTime-startTime)/1000 +"seconds");
		return w;
	}
}
