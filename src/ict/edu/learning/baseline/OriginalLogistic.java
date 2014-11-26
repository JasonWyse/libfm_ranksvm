package ict.edu.learning.baseline;

import ict.edu.learning.measure.Measurement;
import ict.edu.learning.multiThread.ThreadCalculateLRObj_Jfun;
import ict.edu.learning.multiThread.ThreadCalculateLR_Gradient;
import ict.edu.learning.multiThread.ThreadUpdateVMatrix;
import ict.edu.learning.utilities.FileUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
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
import ciir.umass.edu.learning.PartialPair;
import ciir.umass.edu.learning.PartialPairList;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Vector;
import ciir.umass.edu.metric.ERRScorer;

public class OriginalLogistic {
public static int w_length = 5;
public static double learningRate = 0.001;
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
public static int nThread = 5;
public static int maxIterations=1000;
public static int learningRateAttenuationTime = 5;
public static String allFile_prefix;
String fold_n = null;
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
			else if(args[i].compareTo("-epsilon")==0)
				epsilon = Double.parseDouble(args[++i]);			
			else if(args[i].compareTo("-maxIterations")==0)
				maxIterations = Integer.parseInt(args[++i]);
			else if(args[i].compareTo("-learningRateAttenuationTime")==0)
				learningRateAttenuationTime = Integer.parseInt(args[++i]);
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
			else if(args[i].compareTo("-allFile_prefix")==0)
				allFile_prefix = args[++i]+"-";
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
				double scoreByFun = (Vector.dotProduct(w, v)-w.getVec()[0]*v.getVec()[0]);				
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
	public double Obj_Jfun_parallelOriginalLogistic(List<PartialPairList> ppll, Vector w) throws InterruptedException{
		double total = 0;	
		ExecutorService es = Executors.newFixedThreadPool(nThread);
		List<Future<Double>> resultList = new ArrayList<Future<Double>>();		
		for (int i = 0; i < ppll.size(); i++) {//iterate all the queries						
			Future<Double> fu = es.submit(new ThreadCalculateLRObj_Jfun(ppll, i , w));	
			resultList.add(fu);
		}		
		es.shutdown();		 
		while (!es.awaitTermination(10, TimeUnit.SECONDS));		
		for (int i = 0; i < resultList.size(); i++) {
			try {
				total += resultList.get(i).get(); 
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
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
	public Vector derivate_w_parallel(List<PartialPairList> ppll, Vector w) throws InterruptedException{
		Vector gradient = new Vector();
		ExecutorService es = Executors.newFixedThreadPool(nThread);
		List<Future<Vector>> resultList = new ArrayList<Future<Vector>>();
		for (int i = 0; i < ppll.size(); i++) {			
			Future<Vector> fu = es.submit(new ThreadCalculateLR_Gradient(ppll, i , w));	
			resultList.add(fu);			
		}
		es.shutdown();		 
		while (!es.awaitTermination(10, TimeUnit.SECONDS));		
		for (int i = 0; i < resultList.size(); i++) {
			try {
				gradient = Vector.addition(gradient, resultList.get(i).get());//gradient += resultList.get(i).get(); 
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return gradient;		
	}
	public void normalize(List<RankList> samples, int[] fids)
	{
		for(int i=0;i<samples.size();i++)
			nml.normalize(samples.get(i), fids);
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
		Vector.setVectorSize(features.length+1);
		if(normalize)
		{
			normalize(train, features);
			if(validation != null)
				normalize(validation, features);
			if(test != null)
				normalize(test, features);
		}	
		// get all partialPairs sorted by different queries
		fold_n = (String) trainFile.subSequence(trainFile.indexOf("Fold"), trainFile.indexOf("Fold")+5);
		Vector w = learn(train);					
		String dir = "output_data/originalLR/final_w/"+fold_n;
		makeDir(dir);	
		String dir2 = "output_data/originalLR/prediction/" + fold_n;
		makeDir(dir2);
		String filename = dir + "/w.txt";
		FileUtils.write2File(filename, w, fold_n);
		List<ArrayList<Double>> dll_train = getScoreByFun(train,w);
		List<ArrayList<Double>> dll_vali = getScoreByFun(validation,w);
		List<ArrayList<Double>> dll_test = getScoreByFun(test,w);		
		filename = dir2 + "/" + allFile_prefix + "prediction_train.txt";
		FileUtils.write2File(filename, dll_train, fold_n);
		filename = dir2 + "/" + allFile_prefix + "prediction_validation.txt";
		FileUtils.write2File(filename, dll_vali, fold_n);
		filename = dir2 + "/" + allFile_prefix + "prediction_test.txt";
		FileUtils.write2File(filename, dll_test, fold_n);
		//----------------------------------------------------------
		String perlResult_dir = "perlEvaluate/originalLR/" + fold_n;
		makeDir(perlResult_dir);
		SimpleDateFormat sdf = new SimpleDateFormat("yy-MM-dd-HH-mm-ss");
		String timeStamp = sdf.format(new Date())+"-";
		String perlResult_filename = perlResult_dir	+ "/" + allFile_prefix + timeStamp+ "test.txt";
		String[] perl_cmd = {
				"perl",
				"perlEvaluate/eval-score-mslr.pl",
				"data/OHSUMED/OHSUMED/QueryLevelNorm/" + fold_n + "/test.txt",
				"output_data/originalLR/prediction/" + fold_n + "/" + allFile_prefix
						+ "prediction_test.txt", 
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
		
		System.out.println("evaluate process over");
	}
	
	public Vector learn(List<RankList> train) throws InterruptedException{
		List<PartialPairList> ppll = getPartialPairForAllQueries(train);
		List<RankList> rll = train;		
		long startTime = 0;
		long endTime = 0;
		System.out.println(new Date());
		startTime=System.currentTimeMillis();
		double Jfun_pre = Double.MAX_VALUE-1;
		double Jfun_new = Double.MAX_VALUE;
		double Jfun_pre2;
		Vector w = new Vector();
		w.randomize();
//		Jfun_pre = Obj_Jfun_originalLogistic(ppll, w);
		Jfun_pre = Obj_Jfun_parallelOriginalLogistic(ppll, w);
		Jfun_new = Jfun_pre;
		int roundCount = 0;
		SimpleDateFormat sdf = new SimpleDateFormat("yy-MM-dd-HH-mm");
		String date_timeStamp = sdf.format(new Date()) + "-";
		String dir = "output_data/originalLR/inlearning_w/"+fold_n;
		makeDir(dir);
		boolean isAmplifyLearningRate = false;
		do{	
			
			Jfun_pre = Jfun_new;
//			Vector gradient = derivate_w(ppll,w);
			Vector gradient = derivate_w_parallel(ppll, w);
			if(isAmplifyLearningRate){
				this.learningRate *= 1.05;
				isAmplifyLearningRate = false;
			}
			Vector tem_w = Vector.addition(w, Vector.multiply(-(this.learningRate) ,gradient));
			Jfun_new = Obj_Jfun_originalLogistic(ppll, tem_w);			
			if(Jfun_new<Jfun_pre){
				roundCount++;				
				w = tem_w; 
				if(roundCount%3==0){
					System.out.println("Jfun_pre is:" + Jfun_pre);
					System.out.println("Jfun_new is:" + Jfun_new);	
					isAmplifyLearningRate = true;
					FileUtils.write2File(dir+ date_timeStamp +"/w.txt", w, "");
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
					if(roundCount%3==0){
						System.out.println("Jfun_pre is:" + Jfun_pre);
						System.out.println("Jfun_new is:" + Jfun_new);	
						isAmplifyLearningRate = true;
						FileUtils.write2File(dir+"/w.txt", w, "");
					}
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
