package ciir.umass.edu.learning;

public class PartialPair {
	public static int MAX_FEATURE = 51;
	public static int featureCount = 0;
	protected double[] partialFVals = null;//partialFVals[0] is un-used. Feature id MUST start from 1
	double largeLabel = 0.0f;
	double smallLabel =0.0f;
	protected String queryID = null;
	protected String largeDocID = null;
	protected String smallDocID = null;
	protected String partialPairID = null;
	public PartialPair(DataPoint dp1,DataPoint dp2)
	{
		if (dp1.id.compareTo(dp2.id)!=0) {
			System.out.println("cann't compare two docs within two different queries. "+"doc1_query = "
		+dp1.id+" doc2_query = "+"dp2.id");
			return;
		}		
		setFeatureCount(DataPoint.featureCount);
		partialFVals = new double[PartialPair.featureCount+1];// index starts from 1
		setQueryID(dp1.id);
		if (dp1.getLabel()>dp2.getLabel()) {
			for (int i = 0; i <=PartialPair.featureCount; i++) {
				partialFVals[i]=dp1.fVals[i]-dp2.fVals[i];
			}
			String doc1ID = dp1.description.substring(dp1.description.lastIndexOf("=")+1).trim();
			String doc2ID = dp2.description.substring(dp2.description.lastIndexOf("=")+1).trim();
			partialPairID = dp1.id+"-" + doc1ID + "-" + doc2ID;
			this.largeLabel = dp1.getLabel();
			setLargeDocID(doc1ID);
			this.smallLabel = dp2.getLabel();
			setSmallDocID(doc2ID);
			
		} else if(dp1.getLabel()<dp2.getLabel()) {
			for (int i = 0; i <=PartialPair.featureCount; i++) {
				partialFVals[i]=dp2.fVals[i]-dp1.fVals[i];
			}
			String doc1ID = dp1.description.substring(dp1.description.lastIndexOf("=")+1).trim();
			String doc2ID = dp2.description.substring(dp2.description.lastIndexOf("=")+1).trim();
			partialPairID = dp1.id+"-" + doc2ID + "-" + doc1ID;
			this.largeLabel = dp2.getLabel();
			setLargeDocID(doc2ID);
			this.smallLabel = dp1.getLabel();
			setSmallDocID(doc1ID);

		}
		else{
			System.out.println("two docs with same labels under the same query cann't be compared");
			return;
		}		
	}
	public double dotProduct(PartialPair pp){
		double innerProduct = 0f;
		for (int i = 1; i <= PartialPair.featureCount; i++) {
			innerProduct += this.partialFVals[i] * pp.partialFVals[i];
		}
		return innerProduct;
	}
	public String getQueryID() {
		return queryID;
	}
	public void setQueryID(String queryID) {
		this.queryID = queryID;
	}
	public String getLargeDocID() {
		return largeDocID;
	}
	public void setLargeDocID(String largeDocID) {
		this.largeDocID = largeDocID;
	}
	public String getSmallDocID() {
		return smallDocID;
	}
	public void setSmallDocID(String smallDocID) {
		this.smallDocID = smallDocID;
	}
	public double getLargeLabel() {
		return largeLabel;
	}
	public void setLargeLabel(double largeLabel) {
		this.largeLabel = largeLabel;
	}
	public double getSmallLabel() {
		return smallLabel;
	}
	public void setSmallLabel(double smallLabel) {
		this.smallLabel = smallLabel;
	}
	public static int getFeatureCount() {
		return featureCount;
	}
	public static void setFeatureCount(int featureCount) {
		PartialPair.featureCount = featureCount;
	}
	public double[] getPartialFVals() {
		return partialFVals;
	}
	public void setPartialFVals(double[] partialFVals) {
		this.partialFVals = partialFVals;
	}
	
	public String getPartialPairID() {
		return partialPairID;
	}
	public void setPartialPairID(String partialPairID) {
		this.partialPairID = partialPairID;
	}
}
