package ciir.umass.edu.learning;

public class Vector {
	private double [] Vec = null;
	public double[] getVec() {
		return Vec;
	}
	public void setVec(double[] vec) {
		Vec = vec;
	}
		// public members
		public static int VectorSize = 5;
		public static int getVectorSize() {
			return VectorSize;
		}
		public static void setVectorSize(int vectorSize) {
			VectorSize = vectorSize;//index starts from 1,
		}
		
		public Vector(){
			Vec = new double[VectorSize];
			for (int i = 0; i < Vector.VectorSize; i++) {
				Vec[i] = 0;
			}
			
		}
		public Vector(int vectorSize){
			Vector.VectorSize = vectorSize;
			Vec = new double[VectorSize];
			for (int i = 0; i < Vector.VectorSize; i++) {
				Vec[i] = 0;
			}
			
		}
		public void randomize(){
			for (int i = 0; i < Vector.VectorSize; i++) {
				Vec[i] = Math.random();
			}
		}
		public Vector(float[] fs){
			Vec = new double[VectorSize+1];
			for (int i = 0; i < Vector.VectorSize; i++) {
				Vec[i] = fs[i]; 
			}
			
		}
		public Vector(double[] fs){
			Vec = new double[VectorSize+1];
			for (int i = 0; i < Vector.VectorSize; i++) {
				Vec[i] = fs[i]; 
			}
			
		}
		public static double dotProduct(Vector v1, Vector v2){
			double result = 0;
			for (int i = 0; i < Vector.VectorSize; i++) {
				result += v1.getVec()[i] * v2.getVec()[i];				
			}
			return result;
		}
		
		public static Vector multiply(double coefficient, Vector v){
			Vector v2 = new Vector();
			for (int i = 0; i < Vector.VectorSize; i++) {
				v2.Vec[i] = v.Vec[i] * coefficient;
			}
			return v2;
		}
		public static Vector addition(Vector v1, Vector v2){
			Vector v = new Vector();
			for (int i = 0; i < Vector.VectorSize; i++) {
				v.Vec[i] = v1.Vec[i] + v2.Vec[i];
			}
			return v;
		}
		public void assignment(Vector v){
			for (int i = 0; i < Vector.VectorSize; i++) {
				this.Vec[i] = v.Vec[i];
			}
		}
		
}
