package ciir.umass.edu.learning;

public class Matrix {
	private double [][] V = null;
// public members
	public static int RowsOfVMatrix = 9219;
	public static int ColsOfVMatrix = 5;
	public static int ROW_INCREASE = 20;
	
	public double[][] getV() {
		return V;
	}
	public void setV(double[][] v) {
		V = v;
	}
	public static int getRowsOfVMatrix() {
		return RowsOfVMatrix;
	}
	public static void setRowsOfVMatrix(int rowsOfVMatrix) {
		RowsOfVMatrix = rowsOfVMatrix;
	}
	public static int getColsOfVMatrix() {
		return ColsOfVMatrix;
	}
	public static void setColsOfVMatrix(int colsOfVMatrix) {
		ColsOfVMatrix = colsOfVMatrix;
	}
	public static int getROW_INCREASE() {
		return ROW_INCREASE;
	}
	public static void setROW_INCREASE(int rOW_INCREASE) {
		ROW_INCREASE = rOW_INCREASE;
	}
	public Matrix(){
		V = new double[Matrix.RowsOfVMatrix][Matrix.ColsOfVMatrix];
	}
	public Matrix(int rowsSize, int colSize){
		Matrix.RowsOfVMatrix = rowsSize;
		Matrix.ColsOfVMatrix = colSize;
		V = new double[Matrix.RowsOfVMatrix][Matrix.ColsOfVMatrix];
	}
	public Matrix(Matrix m){
		V = new double[Matrix.RowsOfVMatrix][Matrix.ColsOfVMatrix];
		for (int i = 0; i < Matrix.RowsOfVMatrix; i++) {
			for (int j = 0; j < Matrix.ColsOfVMatrix; j++) {
				this.V[i][j] = m.V[i][j];
			}
		}		
	}
	public void assignment(Matrix m){
		for (int i = 0; i < Matrix.RowsOfVMatrix; i++) {
			for (int j = 0; j < Matrix.ColsOfVMatrix; j++) {
				this.V[i][j] = m.V[i][j];
			}
		}		
	}
	
	public void randomize(){
		for (int i = 0; i < Matrix.RowsOfVMatrix; i++) {
			for (int j = 0; j < Matrix.ColsOfVMatrix; j++) {
				this.V[i][j] = (double) Math.random();
			}
		}
				
	}
	public void update(double[] V_ac, int rowIndex){
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			this.V[rowIndex][i] = V_ac[i];
		}
	}
	public double[] getRowVector(int k){
		return V[k];
	}
	public void setRowVector(double[] v, int rowNum){
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			this.V[rowNum][i] =v[i];
		}
	}
	public static void SetRowVector(double[] v1, double[] v2){
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			v1[i] = v2[i];
		}
	}
	public void additionMatrix(Matrix m){
		for (int i = 0; i < Matrix.RowsOfVMatrix; i++) {
			additionRowVector(i, m.getRowVector(i));
		}
	}
	public static Matrix multiplyCoefficient(double eta, Matrix m){
		Matrix mat = new Matrix();
		for (int i = 0; i < Matrix.RowsOfVMatrix; i++) {
			for (int j = 0; j < Matrix.ColsOfVMatrix; j++) {
				mat.V[i][j] = m.V[i][j] * eta;
			}
			
		}
		return mat;
	}
	
	public void additionRowVector(int rowNum, double[] v){
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			this.V[rowNum][i] +=v[i];
		}
	}
	public  double[] multiplyRowVector(int rowNum, double multiplier, double[] result){
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			result[i] = this.V[rowNum][i] * multiplier;
		}
		return result;
	}
	public  static double[]  multiplyRowVector( double factor1, double[] v){
		double[] result = new double[v.length];
		for (int i = 0; i < v.length; i++) {
			result[i] = (double) (v[i] * factor1);
		}
		return result;
	}
	public static void RowVectorAddition(double[] array1, double[] array2){
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			array1[i] += array2[i]; 
		}
		
	}
	public double getInnerProduct(int V_iq,  int V_jq){
		double innerProduct = 0;
		for (int i = 0; i < Matrix.ColsOfVMatrix; i++) {
			innerProduct += this.V[V_iq][i]*this.V[V_jq][i];
		}
		return innerProduct;
	}
}
