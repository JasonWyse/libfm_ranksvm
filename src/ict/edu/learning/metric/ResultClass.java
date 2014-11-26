package ict.edu.learning.metric;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.Vector;

public class ResultClass {
	public Matrix matrix = null;
	public Vector vector = null;
	public String timeStamp = null;
	public ResultClass(Matrix m, Vector v, String timeStamp) {
		super();
		this.matrix = m;
		this.vector = v;
		this.timeStamp = timeStamp;
	}
	public ResultClass(Matrix m,  String timeStamp) {
		super();
		this.matrix = m;		
		this.timeStamp = timeStamp;
	}
	public ResultClass(Vector v,  String timeStamp) {
		super();
		this.vector = v;		
		this.timeStamp = timeStamp;
	}
	public Matrix getMatrix() {
		return matrix;
	}
	public void setMatrix(Matrix matrix) {
		this.matrix = matrix;
	}
	public Vector getVector() {
		return vector;
	}
	public void setVector(Vector vector) {
		this.vector = vector;
	}

	public String getTimeStamp() {
		return timeStamp;
	}

	public void setTimeStamp(String timeStamp) {
		this.timeStamp = timeStamp;
	}

	
}
