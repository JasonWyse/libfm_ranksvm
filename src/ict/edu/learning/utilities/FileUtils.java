package ict.edu.learning.utilities;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ciir.umass.edu.learning.Matrix;
import ciir.umass.edu.learning.Vector;

public class FileUtils {

	public static boolean write2File(String filename , Matrix m, String description)  {
		//BufferedWriter out = null;
		StringBuffer sb = new StringBuffer();
		sb.append("# " + description + System.getProperty("line.separator"));
		try {  
            // 打开一个写文件器，构造函数中的第二个参数true表示以追加形式写文件  
			File file=new File(filename);    
			if(!file.exists())    
			{    
			    try {    
			        file.createNewFile();    
			    } catch (IOException e) {    
			        // TODO Auto-generated catch block    
			        e.printStackTrace();    
			    }    
			}    
			FileWriter writer = new FileWriter(filename, true); 
            for (int i = 0; i < Matrix.RowsOfVMatrix; i++) {
				for (int j = 0; j < Matrix.ColsOfVMatrix; j++) {
					sb.append(m.getV()[i][j]).append("\t");
				}
				sb.append(System.getProperty("line.separator"));
			}
            sb.append(System.getProperty("line.separator"));
            
            writer.write(sb.toString());  
            writer.close();  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
		return true;
	}
	public static boolean write2File(String filename , List<ArrayList<Double>> m, String description)  {
		//BufferedWriter out = null;
		StringBuffer sb = new StringBuffer();
//		sb.append("# " + description + System.getProperty("line.separator"));
		try {  
			File file=new File(filename);    
			if(!file.exists())    
			{    
			    try {    
			        file.createNewFile();    
			    } catch (IOException e) {    
			        // TODO Auto-generated catch block    
			        e.printStackTrace();    
			    }    
			}    
            // 打开一个写文件器，构造函数中的第二个参数true表示以追加形式写文件  
            FileWriter writer = new FileWriter(filename, true); 
            for (int i = 0; i < m.size(); i++) {
				for (int j = 0; j < m.get(i).size(); j++) {
					sb.append(m.get(i).get(j)).append(System.getProperty("line.separator"));
				}				
			}  
            sb.append(System.getProperty("line.separator"));
            writer.write(sb.toString());  
            writer.close();  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
		return true;
	}
	public static boolean write2File(String filename , Vector v, String description)  {
		//BufferedWriter out = null;
		StringBuffer sb = new StringBuffer();
		sb.append("# " + description + System.getProperty("line.separator"));
		try {  
            // 打开一个写文件器，构造函数中的第二个参数true表示以追加形式写文件  
			File file=new File(filename);    
			if(!file.exists())    
			{    
			    try {    
			        file.createNewFile();    
			    } catch (IOException e) {    
			        // TODO Auto-generated catch block    
			        e.printStackTrace();    
			    }    
			}    
            FileWriter writer = new FileWriter(filename, true); 
            for (int i = 0; i < Vector.getVectorSize(); i++) {				
					sb.append(v.getVec()[i]).append("\t");					
			}
            sb.append(System.getProperty("line.separator"));
            
            writer.write(sb.toString());  
            writer.close();  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
		return true;
	}
	public static boolean write2File(String filename , StringBuffer sb, String description)  {
		//BufferedWriter out = null;
		
		try {  
            // 打开一个写文件器，构造函数中的第二个参数true表示以追加形式写文件  
			File file=new File(filename);    
			if(!file.exists())    
			{    
			    try {    
			        file.createNewFile();    
			    } catch (IOException e) {    
			        // TODO Auto-generated catch block    
			        e.printStackTrace();    
			    }    
			}    
            FileWriter writer = new FileWriter(filename, true); 
            writer.write(sb.toString());  
            writer.close();  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
		return true;
	}
	public static Matrix readFromFileGetMatrix(String filename) throws IOException{
		Matrix  m = new Matrix();
		FileReader fr = new FileReader(filename);
		BufferedReader br= new BufferedReader(fr);		 
		String lineContent;
		int matrixRow_index = 0;
		 while((lineContent = br.readLine()) != null) {
			 if (lineContent.equals("")||lineContent.contains("#")) {
				continue;
			 }
			 String [] str = lineContent.split("\t");
			 for (int i = 0; i < str.length; i++) {				
				m.getV()[matrixRow_index][i] = Double.parseDouble(str[i]);
			 }
			 if(matrixRow_index>=99)
				 System.out.println(matrixRow_index);
			 matrixRow_index++;
		  }
		return m;
	}
	public static List<Matrix> readFromFileGetMatrixList(String filename) throws IOException{
		List<Matrix> ml= new ArrayList<Matrix>();		
		Matrix  m = null;
		FileReader fr = new FileReader(filename);
		BufferedReader br= new BufferedReader(fr);		 
		String lineContent;
		int matrixNum = 0;
		while((lineContent = br.readLine()) != null) {
			 if (lineContent.contains("#")) {
				 matrixNum++;
			 }			 
		  }
		matrixNum--;//the last one may be incomplete,so remove it from the list
		br.close();
		fr.close();
		fr = new FileReader(filename);
		br = new BufferedReader(fr);		
		int matrixRow_index = 0;
		int matrixCount = 0;
		while ((lineContent = br.readLine()) != null) {
			if (lineContent.contains("#")) {				
				if(matrixCount>0){
					ml.add(m);
					matrixRow_index = 0;
				}					
				if (matrixCount == matrixNum)
					break;
				m = new Matrix();
				continue;
			}
			if (lineContent.equals("")){
				matrixCount++;
				continue;
			}
			String[] str = lineContent.split("\t");
			for (int j = 0; j < str.length; j++) {
				m.getV()[matrixRow_index][j] = Double.parseDouble(str[j]);
			}
			/*
			 * if(matrixRow_index>=99) System.out.println(matrixRow_index);
			 */
			matrixRow_index++;
		}
		
		
		
		return ml;
	}
	public static Vector readFromFileGetVector(String filename) throws NumberFormatException, IOException{
		Vector v = new Vector();
		FileReader fr = new FileReader(filename);
		BufferedReader br= new BufferedReader(fr);		 
		String lineContent;
		while((lineContent = br.readLine()) != null) {
			 if (lineContent=="") {
				continue;
			 }
			 String [] str = lineContent.split("\t");
			 for (int i = 0; i < str.length; i++) {				
				v.getVec()[i] = Double.parseDouble(str[i]);
			 }			 
		}
		return v;
	}
 public static void main(String[] args){
	 String filename  = "MatrixV.txt";
	 for (int i = 0; i < 2; i++) {
		 Matrix m = new Matrix();
		 FileUtils.write2File(filename, m , null);
		
	}
	 
 }
}