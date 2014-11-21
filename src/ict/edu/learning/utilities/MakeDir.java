package ict.edu.learning.utilities;

import java.io.File;

public class MakeDir {
	public static String makeDir(String tail) {  
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
}
