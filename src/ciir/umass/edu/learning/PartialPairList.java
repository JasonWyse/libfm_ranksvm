package ciir.umass.edu.learning;

import java.util.ArrayList;
import java.util.List;

import ciir.umass.edu.utilities.Sorter;

public class PartialPairList {
	protected List<PartialPair> ppl = null;
	
	public PartialPairList()
	{
		ppl = new ArrayList<PartialPair>();
	}
	public PartialPairList(PartialPairList ppl)
	{
		this.ppl = new ArrayList<PartialPair>();
		for(int i=0;i<ppl.size();i++)
			this.ppl.add(ppl.get(i));
	}
	public PartialPairList(PartialPairList ppl, int[] idx)
	{
		this.ppl = new ArrayList<PartialPair>();
		for(int i=0;i<idx.length;i++)
			this.ppl.add(ppl.get(idx[i]));
	}
	
	public String getID()
	{
		return get(0).getPartialPairID();
	}
	public int size()
	{
		return ppl.size();
	}
	public PartialPair get(int k)
	{
		return ppl.get(k);
	}
	
	public void add(PartialPair p)
	{
		ppl.add(p);
	}
	public void remove(int k)
	{
		ppl.remove(k);
	}
	
	

	
}
