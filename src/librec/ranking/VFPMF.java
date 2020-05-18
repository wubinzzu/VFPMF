package librec.ranking;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Date;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.VectorEntry;
import librec.intf.IterativeRecommender;
import librec.util.FileIO;
import librec.util.Logs;
import librec.util.Strings;

/**
 * Created by wubin  on 2017/10/13.
 */
@Configuration(" alpha, factors, regU, regI,beta,gama,lamutaE, numIters")
public class VFPMF extends IterativeRecommender {
	// private float alpha;
	// item confidence
	// private SparseMatrix w, q;
	double alpha, beta, gama, lamutaE;
	// SparseVector[] userItemList, itemUserList, itemrelatedList,
	// relateditemList;
	String functionalfile;
	String visualfile;
	SparseMatrix itemfeatures;
	DenseVector D;
	int numvisualfactors;
	DenseMatrix E, Z, QS;
	public SparseMatrix functionalmatrix;

	public VFPMF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception {
		super(trainMatrix, testMatrix, fold);
		isRankingPred = true; // item recommendation
		alpha = 10;
		beta = algoOptions.getDouble("-beta");
		gama = algoOptions.getDouble("-gama");
		lamutaE = algoOptions.getDouble("-lamutaE", 1000);
		numvisualfactors = 4096;
		visualfile = cf.getPath("dataset.visual");
		functionalfile = cf.getPath("dataset.functional");
		functionalmatrix = this.getboughttogether();
		itemfeatures = this.getVisualFactors();
	}

	protected void initModel() throws Exception {
		super.initModel();
		P.init(0, 0.01);
		Q.init(0, 0.01);
		Z = new DenseMatrix(numItems, numFactors);
		Z.init(0, 0.01);
		E = new DenseMatrix(numvisualfactors, numFactors);
		E.init(0, 0.01);
		QS = new DenseMatrix(numFactors, numItems);
		D = new DenseVector(numvisualfactors);
		for (int d = 0; d < numvisualfactors; d++) {
			for (int j = 0; j < numItems; j++) {
				D.set(d, D.get(d) + itemfeatures.get(d, j) * itemfeatures.get(d, j));
			}
		}
	}

	@Override
	protected void buildModel() throws Exception {
		// Init caches
		double[] prediction_users = new double[numUsers];
		double[] prediction_items = new double[numItems];
		double[] prediction_itemrelated = new double[numItems];
		double[] prediction_relateditem = new double[numItems];
		double[] w_users = new double[numUsers];
		double[] w_items = new double[numItems];
		double[] q_itemrelated = new double[numItems];
		double[] q_relateditem = new double[numItems];

		// Init Sq
		DenseMatrix Sq = new DenseMatrix(numFactors, numFactors);
		// Init Sp
		DenseMatrix Sp, Sz;
		for (int iter = 1; iter <= numIters; iter++) {
			Logs.debug("{}{} runs at iteration = {} {}", algoName, foldInfo, iter, new Date());
			// Update the Sq cache
			Sq = Q.transMult();
			// Step 1: update user factors;
			for (int u = 0; u < numUsers; u++) {
				SparseVector row = trainMatrix.row(u);
				for (VectorEntry entry : row) {
					int i = entry.index();
					prediction_items[i] = DenseMatrix.rowMult(P, u, Q, i);
					w_items[i] = 1.0 + alpha * entry.get();
				}
				for (int f = 0; f < numFactors; f++) {
					double numer = 0, denom = regU + Sq.get(f, f);

					for (int k = 0; k < numFactors; k++) {
						if (f != k) {
							numer -= P.get(u, k) * Sq.get(f, k);
						}
					}
					double puf = P.get(u, f);
					for (VectorEntry entry : row) {
						int i = entry.index();
						// for (int i : indexes) {
						double qif = Q.get(i, f);
						prediction_items[i] -= puf * qif;
						numer += (w_items[i] - (w_items[i] - 1) * prediction_items[i]) * qif;
						denom += (w_items[i] - 1) * qif * qif;
					}
					// update puf
					puf = numer / denom;
					P.set(u, f, puf);
					for (VectorEntry entry : row) {
						int i = entry.index();
						// for (int i : indexes) {
						prediction_items[i] += puf * Q.get(i, f);
					}
				}
			}
			// Update the Sp cache
			Sp = P.transMult();
			Sz = Z.transMult();

			DenseMatrix ETF = E.transpose().mult(itemfeatures);// numFactors X numItems

			// Step 2: update item factors;
			for (int i = 0; i < numItems; i++) {
				SparseVector column = trainMatrix.column(i);
				SparseVector functional = functionalmatrix.row(i);
				// SparseVector column = itemUserList[i];
				// SparseVector related = itemrelatedList[i];
				// int[] uIndexes = new int[column.size()];
				// int[] gIndexes = new int[related.size()];
				// int number = 0;
				for (VectorEntry entry : column) {
					int u = entry.index();
					// uIndexes[number++] = u;
					prediction_users[u] = DenseMatrix.rowMult(P, u, Q, i);
					w_users[u] = 1.0 + alpha * entry.get();
				}
				// number = 0;
				for (VectorEntry entry : functional) {
					int g = entry.index();
					// gIndexes[number++] = g;
					prediction_itemrelated[g] = DenseMatrix.rowMult(Q, i, Z, g);
					q_itemrelated[g] = 1.0 + alpha * entry.get();
				}
				for (int f = 0; f < numFactors; f++) {
					double numer = 0, denom = Sp.get(f, f) + regI;
					double numer1 = 0, denom1 = Sz.get(f, f);
					for (int k = 0; k < numFactors; k++) {
						if (f != k) {
							numer -= Q.get(i, k) * Sp.get(k, f);
							numer1 -= Q.get(i, k) * Sz.get(k, f);
						}
					}
					double qif = Q.get(i, f);
					for (VectorEntry entry : column) {
						int u = entry.index();
						// for (int u : uIndexes) {
						double puf = P.get(u, f);
						prediction_users[u] -= puf * qif;
						numer += (w_users[u] - (w_users[u] - 1) * prediction_users[u]) * puf;
						denom += (w_users[u] - 1) * puf * puf;
					}
					for (VectorEntry entry : functional) {
						int g = entry.index();
						// for (int g : gIndexes) {
						double zgf = Z.get(g, f);
						prediction_itemrelated[g] -= zgf * qif;
						numer1 += (q_itemrelated[g] - (q_itemrelated[g] - 1) * prediction_itemrelated[g]) * zgf;
						denom1 += (q_itemrelated[g] - 1) * zgf * zgf;
					}
					// update qif
					qif = (numer + numer1 * beta + gama * ETF.get(f, i)) / (denom + denom1 * beta + gama);
					Q.set(i, f, qif);
					for (VectorEntry entry : column) {
						int u = entry.index();
						// for (int u : uIndexes) {
						prediction_users[u] += P.get(u, f) * qif;
					}
					for (VectorEntry entry : functional) {
						int g = entry.index();
						// for (int g : gIndexes) {
						prediction_itemrelated[g] += Z.get(g, f) * qif;
					}
				}
			}
			Sq = Q.transMult();
			// Step 1: update Z factors;
			for (int g = 0; g < numItems; g++) {
				SparseVector item = functionalmatrix.column(g);
				// SparseVector item = relateditemList[g];
				// int[] iIndexes = new int[item.size()];
				// int number = 0;
				for (VectorEntry entry : item) {
					int i = entry.index();
					// iIndexes[number++] = i;
					prediction_relateditem[i] = DenseMatrix.rowMult(Q, i, Z, g);
					q_relateditem[i] = 1.0 + alpha * entry.get();
				}
				for (int f = 0; f < numFactors; f++) {
					double numer = 0, denom = Sq.get(f, f);
					for (int k = 0; k < numFactors; k++) {
						if (f != k) {
							numer -= Z.get(g, k) * Sq.get(f, k);
						}
					}
					double zgf = Z.get(g, f);
					for (VectorEntry entry : item) {
						int i = entry.index();
						// for (int i : iIndexes) {
						double qif = Q.get(i, f);
						prediction_relateditem[i] -= zgf * qif;
						numer += (q_relateditem[i] - (q_relateditem[i] - 1) * prediction_relateditem[i]) * qif;
						denom += (q_relateditem[i] - 1) * qif * qif;
					}
					// update puf
					zgf = beta * numer / (beta * denom + regI);
					Z.set(g, f, zgf);
					for (VectorEntry entry : item) {
						int i = entry.index();
						// for (int i : iIndexes) {
						prediction_relateditem[i] += zgf * Q.get(i, f);
					}
				}
			}

			DenseMatrix Y = ETF.clone();
			for (int k = 0; k < numFactors; k++) {
				for (int d = 0; d < numvisualfactors; d++) {
					SparseVector vector = itemfeatures.row(d);
					double numer = 0.0;
					double edk = E.get(d, k);
					for (VectorEntry entry : vector) {
						double idj = entry.get();
						int j = entry.index();
						Y.set(k, j, Y.get(k, j) - edk * idj);
						numer += (Q.get(j, k) - Y.get(k, j)) * idj;
					}
					edk = numer * gama / (gama * D.get(d) + lamutaE);
					E.set(d, k, edk);
					for (VectorEntry entry : vector) {
						double idj = entry.get();
						int j = entry.index();
						Y.set(k, j, Y.get(k, j) + edk * idj);
					}
				}
			}
			QS=E.transpose().mult(itemfeatures);
		}
	}

	public SparseMatrix getboughttogether() throws IOException {
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		Multimap<Integer, Integer> colMap = HashMultimap.create();
		BufferedReader br = FileIO.getReader(functionalfile);
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] itemrelations = line.split(",");
			String realitemid = itemrelations[0];
			if (rateDao.getItemIds().containsKey(realitemid)) {
				int inneritemid = rateDao.getItemIds().get(realitemid);
				for (int i = 1; i < itemrelations.length; i++) {
					if (rateDao.getItemIds().containsKey(itemrelations[i])) {
						int relatedinneriid = rateDao.getItemIds().get(itemrelations[i]);
						dataTable.put(inneritemid, relatedinneriid, 1.0);
						colMap.put(relatedinneriid, inneritemid);
					}
				}
			}
		}
		br.close();
		SparseMatrix itemrelatedmatrix = new SparseMatrix(numItems, numItems, dataTable, colMap);
		return itemrelatedmatrix;
	}
	public SparseMatrix getVisualFactors() throws Exception {
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		Multimap<Integer, Integer> colMap = HashMultimap.create();
		BufferedReader br = FileIO.getReader(visualfile);
		String line = null;
		float max = 0;
		float min = 10;
		while ((line = br.readLine()) != null) {
			String[] itemfeatures = line.split(",");
			String realitemid = itemfeatures[0];
			if (rateDao.getItemIds().containsKey(realitemid)) {
				int inneritemid = rateDao.getItemIds().get(realitemid);
				for (int i = 0; i < numvisualfactors; i++) {
					float value = Float.parseFloat(itemfeatures[i + 1]);
					if (value > 0.0) {
						dataTable.put(i, inneritemid, Double.parseDouble(String.valueOf(value)));
						colMap.put(inneritemid, i);
						if (value > max) {
							max = value;
						}
						if (value < min) {
							min = value;
						}
					}
				}
			}
		}
		br.close();
		SparseMatrix itemfeaturematrix = new SparseMatrix(numvisualfactors, numItems, dataTable, colMap);
		itemfeaturematrix.normalize(min, max);
		return itemfeaturematrix;
	}

	@Override
	protected double predict(int u, int j) throws Exception {
		double rating = 0.0;
		if (trainMatrix.column(j).size() == 0) {
			rating = P.row(u).inner(QS.column(j));
			// System.out.println(rating);
		} else {
			rating = DenseMatrix.rowMult(P, u, Q, j);
		}
		return rating;
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { alpha, numFactors, regU, regI, beta, gama, lamutaE, numIters }, ",");
	}
}
