// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.intf;

import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.cache.LoadingCache;
import librec.data.*;
import librec.util.*;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * General recommenders
 *
 * @author wubin
 */
@Configuration
public abstract class Recommender implements Runnable {

    /**
     * Recommendation measures
     */
    public enum Measure {
        Loss, /* prediction-based measures */
        MAP10,NDCG10,Pre10,Rec10,TestTime,
        /* execution time */
        TrainTime
    }
    // threshold to binarize ratings
    public static float binThold;

    // Guava cache configuration
    protected static String cacheSpec;

    /************************************
     * Static parameters for all recommenders
     ***********************************/
    // configer
    public static FileConfiger cf;

    // early-stop criteria
    protected static Measure earlyStopMeasure = null;

    // init mean and standard deviation
    protected static double initMean, initStd;

    // is diversity-based measures used
    protected static boolean isDiverseUsed;

    // is ranking/rating prediction
    public static boolean isRankingPred;

    // is save model
    protected static boolean isSaveModel = false;
    // is split data by date
    protected static boolean isSplitByDate;
    // number of nearest neighbors
    protected static int knn;
    // Maximum, minimum values of rating scales
    protected static double maxRate, minRate;
    // minimum, maximum timestamp
    protected static long minTimestamp, maxTimestamp;
    // number of cpu cores used for parallelization
    protected static int numCPUs;
    // number of rating levels
    protected static int numLevels;
    // number of recommended items
    protected static int numRecs, numIgnore;

    // number of users, items, ratings
    protected static int numUsers, numItems, numRates;

    // params used for multiple runs
    public static Map<String, List<Float>> params = new HashMap<>();
    // line configer for item ranking, evaluation
    protected static LineConfiger rankOptions, algoOptions;

    // rate DAO object
    public static DataDAO rateDao;
    // matrix of rating data
    public static SparseMatrix rateMatrix, timeMatrix;
    // a list of rating scales
    protected static List<Double> ratingScale;

    /**
     * An indicator of initialization of static fields. This enables us to control when static fields are initialized,
     * while "static block" will be always initialized or executed. The latter could cause unexpected exceptions when
     * multiple runs (with different configuration files) are conducted sequentially, because some static settings will
     * not be override in such a "staic block".
     */
    public static boolean resetStatics = true;
    // similarity measure
    protected static String similarityMeasure;

    // number of shrinkage
    protected static int similarityShrinkage;
    // small value for initialization
    protected static double smallValue = 0.01;

    // default temporary file directory
    public static String tempDirPath;
    // ratings' timestamps
    public static SparseMatrix testTimeMatrix;
    // the ratio of validation data split from training data
    public static float validationRatio;

    // verbose
    protected static boolean verbose = true;

    // view of rating predictions
    public static String view;
    /**
     * @return the evaluation information of a recommend
     */
    public static String getEvalInfo(Map<Measure, Double> measures) {
        String evalInfo = String.format("Precision@10=%.6f,Recall@10=%.6f,MAP@10=%.6f,NDCG@10=%.6f",
        		measures.get(Measure.Pre10),
        	    measures.get(Measure.Rec10),
        		measures.get(Measure.MAP10),
        		measures.get(Measure.NDCG10));

        return evalInfo;
    }
    /************************************
     * Recommender-specific parameters
     ****************************************/
    // algorithm's name
    public String algoName;
    // upper symmetric matrix of item-item correlations
    protected SymmMatrix corrs;

    // current fold
    protected int fold;

    // fold information
    protected String foldInfo;

    // global average of training rates
    protected double globalMean;

    // is output recommendation results
    protected boolean isResultsOut = true;

    // performance measures
    public Map<Measure, Double> measures;
    // rating matrix for training, validation and test
    protected SparseMatrix trainMatrix, validationMatrix, testMatrix;

    // user-vector cache, item-vector cache
    protected LoadingCache<Integer, SparseVector> userCache, itemCache;

    // user-items cache, item-users cache
    protected LoadingCache<Integer, List<Integer>> userItemsCache, itemUsersCache;

    /**
     * Constructor for Recommender
     *
     * @param trainMatrix train matrix
     * @param testMatrix  test matrix
     * @throws FileNotFoundException 
     */
    public Recommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {

        // config recommender
        if (cf == null || rateMatrix == null) {
            Logs.error("Recommender is not well configed");
            System.exit(-1);
        }

        // static initialization (reset), only done once
        if (resetStatics) {
            // change the indicator
            resetStatics = false;

            ratingScale = rateDao.getRatingScale();
            minRate = ratingScale.get(0);
            maxRate = ratingScale.get(ratingScale.size() - 1);
            numLevels = ratingScale.size();

            numUsers = rateDao.numUsers();
            numItems = rateDao.numItems();

            // ratings' timestamps
            minTimestamp = rateDao.getMinTimestamp();
            maxTimestamp = rateDao.getMaxTimestamp();
            if (testTimeMatrix == null)
                testTimeMatrix = timeMatrix;

            initMean = 0.0;
            initStd = 0.1;

            cacheSpec = cf.getString("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");

            rankOptions = cf.getParamOptions("item.ranking");
            isRankingPred = rankOptions.isMainOn();
            isDiverseUsed = rankOptions.contains("-diverse");
            numRecs = rankOptions.getInt("-topN", -1);
            numIgnore = rankOptions.getInt("-ignore", -1);

            LineConfiger evalOptions = cf.getParamOptions("evaluation.setup");
            view = evalOptions.getString("--test-view", "all");
            validationRatio = evalOptions.getFloat("-v", 0.0f);
            isSplitByDate = evalOptions.contains("--by-date");

            String earlyStop = evalOptions.getString("--early-stop");
            if (earlyStop != null) {
                for (Measure m : Measure.values()) {
                    if (m.name().equalsIgnoreCase(earlyStop)) {
                        earlyStopMeasure = m;
                        break;
                    }
                }
            }

            int numProcessors = Runtime.getRuntime().availableProcessors();
            numCPUs = evalOptions.getInt("-cpu", numProcessors);

            // output options
            LineConfiger outputOptions = cf.getParamOptions("output.setup");
            if (outputOptions != null) {
                verbose = outputOptions.isOn("-verbose", true);
                isSaveModel = outputOptions.contains("--save-model");
            }

            knn = cf.getInt("num.neighbors", 20);
            similarityMeasure = cf.getString("similarity", "PCC");
            similarityShrinkage = cf.getInt("num.shrinkage", 30);
        }

        // training, validation, test data
        if (validationRatio > 0 && validationRatio < 1) {
            DataSplitter ds = new DataSplitter(trainMatrix);
            double ratio = 1 - validationRatio;

            SparseMatrix[] trainSubsets = isSplitByDate ? ds.getRatioByRatingDate(ratio, timeMatrix) : ds
                    .getRatioByRating(ratio);
            this.trainMatrix = trainSubsets[0];
            this.validationMatrix = trainSubsets[1];
        } else {
            this.trainMatrix = trainMatrix;
        }

        this.testMatrix = testMatrix;

        // fold info
        this.fold = fold;
        foldInfo = fold > 0 ? " fold [" + fold + "]" : "";

        // whether to write out results
        LineConfiger outputOptions = cf.getParamOptions("output.setup");
        if (outputOptions != null) {
            isResultsOut = outputOptions.isMainOn();
        }

        // global mean
        numRates = trainMatrix.size();
        globalMean = trainMatrix.sum() / numRates;

        // class name as the default algorithm name
        setAlgoName(this.getClass().getSimpleName());

        // compute item-item correlations
        if (isRankingPred && isDiverseUsed)
            corrs = new SymmMatrix(numItems);
    }

    /**
     * build user-user or item-item correlation matrix from training data
     *
     * @param isUser whether it is user-user correlation matrix
     * @return a upper symmetric matrix with user-user or item-item coefficients
     */
    protected SymmMatrix buildCorrs(boolean isUser) {
        Logs.debug("Build {} similarity matrix ...", isUser ? "user" : "item");

        int count = isUser ? numUsers : numItems;
        SymmMatrix corrs = new SymmMatrix(count);

        for (int i = 0; i < count; i++) {
            SparseVector iv = isUser ? trainMatrix.row(i) : trainMatrix.column(i);
            if (iv.getCount() == 0)
                continue;
            // user/item itself exclusive
            for (int j = i + 1; j < count; j++) {
                SparseVector jv = isUser ? trainMatrix.row(j) : trainMatrix.column(j);

                double sim = correlation(iv, jv);

                if (!Double.isNaN(sim))
                    corrs.set(i, j, sim);
            }
        }

        return corrs;
    }



    /**
     * Learning method: override this method to build a model, for a model-based method. Default implementation is
     * useful for memory-based methods.
     */
    protected void buildModel() throws Exception {
    }

    /**
     * Check if ratings have been binarized; useful for methods that require binarized ratings;
     */
    protected void checkBinary() {
        if (binThold < 0) {
            Logs.error("val.binary.threshold={}, ratings must be binarized first! Try set a non-negative value.",
                    binThold);
            System.exit(-1);
        }
    }

    /**
     * Compute the correlation between two vectors using method specified by configuration key "similarity"
     *
     * @param iv vector i
     * @param jv vector j
     * @return the correlation between vectors i and j
     */
    protected double correlation(SparseVector iv, SparseVector jv) {
        return correlation(iv, jv, similarityMeasure);
    }

    /**
     * Compute the correlation between two vectors for a specific method
     *
     * @param iv     vector i
     * @param jv     vector j
     * @param method similarity method
     * @return the correlation between vectors i and j; return NaN if the correlation is not computable.
     */
    protected double correlation(SparseVector iv, SparseVector jv, String method) {

        // compute similarity
        List<Double> is = new ArrayList<>();
        List<Double> js = new ArrayList<>();

        for (Integer idx : jv.getIndex()) {
            if (iv.contains(idx)) {
                is.add(iv.get(idx));
                js.add(jv.get(idx));
            }
        }

        double sim = 0;
        switch (method.toLowerCase()) {
            case "cos":
                // for ratings along the overlappings
                sim = Sims.cos(is, js);
                break;
            case "cos-binary":
                // for ratings along all the vectors (including one-sided 0s)
                sim = iv.inner(jv) / (Math.sqrt(iv.inner(iv)) * Math.sqrt(jv.inner(jv)));
                break;
            case "msd":
                sim = Sims.msd(is, js);
                break;
            case "cpc":
                sim = Sims.cpc(is, js, (minRate + maxRate) / 2.0);
                break;
            case "exjaccard":
                sim = Sims.exJaccard(is, js);
                break;
            case "pcc":
            default:
                sim = Sims.pcc(is, js);
                break;
        }

        // shrink to account for vector size
        if (!Double.isNaN(sim)) {
            int n = is.size();
            int shrinkage = cf.getInt("num.shrinkage");
            if (shrinkage > 0)
                sim *= n / (n + shrinkage + 0.0);
        }

        return sim;
    }

    /**
     * denormalize a prediction to the region (minRate, maxRate)
     */
    protected double denormalize(double pred) {
        return minRate + pred * (maxRate - minRate);
    }

    /**
     * @param rankedItems the list of ranked items to be recommended
     * @param cutoff      cutoff in the list
     * @return diversity at a specific cutoff position
     */
    protected double diverseAt(List<Integer> rankedItems, int cutoff) {

        int num = 0;
        double sum = 0.0;
        for (int id = 0; id < cutoff; id++) {
            int i = rankedItems.get(id);
            SparseVector iv = trainMatrix.column(i);

            for (int jd = id + 1; jd < cutoff; jd++) {
                int j = rankedItems.get(jd);

                double corr = corrs.get(i, j);
                if (corr == 0) {
                    // if not found
                    corr = correlation(iv, trainMatrix.column(j));
                    if (!Double.isNaN(corr))
                        corrs.set(i, j, corr);
                }

                if (!Double.isNaN(corr)) {
                    sum += (1 - corr);
                    num++;
                }
            }
        }

        return 0.5 * (sum / num);
    }
    
    private ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    /**
     * @return the evaluation results of ranking predictions
     */
    protected Map<Measure, Double> evalRankings() throws Exception {
        int capacity = Lists.initSize(testMatrix.numRows());
        final List<Double> precs10 = new ArrayList<>(capacity);
        final List<Double> recalls10 = new ArrayList<>(capacity);
        final List<Double> aps10 = new ArrayList<>(capacity);
        final List<Double> ndcgs10 = new ArrayList<>(capacity);
        
        // candidate items for all users: here only training items
        // use HashSet instead of ArrayList to speedup removeAll() and contains() operations: HashSet: O(1); ArrayList: O(log n).
        final List<Integer> candItems = new ArrayList<Integer>(trainMatrix.allcolumns());
        final int numTopNRanks = numRecs < 0 ? 10 : numRecs;
//        if (verbose)
//            Logs.debug("{}{} has candidate items: {}", algoName, foldInfo, candItems.size());
        final CountDownLatch latch = new CountDownLatch(numUsers);
        // for each test user
        for (int u = 0, um = testMatrix.numRows(); u < um; u++) {
        	final int user = u;
			executor.submit(new Runnable() {
				@Override
				public void run() {
					try {
            // number of candidate items for all users
            // get positive items from test matrix
            Set<Integer> correctItems = testMatrix.getColumnsSet(user);
            if (correctItems.size() == 0)
                return; // no testing data for user u
            
            // remove rated items from candidate items
            Set<Integer> ratedItems = trainMatrix.getColumnsSet(user);
            // predict the ranking scores (unordered) of all candidate items
            List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));
            for (int i=0;i<candItems.size();i++) {
            	int j=candItems.get(i);
                // item j is not rated
                if (!ratedItems.contains(j)) {
                    final double rank = ranking(user, j);
                    if (!Double.isNaN(rank)) {
                        itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
                    } else {
    					continue;
    				}
                }
            }

            if (itemScores.size() == 0)
                return; // no recommendations available for user u

            // order the ranking scores from highest to lowest: List to preserve orders
            itemScores = Lists.sortListTopK(itemScores, true, numTopNRanks);
            List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || itemScores.size() <= numRecs) ? itemScores
                    : itemScores.subList(0, numRecs);
            List<Integer> rankedItems = new ArrayList<Integer>();
            for (Map.Entry<Integer, Double> kv : recomd) {
                Integer item = kv.getKey();
                rankedItems.add(item);
            }

            List<Integer> cutoffs = Arrays.asList(5,10);
            Map<Integer, Double> precs = Measures.PrecAt(rankedItems, correctItems, cutoffs);
            Map<Integer, Double> recalls = Measures.RecallAt(rankedItems, correctItems, cutoffs);
            synchronized (precs10) {precs10.add(precs.get(10));}
            synchronized (recalls10) {recalls10.add(recalls.get(10));}
            synchronized (aps10) {aps10.add(Measures.AP(rankedItems.subList(0, 10),correctItems));}
            synchronized (ndcgs10){ndcgs10.add(Measures.nDCG(rankedItems.subList(0, 10), correctItems));};
					} catch (Exception e) {
						Logs.error("evalRanking errors", e);
					} finally {
						latch.countDown();
					}
				}
			});
		}

		while (latch.getCount() > 0) {
			Thread.sleep(1000);
		}
		latch.await();
        // measure the performance
        Map<Measure, Double> measures = new HashMap<>();
        measures.put(Measure.Pre10, Stats.mean(precs10));
        measures.put(Measure.Rec10, Stats.mean(recalls10));
        measures.put(Measure.MAP10, Stats.mean(aps10));
        measures.put(Measure.NDCG10, Stats.mean(ndcgs10));
        executor.shutdownNow();
        return measures;
    }

    /**
     * execution method of a recommender
     */
    public void execute() throws Exception {

        Stopwatch sw = Stopwatch.createStarted();
        if (Debug.ON) {
            // learn a recommender model
            initModel();

            // show algorithm's configuration
            printAlgoConfig();

            // build the model
            buildModel();

            // post-processing after building a model, e.g., release intermediate memory to avoid memory leak
            postModel();
        } else {
            /**
             * load a learned model: this code will not be executed unless "Debug.OFF" mainly for the purpose of
             * exemplifying how to use the saved models
             */
            loadModel();
        }
        long trainTime = sw.elapsed(TimeUnit.MILLISECONDS);

        // validation
        if (validationRatio > 0 && validationRatio < 1) {
            validateModel();

            trainTime = sw.elapsed(TimeUnit.MILLISECONDS);
        }

        // evaluation
        if (verbose)
            Logs.debug("{}{} evaluate test data ... ", algoName, foldInfo);
        // TODO: to predict ratings only, or do item recommendations only
        measures = isRankingPred ? evalRankings() : null;
        String measurements = getEvalInfo(measures);
        sw.stop();
        long testTime = sw.elapsed(TimeUnit.MILLISECONDS) - trainTime;

        // collecting results
        measures.put(Measure.TrainTime, (double) trainTime);
        measures.put(Measure.TestTime, (double) testTime);

        String evalInfo = algoName + foldInfo + ": " + measurements + "\tTime: "
                + Dates.parse(measures.get(Measure.TrainTime).longValue()) + ", "
                + Dates.parse(measures.get(Measure.TestTime).longValue());
        if (!isRankingPred)
            evalInfo += "\tView: " + view;

        if (fold > 0)
            Logs.debug(evalInfo);

        if (isSaveModel)
            saveModel();
    }

    /**
     * logistic function g(x)
     */
    protected double g(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    /**
     * @param x     input value
     * @param mu    mean of normal distribution
     * @param sigma standard deviation of normation distribution
     * @return a gaussian value with mean {@code mu} and standard deviation {@code sigma};
     */
    protected double gaussian(double x, double mu, double sigma) {
        return Math.exp(-0.5 * Math.pow(x - mu, 2) / (sigma * sigma));
    }

    /**
     * gradient value of logistic function g(x)
     */
    protected double gd(double x) {
        return g(x) * g(-x);
    }

    protected LineConfiger getModelParams(String algoName) {
        return cf.contains(algoName) ? cf.getParamOptions(algoName) : null;
    }
    /**
     * initilize recommender model
     */
    protected void initModel() throws Exception {
    }
    protected boolean isranking(int u) {
            	boolean result=false;
            	List<Integer> items=testMatrix.row(u).getIndexList();
            	for(int j:items){
            		if(trainMatrix.columnSize(j)>=5){
            			result=true;
            			break;
            		}
            	}
            return  result;
    }
    /**
     * determine whether the rating of a user-item (u, j) is used to predicted
     */
    protected boolean isTestable(int u,int i) {
        switch (view) {
            case "cold-start":
            	boolean result=false;
            	List<Integer> items=testMatrix.row(u).getIndexList();
            	for(int j:items ){
            		if(trainMatrix.columnSize(j)>5){
            			result=true;
            			break;
            		}
            	}
            return  result;
            case "all":
            default:
                return true;
        }
    }
    /**
     * Deserializing a learned model (i.e., variable data) from files.
     */
    protected void loadModel() throws Exception {
    }

    /**
     * normalize a rating to the region (0, 1)
     */
    protected double normalize(double rate) {
        return (rate - minRate) / (maxRate - minRate);
    }

    protected double perplexity(int u, int j, double r) throws Exception {
        return 0;
    }

    /**
     * After learning model: release some intermediate data to avoid memory leak
     */
    protected void postModel() throws Exception {
    }

    /**
     * predict a specific rating for user u on item j, note that the prediction is not bounded. It is useful for
     * building models with no need to bound predictions.
     *
     * @param u user id
     * @param j item id
     * @return raw prediction without bounded
     */
    protected double predict(int u, int j) throws Exception {
        return globalMean;
    }

    /**
     * Below are a set of mathematical functions. As many recommenders often adopts them, for conveniency's sake, we put
     * these functions in the base Recommender class, though they belong to Math class.
     *
     */

    /**
     * predict a specific rating for user u on item j. It is useful for evalution which requires predictions are
     * bounded.
     *
     * @param u     user id
     * @param j     item id
     * @param bound whether to bound the prediction
     * @return prediction
     */
    protected double predict(int u, int j, boolean bound) throws Exception {
        double pred = predict(u, j);

        if (bound) {
            if (pred > maxRate)
                pred = maxRate;
            if (pred < minRate)
                pred = minRate;
        }

        return pred;
    }

    private void printAlgoConfig() {
        String algoInfo = toString();

        Class<? extends Recommender> cl = this.getClass();
        // basic annotation
        String algoConfig = cl.getAnnotation(Configuration.class).value();

        // additional algorithm-specific configuration
        if (cl.isAnnotationPresent(AddConfiguration.class)) {
            AddConfiguration add = cl.getAnnotation(AddConfiguration.class);

            String before = add.before();
            if (!Strings.isNullOrEmpty(before))
                algoConfig = before + ", " + algoConfig;

            String after = add.after();
            if (!Strings.isNullOrEmpty(after))
                algoConfig += ", " + after;
        }

        if (!algoInfo.isEmpty()) {
            if (!algoConfig.isEmpty())
                Logs.debug("{}: [{}] = [{}]", algoName, algoConfig, algoInfo);
            else
                Logs.debug("{}: {}", algoName, algoInfo);
        }
    }

    /**
     * predict a ranking score for user u on item j: default case using the unbounded predicted rating values
     *
     * @param u user id
     * @param j item id
     * @return a ranking score for user u on item j
     */
    protected double ranking(int u, int j) throws Exception {
        return predict(u, j, false);
    }

    public void run() {
        try {
            execute();
        } catch (Exception e) {
            // capture error message
            Logs.error(e.getMessage());

            e.printStackTrace();
        }
    }

    /**
     * Serializing a learned model (i.e., variable data) to files.
     */
    protected void saveModel() throws Exception {
    }

    /**
     * Set a user-specific name of an algorithm
     */
    protected void setAlgoName(String algoName) {
        this.algoName = algoName;

        // get parameters of an algorithm
        algoOptions = getModelParams(algoName);
    }

    /**
     * useful to print out specific recommender's settings
     */
    @Override
    public String toString() {
        return "";
    }

    /**
     * validate model with held-out validation data
     */
    protected void validateModel(){
    }
}
