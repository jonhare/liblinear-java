package de.bwaldvogel.liblinear;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.Formatter;
import java.util.Locale;
import java.util.Random;
import java.util.regex.Pattern;

/**
 * <h2>Java port of <a
 * href="http://www.csie.ntu.edu.tw/~cjlin/liblinear/">liblinear</a></h2>
 * 
 * <p>
 * The usage should be pretty similar to the C version of <tt>liblinear</tt>.
 * </p>
 * <p>
 * Please consider reading the <tt>README</tt> file of <tt>liblinear</tt>.
 * </p>
 * 
 * <p>
 * <em>The port was done by Benedikt Waldvogel (mail at bwaldvogel.de)</em>
 * </p>
 * 
 * @version 1.92
 */
public class DenseLinear {

	static final Charset FILE_CHARSET = Charset.forName("ISO-8859-1");

	static final Locale DEFAULT_LOCALE = Locale.ENGLISH;

	private static Object OUTPUT_MUTEX = new Object();
	private static PrintStream DEBUG_OUTPUT = System.out;

	private static final long DEFAULT_RANDOM_SEED = 0L;
	static Random random = new Random(DEFAULT_RANDOM_SEED);

	/**
	 * @param target
	 *            predicted classes
	 */
	public static void crossValidation(DenseProblem prob, Parameter param, int nr_fold, double[] target) {
		int i;
		final int[] fold_start = new int[nr_fold + 1];
		final int l = prob.l;
		final int[] perm = new int[l];

		for (i = 0; i < l; i++)
			perm[i] = i;
		for (i = 0; i < l; i++) {
			final int j = i + random.nextInt(l - i);
			swap(perm, i, j);
		}
		for (i = 0; i <= nr_fold; i++)
			fold_start[i] = i * l / nr_fold;

		for (i = 0; i < nr_fold; i++) {
			final int begin = fold_start[i];
			final int end = fold_start[i + 1];
			int j, k;
			final DenseProblem subprob = new DenseProblem();

			subprob.bias = prob.bias;
			subprob.n = prob.n;
			subprob.l = l - (end - begin);
			subprob.x = new double[subprob.l][];
			subprob.y = new double[subprob.l];

			k = 0;
			for (j = 0; j < begin; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			for (j = end; j < l; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			final Model submodel = train(subprob, param);
			for (j = begin; j < end; j++)
				target[perm[j]] = predict(submodel, prob.x[perm[j]]);
		}
	}

	/** used as complex return type */
	private static class GroupClassesReturn {

		final int[] count;
		final int[] label;
		final int nr_class;
		final int[] start;

		GroupClassesReturn(int nr_class, int[] label, int[] start, int[] count) {
			this.nr_class = nr_class;
			this.label = label;
			this.start = start;
			this.count = count;
		}
	}

	private static GroupClassesReturn groupClasses(DenseProblem prob, int[] perm) {
		final int l = prob.l;
		int max_nr_class = 16;
		int nr_class = 0;

		int[] label = new int[max_nr_class];
		int[] count = new int[max_nr_class];
		final int[] data_label = new int[l];
		int i;

		for (i = 0; i < l; i++) {
			final int this_label = (int) prob.y[i];
			int j;
			for (j = 0; j < nr_class; j++) {
				if (this_label == label[j]) {
					++count[j];
					break;
				}
			}
			data_label[i] = j;
			if (j == nr_class) {
				if (nr_class == max_nr_class) {
					max_nr_class *= 2;
					label = copyOf(label, max_nr_class);
					count = copyOf(count, max_nr_class);
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		final int[] start = new int[nr_class];
		start[0] = 0;
		for (i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + count[i - 1];
		for (i = 0; i < l; i++) {
			perm[start[data_label[i]]] = i;
			++start[data_label[i]];
		}
		start[0] = 0;
		for (i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + count[i - 1];

		return new GroupClassesReturn(nr_class, label, start, count);
	}

	static void info(String message) {
		synchronized (OUTPUT_MUTEX) {
			if (DEBUG_OUTPUT == null)
				return;
			DEBUG_OUTPUT.printf(message);
			DEBUG_OUTPUT.flush();
		}
	}

	static void info(String format, Object... args) {
		synchronized (OUTPUT_MUTEX) {
			if (DEBUG_OUTPUT == null)
				return;
			DEBUG_OUTPUT.printf(format, args);
			DEBUG_OUTPUT.flush();
		}
	}

	/**
	 * @param s
	 *            the string to parse for the double value
	 * @throws IllegalArgumentException
	 *             if s is empty or represents NaN or Infinity
	 * @throws NumberFormatException
	 *             see {@link Double#parseDouble(String)}
	 */
	static double atof(String s) {
		if (s == null || s.length() < 1)
			throw new IllegalArgumentException("Can't convert empty string to integer");
		final double d = Double.parseDouble(s);
		if (Double.isNaN(d) || Double.isInfinite(d)) {
			throw new IllegalArgumentException("NaN or Infinity in input: " + s);
		}
		return (d);
	}

	/**
	 * @param s
	 *            the string to parse for the integer value
	 * @throws IllegalArgumentException
	 *             if s is empty
	 * @throws NumberFormatException
	 *             see {@link Integer#parseInt(String)}
	 */
	static int atoi(String s) throws NumberFormatException {
		if (s == null || s.length() < 1)
			throw new IllegalArgumentException("Can't convert empty string to integer");
		// Integer.parseInt doesn't accept '+' prefixed strings
		if (s.charAt(0) == '+')
			s = s.substring(1);
		return Integer.parseInt(s);
	}

	/**
	 * Java5 'backport' of Arrays.copyOf
	 */
	public static double[] copyOf(double[] original, int newLength) {
		final double[] copy = new double[newLength];
		System.arraycopy(original, 0, copy, 0, Math.min(original.length, newLength));
		return copy;
	}

	/**
	 * Java5 'backport' of Arrays.copyOf
	 */
	public static int[] copyOf(int[] original, int newLength) {
		final int[] copy = new int[newLength];
		System.arraycopy(original, 0, copy, 0, Math.min(original.length, newLength));
		return copy;
	}

	/**
	 * Loads the model from inputReader. It uses
	 * {@link java.util.Locale#ENGLISH} for number formatting.
	 * 
	 * <p>
	 * Note: The inputReader is <b>NOT closed</b> after reading or in case of an
	 * exception.
	 * </p>
	 */
	public static Model loadModel(Reader inputReader) throws IOException {
		final Model model = new Model();

		model.label = null;

		final Pattern whitespace = Pattern.compile("\\s+");

		BufferedReader reader = null;
		if (inputReader instanceof BufferedReader) {
			reader = (BufferedReader) inputReader;
		} else {
			reader = new BufferedReader(inputReader);
		}

		String line = null;
		while ((line = reader.readLine()) != null) {
			final String[] split = whitespace.split(line);
			if (split[0].equals("solver_type")) {
				final SolverType solver = SolverType.valueOf(split[1]);
				if (solver == null) {
					throw new RuntimeException("unknown solver type");
				}
				model.solverType = solver;
			} else if (split[0].equals("nr_class")) {
				model.nr_class = atoi(split[1]);
				Integer.parseInt(split[1]);
			} else if (split[0].equals("nr_feature")) {
				model.nr_feature = atoi(split[1]);
			} else if (split[0].equals("bias")) {
				model.bias = atof(split[1]);
			} else if (split[0].equals("w")) {
				break;
			} else if (split[0].equals("label")) {
				model.label = new int[model.nr_class];
				for (int i = 0; i < model.nr_class; i++) {
					model.label[i] = atoi(split[i + 1]);
				}
			} else {
				throw new RuntimeException("unknown text in model file: [" + line + "]");
			}
		}

		int w_size = model.nr_feature;
		if (model.bias >= 0)
			w_size++;

		int nr_w = model.nr_class;
		if (model.nr_class == 2 && model.solverType != SolverType.MCSVM_CS)
			nr_w = 1;

		model.w = new double[w_size * nr_w];
		final int[] buffer = new int[128];

		for (int i = 0; i < w_size; i++) {
			for (int j = 0; j < nr_w; j++) {
				int b = 0;
				while (true) {
					final int ch = reader.read();
					if (ch == -1) {
						throw new EOFException("unexpected EOF");
					}
					if (ch == ' ') {
						model.w[i * nr_w + j] = atof(new String(buffer, 0, b));
						break;
					} else {
						buffer[b++] = ch;
					}
				}
			}
		}

		return model;
	}

	/**
	 * Loads the model from the file with ISO-8859-1 charset. It uses
	 * {@link java.util.Locale#ENGLISH} for number formatting.
	 */
	public static Model loadModel(File modelFile) throws IOException {
		final BufferedReader inputReader = new BufferedReader(new InputStreamReader(new FileInputStream(modelFile),
				FILE_CHARSET));
		try {
			return loadModel(inputReader);
		} finally {
			inputReader.close();
		}
	}

	static void closeQuietly(Closeable c) {
		if (c == null)
			return;
		try {
			c.close();
		} catch (final Throwable t) {
		}
	}

	public static double predict(Model model, double[] x) {
		final double[] dec_values = new double[model.nr_class];
		return predictValues(model, x, dec_values);
	}

	/**
	 * @throws IllegalArgumentException
	 *             if model is not probabilistic (see
	 *             {@link Model#isProbabilityModel()})
	 */
	public static double predictProbability(Model model, double[] x, double[] prob_estimates)
			throws IllegalArgumentException
	{
		if (!model.isProbabilityModel()) {
			final StringBuilder sb = new StringBuilder("probability output is only supported for logistic regression");
			sb.append(". This is currently only supported by the following solvers: ");
			int i = 0;
			for (final SolverType solverType : SolverType.values()) {
				if (solverType.isLogisticRegressionSolver()) {
					if (i++ > 0) {
						sb.append(", ");
					}
					sb.append(solverType.name());
				}
			}
			throw new IllegalArgumentException(sb.toString());
		}
		final int nr_class = model.nr_class;
		int nr_w;
		if (nr_class == 2)
			nr_w = 1;
		else
			nr_w = nr_class;

		final double label = predictValues(model, x, prob_estimates);
		for (int i = 0; i < nr_w; i++)
			prob_estimates[i] = 1 / (1 + Math.exp(-prob_estimates[i]));

		if (nr_class == 2) // for binary classification
			prob_estimates[1] = 1. - prob_estimates[0];
		else {
			double sum = 0;
			for (int i = 0; i < nr_class; i++)
				sum += prob_estimates[i];

			for (int i = 0; i < nr_class; i++)
				prob_estimates[i] = prob_estimates[i] / sum;
		}

		return label;
	}

	public static double predictValues(Model model, double[] x, double[] dec_values) {
		int n;
		if (model.bias >= 0)
			n = model.nr_feature + 1;
		else
			n = model.nr_feature;

		final double[] w = model.w;

		int nr_w;
		if (model.nr_class == 2 && model.solverType != SolverType.MCSVM_CS)
			nr_w = 1;
		else
			nr_w = model.nr_class;

		for (int i = 0; i < nr_w; i++)
			dec_values[i] = 0;

		for (int idx = 0; idx < n; idx++) {
			// the dimension of testing data may exceed that of training
			for (int i = 0; i < nr_w; i++) {
				dec_values[i] += w[idx * nr_w + i] * x[idx];
			}
		}

		if (model.nr_class == 2) {
			if (model.solverType.isSupportVectorRegression())
				return dec_values[0];
			else
				return (dec_values[0] > 0) ? model.label[0] : model.label[1];
		} else {
			int dec_max_idx = 0;
			for (int i = 1; i < model.nr_class; i++) {
				if (dec_values[i] > dec_values[dec_max_idx])
					dec_max_idx = i;
			}
			return model.label[dec_max_idx];
		}
	}

	static void printf(Formatter formatter, String format, Object... args) throws IOException {
		formatter.format(format, args);
		final IOException ioException = formatter.ioException();
		if (ioException != null)
			throw ioException;
	}

	/**
	 * Writes the model to the modelOutput. It uses
	 * {@link java.util.Locale#ENGLISH} for number formatting.
	 * 
	 * <p>
	 * <b>Note: The modelOutput is closed after reading or in case of an
	 * exception.</b>
	 * </p>
	 */
	public static void saveModel(Writer modelOutput, Model model) throws IOException {
		final int nr_feature = model.nr_feature;
		int w_size = nr_feature;
		if (model.bias >= 0)
			w_size++;

		int nr_w = model.nr_class;
		if (model.nr_class == 2 && model.solverType != SolverType.MCSVM_CS)
			nr_w = 1;

		final Formatter formatter = new Formatter(modelOutput, DEFAULT_LOCALE);
		try {
			printf(formatter, "solver_type %s\n", model.solverType.name());
			printf(formatter, "nr_class %d\n", model.nr_class);

			if (model.label != null) {
				printf(formatter, "label");
				for (int i = 0; i < model.nr_class; i++) {
					printf(formatter, " %d", model.label[i]);
				}
				printf(formatter, "\n");
			}

			printf(formatter, "nr_feature %d\n", nr_feature);
			printf(formatter, "bias %.16g\n", model.bias);

			printf(formatter, "w\n");
			for (int i = 0; i < w_size; i++) {
				for (int j = 0; j < nr_w; j++) {
					final double value = model.w[i * nr_w + j];

					/**
					 * this optimization is the reason for
					 * {@link Model#equals(double[], double[])}
					 */
					if (value == 0.0) {
						printf(formatter, "%d ", 0);
					} else {
						printf(formatter, "%.16g ", value);
					}
				}
				printf(formatter, "\n");
			}

			formatter.flush();
			final IOException ioException = formatter.ioException();
			if (ioException != null)
				throw ioException;
		} finally {
			formatter.close();
		}
	}

	/**
	 * Writes the model to the file with ISO-8859-1 charset. It uses
	 * {@link java.util.Locale#ENGLISH} for number formatting.
	 */
	public static void saveModel(File modelFile, Model model) throws IOException {
		final BufferedWriter modelOutput = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(modelFile),
				FILE_CHARSET));
		saveModel(modelOutput, model);
	}

	/*
	 * this method corresponds to the following define in the C version: #define
	 * GETI(i) (y[i]+1)
	 */
	private static int GETI(byte[] y, int i) {
		return y[i] + 1;
	}

	/**
	 * A coordinate descent algorithm for L1-loss and L2-loss SVM dual problems
	 * 
	 * <pre>
	 *  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
	 *    s.t.      0 <= \alpha_i <= upper_bound_i,
	 * 
	 *  where Qij = yi yj xi^T xj and
	 *  D is a diagonal matrix
	 * 
	 * In L1-SVM case:
	 *     upper_bound_i = Cp if y_i = 1
	 *      upper_bound_i = Cn if y_i = -1
	 *      D_ii = 0
	 * In L2-SVM case:
	 *      upper_bound_i = INF
	 *      D_ii = 1/(2*Cp) if y_i = 1
	 *      D_ii = 1/(2*Cn) if y_i = -1
	 * 
	 * Given:
	 * x, y, Cp, Cn
	 * eps is the stopping tolerance
	 * 
	 * solution will be put in w
	 * 
	 * See Algorithm 3 of Hsieh et al., ICML 2008
	 * </pre>
	 */
	private static void solve_l2r_l1l2_svc(DenseProblem prob, double[] w, double eps, double Cp, double Cn,
			SolverType solver_type)
	{
		final int l = prob.l;
		final int w_size = prob.n;
		int i, s, iter = 0;
		double C, d, G;
		final double[] QD = new double[l];
		final int max_iter = 1000;
		final int[] index = new int[l];
		final double[] alpha = new double[l];
		final byte[] y = new byte[l];
		int active_size = l;

		// PG: projected gradient, for shrinking and stopping
		double PG;
		double PGmax_old = Double.POSITIVE_INFINITY;
		double PGmin_old = Double.NEGATIVE_INFINITY;
		double PGmax_new, PGmin_new;

		// default solver_type: L2R_L2LOSS_SVC_DUAL
		final double diag[] = new double[] { 0.5 / Cn, 0, 0.5 / Cp };
		final double upper_bound[] = new double[] { Double.POSITIVE_INFINITY, 0, Double.POSITIVE_INFINITY };
		if (solver_type == SolverType.L2R_L1LOSS_SVC_DUAL) {
			diag[0] = 0;
			diag[2] = 0;
			upper_bound[0] = Cn;
			upper_bound[2] = Cp;
		}

		for (i = 0; i < l; i++) {
			if (prob.y[i] > 0) {
				y[i] = +1;
			} else {
				y[i] = -1;
			}
		}

		// Initial alpha can be set here. Note that
		// 0 <= alpha[i] <= upper_bound[GETI(i)]
		for (i = 0; i < l; i++)
			alpha[i] = 0;

		for (i = 0; i < w_size; i++)
			w[i] = 0;
		for (i = 0; i < l; i++) {
			QD[i] = diag[GETI(y, i)];

			for (int j = 0; j < w_size; j++) {
				final double val = prob.x[i][j];
				QD[i] += val * val;
				w[j] += y[i] * alpha[i] * val;
			}
			index[i] = i;
		}

		while (iter < max_iter) {
			PGmax_new = Double.NEGATIVE_INFINITY;
			PGmin_new = Double.POSITIVE_INFINITY;

			for (i = 0; i < active_size; i++) {
				final int j = i + random.nextInt(active_size - i);
				swap(index, i, j);
			}

			for (s = 0; s < active_size; s++) {
				i = index[s];
				G = 0;
				final byte yi = y[i];

				for (int j = 0; j < w_size; j++) {
					G += w[j] * prob.x[i][j];
				}
				G = G * yi - 1;

				C = upper_bound[GETI(y, i)];
				G += alpha[i] * diag[GETI(y, i)];

				PG = 0;
				if (alpha[i] == 0) {
					if (G > PGmax_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					} else if (G < 0) {
						PG = G;
					}
				} else if (alpha[i] == C) {
					if (G < PGmin_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					} else if (G > 0) {
						PG = G;
					}
				} else {
					PG = G;
				}

				PGmax_new = Math.max(PGmax_new, PG);
				PGmin_new = Math.min(PGmin_new, PG);

				if (Math.abs(PG) > 1.0e-12) {
					final double alpha_old = alpha[i];
					alpha[i] = Math.min(Math.max(alpha[i] - G / QD[i], 0.0), C);
					d = (alpha[i] - alpha_old) * yi;

					for (int j = 0; j < w_size; j++) {
						w[j] += d * prob.x[i][j];
					}
				}
			}

			iter++;
			if (iter % 10 == 0)
				info(".");

			if (PGmax_new - PGmin_new <= eps) {
				if (active_size == l)
					break;
				else {
					active_size = l;
					info("*");
					PGmax_old = Double.POSITIVE_INFINITY;
					PGmin_old = Double.NEGATIVE_INFINITY;
					continue;
				}
			}
			PGmax_old = PGmax_new;
			PGmin_old = PGmin_new;
			if (PGmax_old <= 0)
				PGmax_old = Double.POSITIVE_INFINITY;
			if (PGmin_old >= 0)
				PGmin_old = Double.NEGATIVE_INFINITY;
		}

		info("%noptimization finished, #iter = %d%n", iter);
		if (iter >= max_iter)
			info("%nWARNING: reaching max number of iterations%nUsing -s 2 may be faster (also see FAQ)%n%n");

		// calculate objective value

		double v = 0;
		int nSV = 0;
		for (i = 0; i < w_size; i++)
			v += w[i] * w[i];
		for (i = 0; i < l; i++) {
			v += alpha[i] * (alpha[i] * diag[GETI(y, i)] - 2);
			if (alpha[i] > 0)
				++nSV;
		}
		info("Objective value = %g%n", v / 2);
		info("nSV = %d%n", nSV);
	}

	// To support weights for instances, use GETI(i) (i)
	private static int GETI_SVR(int i) {
		return 0;
	}

	/**
	 * A coordinate descent algorithm for L1-loss and L2-loss epsilon-SVR dual
	 * problem
	 * 
	 * min_\beta 0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| +
	 * \sum_{i=1}^l yi\beta_i, s.t. -upper_bound_i <= \beta_i <= upper_bound_i,
	 * 
	 * where Qij = xi^T xj and D is a diagonal matrix
	 * 
	 * In L1-SVM case: upper_bound_i = C lambda_i = 0 In L2-SVM case:
	 * upper_bound_i = INF lambda_i = 1/(2*C)
	 * 
	 * Given: x, y, p, C eps is the stopping tolerance
	 * 
	 * solution will be put in w
	 * 
	 * See Algorithm 4 of Ho and Lin, 2012
	 */
	private static void solve_l2r_l1l2_svr(DenseProblem prob, double[] w, Parameter param) {
		final int l = prob.l;
		final double C = param.C;
		final double p = param.p;
		final int w_size = prob.n;
		final double eps = param.eps;
		int i, s, iter = 0;
		final int max_iter = 1000;
		int active_size = l;
		final int[] index = new int[l];

		double d, G, H;
		double Gmax_old = Double.POSITIVE_INFINITY;
		double Gmax_new, Gnorm1_new;
		double Gnorm1_init = 0; // initialize to 0 to get rid of Eclipse
								// warning/error
		final double[] beta = new double[l];
		final double[] QD = new double[l];
		final double[] y = prob.y;

		// L2R_L2LOSS_SVR_DUAL
		final double[] lambda = new double[] { 0.5 / C };
		final double[] upper_bound = new double[] { Double.POSITIVE_INFINITY };

		if (param.solverType == SolverType.L2R_L1LOSS_SVR_DUAL) {
			lambda[0] = 0;
			upper_bound[0] = C;
		}

		// Initial beta can be set here. Note that
		// -upper_bound <= beta[i] <= upper_bound
		for (i = 0; i < l; i++)
			beta[i] = 0;

		for (i = 0; i < w_size; i++)
			w[i] = 0;
		for (i = 0; i < l; i++) {
			QD[i] = 0;
			for (int j = 0; j < w_size; j++) {
				final double val = prob.x[i][j];
				QD[i] += val * val;
				w[j] += beta[i] * val;
			}

			index[i] = i;
		}

		while (iter < max_iter) {
			Gmax_new = 0;
			Gnorm1_new = 0;

			for (i = 0; i < active_size; i++) {
				final int j = i + random.nextInt(active_size - i);
				swap(index, i, j);
			}

			for (s = 0; s < active_size; s++) {
				i = index[s];
				G = -y[i] + lambda[GETI_SVR(i)] * beta[i];
				H = QD[i] + lambda[GETI_SVR(i)];

				for (int ind = 0; ind < w_size; ind++) {
					final double val = prob.x[i][ind];
					G += val * w[ind];
				}

				final double Gp = G + p;
				final double Gn = G - p;
				double violation = 0;
				if (beta[i] == 0) {
					if (Gp < 0)
						violation = -Gp;
					else if (Gn > 0)
						violation = Gn;
					else if (Gp > Gmax_old && Gn < -Gmax_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					}
				} else if (beta[i] >= upper_bound[GETI_SVR(i)]) {
					if (Gp > 0)
						violation = Gp;
					else if (Gp < -Gmax_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					}
				} else if (beta[i] <= -upper_bound[GETI_SVR(i)]) {
					if (Gn < 0)
						violation = -Gn;
					else if (Gn > Gmax_old) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					}
				} else if (beta[i] > 0)
					violation = Math.abs(Gp);
				else
					violation = Math.abs(Gn);

				Gmax_new = Math.max(Gmax_new, violation);
				Gnorm1_new += violation;

				// obtain Newton direction d
				if (Gp < H * beta[i])
					d = -Gp / H;
				else if (Gn > H * beta[i])
					d = -Gn / H;
				else
					d = -beta[i];

				if (Math.abs(d) < 1.0e-12)
					continue;

				final double beta_old = beta[i];
				beta[i] = Math.min(Math.max(beta[i] + d, -upper_bound[GETI_SVR(i)]), upper_bound[GETI_SVR(i)]);
				d = beta[i] - beta_old;

				if (d != 0) {
					for (int j = 0; j < w_size; j++) {
						w[j] += d * prob.x[i][j];
					}
				}
			}

			if (iter == 0)
				Gnorm1_init = Gnorm1_new;
			iter++;
			if (iter % 10 == 0)
				info(".");

			if (Gnorm1_new <= eps * Gnorm1_init) {
				if (active_size == l)
					break;
				else {
					active_size = l;
					info("*");
					Gmax_old = Double.POSITIVE_INFINITY;
					continue;
				}
			}

			Gmax_old = Gmax_new;
		}

		info("%noptimization finished, #iter = %d%n", iter);
		if (iter >= max_iter)
			info("%nWARNING: reaching max number of iterations%nUsing -s 11 may be faster%n%n");

		// calculate objective value
		double v = 0;
		int nSV = 0;
		for (i = 0; i < w_size; i++)
			v += w[i] * w[i];
		v = 0.5 * v;
		for (i = 0; i < l; i++) {
			v += p * Math.abs(beta[i]) - y[i] * beta[i] + 0.5 * lambda[GETI_SVR(i)] * beta[i] * beta[i];
			if (beta[i] != 0)
				nSV++;
		}

		info("Objective value = %g%n", v);
		info("nSV = %d%n", nSV);
	}

	/**
	 * A coordinate descent algorithm for the dual of L2-regularized logistic
	 * regression problems
	 * 
	 * <pre>
	 *  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i) ,
	 *     s.t.      0 <= \alpha_i <= upper_bound_i,
	 * 
	 *  where Qij = yi yj xi^T xj and
	 *  upper_bound_i = Cp if y_i = 1
	 *  upper_bound_i = Cn if y_i = -1
	 * 
	 * Given:
	 * x, y, Cp, Cn
	 * eps is the stopping tolerance
	 * 
	 * solution will be put in w
	 * 
	 * See Algorithm 5 of Yu et al., MLJ 2010
	 * </pre>
	 * 
	 * @since 1.7
	 */
	private static void solve_l2r_lr_dual(DenseProblem prob, double w[], double eps, double Cp, double Cn) {
		final int l = prob.l;
		final int w_size = prob.n;
		int i, s, iter = 0;
		final double xTx[] = new double[l];
		final int max_iter = 1000;
		final int index[] = new int[l];
		final double alpha[] = new double[2 * l]; // store alpha and C - alpha
		final byte y[] = new byte[l];
		final int max_inner_iter = 100; // for inner Newton
		double innereps = 1e-2;
		final double innereps_min = Math.min(1e-8, eps);
		final double upper_bound[] = new double[] { Cn, 0, Cp };

		for (i = 0; i < l; i++) {
			if (prob.y[i] > 0) {
				y[i] = +1;
			} else {
				y[i] = -1;
			}
		}

		// Initial alpha can be set here. Note that
		// 0 < alpha[i] < upper_bound[GETI(i)]
		// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
		for (i = 0; i < l; i++) {
			alpha[2 * i] = Math.min(0.001 * upper_bound[GETI(y, i)], 1e-8);
			alpha[2 * i + 1] = upper_bound[GETI(y, i)] - alpha[2 * i];
		}

		for (i = 0; i < w_size; i++)
			w[i] = 0;
		for (i = 0; i < l; i++) {
			xTx[i] = 0;
			for (int j = 0; j < w_size; j++) {
				final double val = prob.x[i][j];
				xTx[i] += val * val;
				w[j] += y[i] * alpha[2 * i] * val;
			}
			index[i] = i;
		}

		while (iter < max_iter) {
			for (i = 0; i < l; i++) {
				final int j = i + random.nextInt(l - i);
				swap(index, i, j);
			}
			int newton_iter = 0;
			double Gmax = 0;
			for (s = 0; s < l; s++) {
				i = index[s];
				final byte yi = y[i];
				final double C = upper_bound[GETI(y, i)];
				double ywTx = 0;
				final double xisq = xTx[i];
				for (int j = 0; j < w_size; j++) {
					ywTx += w[j] * prob.x[i][j];
				}
				ywTx *= y[i];
				final double a = xisq, b = ywTx;

				// Decide to minimize g_1(z) or g_2(z)
				int ind1 = 2 * i, ind2 = 2 * i + 1, sign = 1;
				if (0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
					ind1 = 2 * i + 1;
					ind2 = 2 * i;
					sign = -1;
				}

				// g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 +
				// sign*b(z-alpha_old)
				final double alpha_old = alpha[ind1];
				double z = alpha_old;
				if (C - z < 0.5 * C)
					z = 0.1 * z;
				double gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
				Gmax = Math.max(Gmax, Math.abs(gp));

				// Newton method on the sub-problem
				final double eta = 0.1; // xi in the paper
				int inner_iter = 0;
				while (inner_iter <= max_inner_iter) {
					if (Math.abs(gp) < innereps)
						break;
					final double gpp = a + C / (C - z) / z;
					final double tmpz = z - gp / gpp;
					if (tmpz <= 0)
						z *= eta;
					else
						// tmpz in (0, C)
						z = tmpz;
					gp = a * (z - alpha_old) + sign * b + Math.log(z / (C - z));
					newton_iter++;
					inner_iter++;
				}

				if (inner_iter > 0) // update w
				{
					alpha[ind1] = z;
					alpha[ind2] = C - z;
					for (int j = 0; j < w_size; j++) {
						w[j] += sign * (z - alpha_old) * yi * prob.x[i][j];
					}
				}
			}

			iter++;
			if (iter % 10 == 0)
				info(".");

			if (Gmax < eps)
				break;

			if (newton_iter <= l / 10) {
				innereps = Math.max(innereps_min, 0.1 * innereps);
			}

		}

		info("%noptimization finished, #iter = %d%n", iter);
		if (iter >= max_iter)
			info("%nWARNING: reaching max number of iterations%nUsing -s 0 may be faster (also see FAQ)%n%n");

		// calculate objective value

		double v = 0;
		for (i = 0; i < w_size; i++)
			v += w[i] * w[i];
		v *= 0.5;
		for (i = 0; i < l; i++)
			v += alpha[2 * i] * Math.log(alpha[2 * i]) + alpha[2 * i + 1] * Math.log(alpha[2 * i + 1])
					- upper_bound[GETI(y, i)]
					* Math.log(upper_bound[GETI(y, i)]);
		info("Objective value = %g%n", v);
	}

	/**
	 * A coordinate descent algorithm for L1-regularized L2-loss support vector
	 * classification
	 * 
	 * <pre>
	 *  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
	 * 
	 * Given:
	 * x, y, Cp, Cn
	 * eps is the stopping tolerance
	 * 
	 * solution will be put in w
	 * 
	 * See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)
	 * </pre>
	 * 
	 * @since 1.5
	 */
	private static void solve_l1r_l2_svc(DenseProblem prob_col, double[] w, double eps, double Cp, double Cn) {
		final int l = prob_col.l;
		final int w_size = prob_col.n;
		int j, s, iter = 0;
		final int max_iter = 1000;
		int active_size = w_size;
		final int max_num_linesearch = 20;

		final double sigma = 0.01;
		double d, G_loss, G, H;
		double Gmax_old = Double.POSITIVE_INFINITY;
		double Gmax_new, Gnorm1_new;
		double Gnorm1_init = 0; // eclipse moans this variable might not be
								// initialized
		double d_old, d_diff;
		double loss_old = 0; // eclipse moans this variable might not be
								// initialized
		double loss_new;
		double appxcond, cond;

		final int[] index = new int[w_size];
		final byte[] y = new byte[l];
		final double[] b = new double[l]; // b = 1-ywTx
		final double[] xj_sq = new double[w_size];

		final double[] C = new double[] { Cn, 0, Cp };

		// Initial w can be set here.
		for (j = 0; j < w_size; j++)
			w[j] = 0;

		for (j = 0; j < l; j++) {
			b[j] = 1;
			if (prob_col.y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}
		for (j = 0; j < w_size; j++) {
			index[j] = j;
			xj_sq[j] = 0;
			for (int ind = 0; ind < w_size; ind++) {
				prob_col.x[j][ind] = prob_col.x[j][ind] * y[ind]; // x->value
																	// stores
																	// yi*xij
				final double val = prob_col.x[j][ind];
				b[ind] -= w[j] * val;

				xj_sq[j] += C[GETI(y, ind)] * val * val;
			}
		}

		while (iter < max_iter) {
			Gmax_new = 0;
			Gnorm1_new = 0;

			for (j = 0; j < active_size; j++) {
				final int i = j + random.nextInt(active_size - j);
				swap(index, i, j);
			}

			for (s = 0; s < active_size; s++) {
				j = index[s];
				G_loss = 0;
				H = 0;

				for (int ind = 0; ind < w_size; ind++) {
					if (b[ind] > 0) {
						final double val = prob_col.x[j][ind];
						final double tmp = C[GETI(y, ind)] * val;
						G_loss -= tmp * b[ind];
						H += tmp * val;
					}
				}
				G_loss *= 2;

				G = G_loss;
				H *= 2;
				H = Math.max(H, 1e-12);

				final double Gp = G + 1;
				final double Gn = G - 1;
				double violation = 0;
				if (w[j] == 0) {
					if (Gp < 0)
						violation = -Gp;
					else if (Gn > 0)
						violation = Gn;
					else if (Gp > Gmax_old / l && Gn < -Gmax_old / l) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					}
				} else if (w[j] > 0)
					violation = Math.abs(Gp);
				else
					violation = Math.abs(Gn);

				Gmax_new = Math.max(Gmax_new, violation);
				Gnorm1_new += violation;

				// obtain Newton direction d
				if (Gp < H * w[j])
					d = -Gp / H;
				else if (Gn > H * w[j])
					d = -Gn / H;
				else
					d = -w[j];

				if (Math.abs(d) < 1.0e-12)
					continue;

				double delta = Math.abs(w[j] + d) - Math.abs(w[j]) + G * d;
				d_old = 0;
				int num_linesearch;
				for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
					d_diff = d_old - d;
					cond = Math.abs(w[j] + d) - Math.abs(w[j]) - sigma * delta;

					appxcond = xj_sq[j] * d * d + G_loss * d + cond;
					if (appxcond <= 0) {
						for (int ind = 0; ind < prob_col.x[j].length; ind++) {
							b[ind] += d_diff * prob_col.x[j][ind];
						}
						break;
					}

					if (num_linesearch == 0) {
						loss_old = 0;
						loss_new = 0;
						for (int ind = 0; ind < prob_col.x[j].length; ind++) {
							if (b[ind] > 0) {
								loss_old += C[GETI(y, ind)] * b[ind] * b[ind];
							}
							final double b_new = b[ind] + d_diff * prob_col.x[j][ind];
							b[ind] = b_new;
							if (b_new > 0) {
								loss_new += C[GETI(y, ind)] * b_new * b_new;
							}
						}
					} else {
						loss_new = 0;
						for (int ind = 0; ind < prob_col.x[j].length; ind++) {
							final double b_new = b[ind] + d_diff * prob_col.x[j][ind];
							b[ind] = b_new;
							if (b_new > 0) {
								loss_new += C[GETI(y, ind)] * b_new * b_new;
							}
						}
					}

					cond = cond + loss_new - loss_old;
					if (cond <= 0)
						break;
					else {
						d_old = d;
						d *= 0.5;
						delta *= 0.5;
					}
				}

				w[j] += d;

				// recompute b[] if line search takes too many steps
				if (num_linesearch >= max_num_linesearch) {
					info("#");
					for (int i = 0; i < l; i++)
						b[i] = 1;

					for (int i = 0; i < w_size; i++) {
						if (w[i] == 0)
							continue;
						for (int ind = 0; ind < prob_col.x[j].length; ind++) {
							b[ind] -= w[i] * prob_col.x[j][ind];
						}
					}
				}
			}

			if (iter == 0) {
				Gnorm1_init = Gnorm1_new;
			}
			iter++;
			if (iter % 10 == 0)
				info(".");

			if (Gmax_new <= eps * Gnorm1_init) {
				if (active_size == w_size)
					break;
				else {
					active_size = w_size;
					info("*");
					Gmax_old = Double.POSITIVE_INFINITY;
					continue;
				}
			}

			Gmax_old = Gmax_new;
		}

		info("%noptimization finished, #iter = %d%n", iter);
		if (iter >= max_iter)
			info("%nWARNING: reaching max number of iterations%n");

		// calculate objective value

		double v = 0;
		int nnz = 0;
		for (j = 0; j < w_size; j++) {
			for (int ind = 0; ind < prob_col.x[j].length; ind++) {
				prob_col.x[j][ind] = prob_col.x[j][ind] * prob_col.y[ind]; // restore
																			// x->value
			}
			if (w[j] != 0) {
				v += Math.abs(w[j]);
				nnz++;
			}
		}
		for (j = 0; j < l; j++)
			if (b[j] > 0)
				v += C[GETI(y, j)] * b[j] * b[j];

		info("Objective value = %g%n", v);
		info("#nonzeros/#features = %d/%d%n", nnz, w_size);
	}

	/**
	 * A coordinate descent algorithm for L1-regularized logistic regression
	 * problems
	 * 
	 * <pre>
	 *  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
	 * 
	 * Given:
	 * x, y, Cp, Cn
	 * eps is the stopping tolerance
	 * 
	 * solution will be put in w
	 * 
	 * See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)
	 * </pre>
	 * 
	 * @since 1.5
	 */
	private static void solve_l1r_lr(DenseProblem prob_col, double[] w, double eps, double Cp, double Cn) {
		final int l = prob_col.l;
		final int w_size = prob_col.n;
		int j, s, newton_iter = 0, iter = 0;
		final int max_newton_iter = 100;
		final int max_iter = 1000;
		final int max_num_linesearch = 20;
		int active_size;
		int QP_active_size;

		final double nu = 1e-12;
		double inner_eps = 1;
		final double sigma = 0.01;
		double w_norm, w_norm_new;
		double z, G, H;
		double Gnorm1_init = 0; // eclipse moans this variable might not be
								// initialized
		double Gmax_old = Double.POSITIVE_INFINITY;
		double Gmax_new, Gnorm1_new;
		double QP_Gmax_old = Double.POSITIVE_INFINITY;
		double QP_Gmax_new, QP_Gnorm1_new;
		double delta, negsum_xTd, cond;

		final int[] index = new int[w_size];
		final byte[] y = new byte[l];
		final double[] Hdiag = new double[w_size];
		final double[] Grad = new double[w_size];
		final double[] wpd = new double[w_size];
		final double[] xjneg_sum = new double[w_size];
		final double[] xTd = new double[l];
		final double[] exp_wTx = new double[l];
		final double[] exp_wTx_new = new double[l];
		final double[] tau = new double[l];
		final double[] D = new double[l];

		final double[] C = { Cn, 0, Cp };

		// Initial w can be set here.
		for (j = 0; j < w_size; j++)
			w[j] = 0;

		for (j = 0; j < l; j++) {
			if (prob_col.y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;

			exp_wTx[j] = 0;
		}

		w_norm = 0;
		for (j = 0; j < w_size; j++) {
			w_norm += Math.abs(w[j]);
			wpd[j] = w[j];
			index[j] = j;
			xjneg_sum[j] = 0;
			for (int ind = 0; ind < prob_col.x[j].length; ind++) {
				final double val = prob_col.x[j][ind];
				exp_wTx[ind] += w[j] * val;
				if (y[ind] == -1) {
					xjneg_sum[j] += C[GETI(y, ind)] * val;
				}
			}
		}
		for (j = 0; j < l; j++) {
			exp_wTx[j] = Math.exp(exp_wTx[j]);
			final double tau_tmp = 1 / (1 + exp_wTx[j]);
			tau[j] = C[GETI(y, j)] * tau_tmp;
			D[j] = C[GETI(y, j)] * exp_wTx[j] * tau_tmp * tau_tmp;
		}

		while (newton_iter < max_newton_iter) {
			Gmax_new = 0;
			Gnorm1_new = 0;
			active_size = w_size;

			for (s = 0; s < active_size; s++) {
				j = index[s];
				Hdiag[j] = nu;
				Grad[j] = 0;

				double tmp = 0;
				for (int ind = 0; ind < prob_col.x[j].length; ind++) {
					Hdiag[j] += prob_col.x[j][ind] * prob_col.x[j][ind] * D[ind];
					tmp += prob_col.x[j][ind] * tau[ind];
				}
				Grad[j] = -tmp + xjneg_sum[j];

				final double Gp = Grad[j] + 1;
				final double Gn = Grad[j] - 1;
				double violation = 0;
				if (w[j] == 0) {
					if (Gp < 0)
						violation = -Gp;
					else if (Gn > 0)
						violation = Gn;
					// outer-level shrinking
					else if (Gp > Gmax_old / l && Gn < -Gmax_old / l) {
						active_size--;
						swap(index, s, active_size);
						s--;
						continue;
					}
				} else if (w[j] > 0)
					violation = Math.abs(Gp);
				else
					violation = Math.abs(Gn);

				Gmax_new = Math.max(Gmax_new, violation);
				Gnorm1_new += violation;
			}

			if (newton_iter == 0)
				Gnorm1_init = Gnorm1_new;

			if (Gnorm1_new <= eps * Gnorm1_init)
				break;

			iter = 0;
			QP_Gmax_old = Double.POSITIVE_INFINITY;
			QP_active_size = active_size;

			for (int i = 0; i < l; i++)
				xTd[i] = 0;

			// optimize QP over wpd
			while (iter < max_iter) {
				QP_Gmax_new = 0;
				QP_Gnorm1_new = 0;

				for (j = 0; j < QP_active_size; j++) {
					final int i = random.nextInt(QP_active_size - j);
					swap(index, i, j);
				}

				for (s = 0; s < QP_active_size; s++) {
					j = index[s];
					H = Hdiag[j];

					G = Grad[j] + (wpd[j] - w[j]) * nu;
					for (int ind = 0; ind < prob_col.x[j].length; ind++) {
						G += prob_col.x[j][ind] * D[ind] * xTd[ind];
					}

					final double Gp = G + 1;
					final double Gn = G - 1;
					double violation = 0;
					if (wpd[j] == 0) {
						if (Gp < 0)
							violation = -Gp;
						else if (Gn > 0)
							violation = Gn;
						// inner-level shrinking
						else if (Gp > QP_Gmax_old / l && Gn < -QP_Gmax_old / l) {
							QP_active_size--;
							swap(index, s, QP_active_size);
							s--;
							continue;
						}
					} else if (wpd[j] > 0)
						violation = Math.abs(Gp);
					else
						violation = Math.abs(Gn);

					QP_Gmax_new = Math.max(QP_Gmax_new, violation);
					QP_Gnorm1_new += violation;

					// obtain solution of one-variable problem
					if (Gp < H * wpd[j])
						z = -Gp / H;
					else if (Gn > H * wpd[j])
						z = -Gn / H;
					else
						z = -wpd[j];

					if (Math.abs(z) < 1.0e-12)
						continue;
					z = Math.min(Math.max(z, -10.0), 10.0);

					wpd[j] += z;

					for (int ind = 0; ind < prob_col.x[j].length; ind++) {
						xTd[ind] += prob_col.x[j][ind] * z;
					}
				}

				iter++;

				if (QP_Gnorm1_new <= inner_eps * Gnorm1_init) {
					// inner stopping
					if (QP_active_size == active_size)
						break;
					// active set reactivation
					else {
						QP_active_size = active_size;
						QP_Gmax_old = Double.POSITIVE_INFINITY;
						continue;
					}
				}

				QP_Gmax_old = QP_Gmax_new;
			}

			if (iter >= max_iter)
				info("WARNING: reaching max number of inner iterations%n");

			delta = 0;
			w_norm_new = 0;
			for (j = 0; j < w_size; j++) {
				delta += Grad[j] * (wpd[j] - w[j]);
				if (wpd[j] != 0)
					w_norm_new += Math.abs(wpd[j]);
			}
			delta += (w_norm_new - w_norm);

			negsum_xTd = 0;
			for (int i = 0; i < l; i++)
				if (y[i] == -1)
					negsum_xTd += C[GETI(y, i)] * xTd[i];

			int num_linesearch;
			for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
				cond = w_norm_new - w_norm + negsum_xTd - sigma * delta;

				for (int i = 0; i < l; i++) {
					final double exp_xTd = Math.exp(xTd[i]);
					exp_wTx_new[i] = exp_wTx[i] * exp_xTd;
					cond += C[GETI(y, i)] * Math.log((1 + exp_wTx_new[i]) / (exp_xTd + exp_wTx_new[i]));
				}

				if (cond <= 0) {
					w_norm = w_norm_new;
					for (j = 0; j < w_size; j++)
						w[j] = wpd[j];
					for (int i = 0; i < l; i++) {
						exp_wTx[i] = exp_wTx_new[i];
						final double tau_tmp = 1 / (1 + exp_wTx[i]);
						tau[i] = C[GETI(y, i)] * tau_tmp;
						D[i] = C[GETI(y, i)] * exp_wTx[i] * tau_tmp * tau_tmp;
					}
					break;
				} else {
					w_norm_new = 0;
					for (j = 0; j < w_size; j++) {
						wpd[j] = (w[j] + wpd[j]) * 0.5;
						if (wpd[j] != 0)
							w_norm_new += Math.abs(wpd[j]);
					}
					delta *= 0.5;
					negsum_xTd *= 0.5;
					for (int i = 0; i < l; i++)
						xTd[i] *= 0.5;
				}
			}

			// Recompute some info due to too many line search steps
			if (num_linesearch >= max_num_linesearch) {
				for (int i = 0; i < l; i++)
					exp_wTx[i] = 0;

				for (int i = 0; i < w_size; i++) {
					if (w[i] == 0)
						continue;
					for (int ind = 0; ind < prob_col.x[j].length; ind++) {
						exp_wTx[ind] += w[i] * prob_col.x[i][ind];
					}
				}

				for (int i = 0; i < l; i++)
					exp_wTx[i] = Math.exp(exp_wTx[i]);
			}

			if (iter == 1)
				inner_eps *= 0.25;

			newton_iter++;
			Gmax_old = Gmax_new;

			info("iter %3d  #CD cycles %d%n", newton_iter, iter);
		}

		info("=========================%n");
		info("optimization finished, #iter = %d%n", newton_iter);
		if (newton_iter >= max_newton_iter)
			info("WARNING: reaching max number of iterations%n");

		// calculate objective value

		double v = 0;
		int nnz = 0;
		for (j = 0; j < w_size; j++)
			if (w[j] != 0) {
				v += Math.abs(w[j]);
				nnz++;
			}
		for (j = 0; j < l; j++)
			if (y[j] == 1)
				v += C[GETI(y, j)] * Math.log(1 + 1 / exp_wTx[j]);
			else
				v += C[GETI(y, j)] * Math.log(1 + exp_wTx[j]);

		info("Objective value = %g%n", v);
		info("#nonzeros/#features = %d/%d%n", nnz, w_size);
	}

	// transpose matrix X from row format to column format
	static DenseProblem transpose(DenseProblem prob) {
		final int l = prob.l;
		final int n = prob.n;
		final DenseProblem prob_col = new DenseProblem();
		prob_col.l = l;
		prob_col.n = n;
		prob_col.y = new double[l];
		prob_col.x = new double[n][];

		for (int i = 0; i < l; i++)
			prob_col.y[i] = prob.y[i];

		for (int i = 0; i < n; i++) {
			prob_col.x[i] = new double[l];
		}

		for (int i = 0; i < l; i++) {
			for (int j = 0; j < n; j++) {
				prob_col.x[j][i] = prob.x[i][j];
			}
		}

		return prob_col;
	}

	static void swap(double[] array, int idxA, int idxB) {
		final double temp = array[idxA];
		array[idxA] = array[idxB];
		array[idxB] = temp;
	}

	static void swap(int[] array, int idxA, int idxB) {
		final int temp = array[idxA];
		array[idxA] = array[idxB];
		array[idxB] = temp;
	}

	static void swap(IntArrayPointer array, int idxA, int idxB) {
		final int temp = array.get(idxA);
		array.set(idxA, array.get(idxB));
		array.set(idxB, temp);
	}

	/**
	 * @throws IllegalArgumentException
	 *             if the feature nodes of prob are not sorted in ascending
	 *             order
	 */
	public static Model train(DenseProblem prob, Parameter param) {

		if (prob == null)
			throw new IllegalArgumentException("problem must not be null");
		if (param == null)
			throw new IllegalArgumentException("parameter must not be null");

		if (prob.n == 0)
			throw new IllegalArgumentException("problem has zero features");
		if (prob.l == 0)
			throw new IllegalArgumentException("problem has zero instances");

		final int l = prob.l;
		final int n = prob.n;
		final int w_size = prob.n;
		final Model model = new Model();

		if (prob.bias >= 0)
			model.nr_feature = n - 1;
		else
			model.nr_feature = n;

		model.solverType = param.solverType;
		model.bias = prob.bias;

		if (param.solverType == SolverType.L2R_L2LOSS_SVR || //
				param.solverType == SolverType.L2R_L1LOSS_SVR_DUAL || //
				param.solverType == SolverType.L2R_L2LOSS_SVR_DUAL)
		{
			model.w = new double[w_size];
			model.nr_class = 2;
			model.label = null;

			checkProblemSize(n, model.nr_class);

			train_one(prob, param, model.w, 0, 0);
		} else {
			final int[] perm = new int[l];

			// group training data of the same class
			final GroupClassesReturn rv = groupClasses(prob, perm);
			final int nr_class = rv.nr_class;
			final int[] label = rv.label;
			final int[] start = rv.start;
			final int[] count = rv.count;

			checkProblemSize(n, nr_class);

			model.nr_class = nr_class;
			model.label = new int[nr_class];
			for (int i = 0; i < nr_class; i++)
				model.label[i] = label[i];

			// calculate weighted C
			final double[] weighted_C = new double[nr_class];
			for (int i = 0; i < nr_class; i++)
				weighted_C[i] = param.C;
			for (int i = 0; i < param.getNumWeights(); i++) {
				int j;
				for (j = 0; j < nr_class; j++)
					if (param.weightLabel[i] == label[j])
						break;

				if (j == nr_class)
					throw new IllegalArgumentException("class label " + param.weightLabel[i]
							+ " specified in weight is not found");
				weighted_C[j] *= param.weight[i];
			}

			// constructing the subproblem
			final double[][] x = new double[l][];
			for (int i = 0; i < l; i++)
				x[i] = prob.x[perm[i]];

			final DenseProblem sub_prob = new DenseProblem();
			sub_prob.l = l;
			sub_prob.n = n;
			sub_prob.x = new double[sub_prob.l][];
			sub_prob.y = new double[sub_prob.l];

			for (int k = 0; k < sub_prob.l; k++)
				sub_prob.x[k] = x[k];

			// multi-class svm by Crammer and Singer
			if (param.solverType == SolverType.MCSVM_CS) {
				model.w = new double[n * nr_class];
				for (int i = 0; i < nr_class; i++) {
					for (int j = start[i]; j < start[i] + count[i]; j++) {
						sub_prob.y[j] = i;
					}
				}

				final DenseSolverMCSVM_CS solver = new DenseSolverMCSVM_CS(sub_prob, nr_class, weighted_C, param.eps);
				solver.solve(model.w);
			} else {
				if (nr_class == 2) {
					model.w = new double[w_size];

					final int e0 = start[0] + count[0];
					int k = 0;
					for (; k < e0; k++)
						sub_prob.y[k] = +1;
					for (; k < sub_prob.l; k++)
						sub_prob.y[k] = -1;

					train_one(sub_prob, param, model.w, weighted_C[0], weighted_C[1]);
				} else {
					model.w = new double[w_size * nr_class];
					final double[] w = new double[w_size];
					for (int i = 0; i < nr_class; i++) {
						final int si = start[i];
						final int ei = si + count[i];

						int k = 0;
						for (; k < si; k++)
							sub_prob.y[k] = -1;
						for (; k < ei; k++)
							sub_prob.y[k] = +1;
						for (; k < sub_prob.l; k++)
							sub_prob.y[k] = -1;

						train_one(sub_prob, param, w, weighted_C[i], param.C);

						for (int j = 0; j < n; j++)
							model.w[j * nr_class + i] = w[j];
					}
				}
			}
		}
		return model;
	}

	/**
	 * verify the size and throw an exception early if the problem is too large
	 */
	private static void checkProblemSize(int n, int nr_class) {
		if (n >= Integer.MAX_VALUE / nr_class || n * nr_class < 0) {
			throw new IllegalArgumentException("'number of classes' * 'number of instances' is too large: " + nr_class
					+ "*" + n);
		}
	}

	private static void train_one(DenseProblem prob, Parameter param, double[] w, double Cp, double Cn) {
		final double eps = param.eps;
		int pos = 0;
		for (int i = 0; i < prob.l; i++)
			if (prob.y[i] > 0) {
				pos++;
			}
		final int neg = prob.l - pos;

		final double primal_solver_tol = eps * Math.max(Math.min(pos, neg), 1) / prob.l;

		Function fun_obj = null;
		switch (param.solverType) {
		case L2R_LR: {
			final double[] C = new double[prob.l];
			for (int i = 0; i < prob.l; i++) {
				if (prob.y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj = new DenseL2R_LrFunction(prob, C);
			final Tron tron_obj = new Tron(fun_obj, primal_solver_tol);
			tron_obj.tron(w);
			break;
		}
		case L2R_L2LOSS_SVC: {
			final double[] C = new double[prob.l];
			for (int i = 0; i < prob.l; i++) {
				if (prob.y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj = new DenseL2R_L2_SvcFunction(prob, C);
			final Tron tron_obj = new Tron(fun_obj, primal_solver_tol);
			tron_obj.tron(w);
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, SolverType.L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, SolverType.L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC: {
			final DenseProblem prob_col = transpose(prob);
			solve_l1r_l2_svc(prob_col, w, primal_solver_tol, Cp, Cn);
			break;
		}
		case L1R_LR: {
			final DenseProblem prob_col = transpose(prob);
			solve_l1r_lr(prob_col, w, primal_solver_tol, Cp, Cn);
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR: {
			final double[] C = new double[prob.l];
			for (int i = 0; i < prob.l; i++)
				C[i] = param.C;

			fun_obj = new DenseL2R_L2_SvrFunction(prob, C, param.p);
			final Tron tron_obj = new Tron(fun_obj, param.eps);
			tron_obj.tron(w);
			break;
		}
		case L2R_L1LOSS_SVR_DUAL:
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param);
			break;

		default:
			throw new IllegalStateException("unknown solver type: " + param.solverType);
		}
	}

	public static void disableDebugOutput() {
		setDebugOutput(null);
	}

	public static void enableDebugOutput() {
		setDebugOutput(System.out);
	}

	public static void setDebugOutput(PrintStream debugOutput) {
		synchronized (OUTPUT_MUTEX) {
			DEBUG_OUTPUT = debugOutput;
		}
	}

	/**
	 * resets the PRNG
	 * 
	 * this is i.a. needed for regression testing (eg. the Weka wrapper)
	 */
	public static void resetRandom() {
		random = new Random(DEFAULT_RANDOM_SEED);
	}
}
