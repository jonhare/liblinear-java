package de.bwaldvogel.denseliblinear;

import static org.fest.assertions.Assertions.assertThat;
import static org.fest.assertions.Fail.fail;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.fest.assertions.Delta;
import org.junit.BeforeClass;
import org.junit.Test;
import org.powermock.api.mockito.PowerMockito;

public class LinearTest {

	private static Random random = new Random(12345);

	@BeforeClass
	public static void disableDebugOutput() {
		// Linear.disableDebugOutput();
	}

	public static Model createRandomModel() {
		final Model model = new Model();
		model.solverType = SolverType.L2R_LR;
		model.bias = 2;
		model.label = new int[] { 1, Integer.MAX_VALUE, 2 };
		model.w = new double[model.label.length * 300];
		for (int i = 0; i < model.w.length; i++) {
			// precision should be at least 1e-4
			model.w[i] = Math.round(random.nextDouble() * 100000.0) / 10000.0;
		}

		// force at least one value to be zero
		model.w[random.nextInt(model.w.length)] = 0.0;
		model.w[random.nextInt(model.w.length)] = -0.0;

		model.nr_feature = model.w.length / model.label.length - 1;
		model.nr_class = model.label.length;
		return model;
	}

	public static Problem createRandomProblem(int numClasses) {
		final Problem prob = new Problem();
		prob.bias = -1;
		prob.l = random.nextInt(100) + 1;
		prob.n = random.nextInt(100) + 1;
		prob.x = new double[prob.l][];
		prob.y = new double[prob.l];

		for (int i = 0; i < prob.l; i++) {

			prob.y[i] = random.nextInt(numClasses);

			final Set<Integer> randomNumbers = new TreeSet<Integer>();
			final int num = random.nextInt(prob.n);
			for (int j = 0; j < num; j++) {
				randomNumbers.add(random.nextInt(prob.n));
			}
			final List<Integer> randomIndices = new ArrayList<Integer>(randomNumbers);
			Collections.sort(randomIndices);

			prob.x[i] = new double[prob.n];
			for (int j = 0; j < randomIndices.size(); j++) {
				prob.x[i][randomIndices.get(j)] = random.nextDouble();
			}
		}
		return prob;
	}

	/**
	 * create a very simple problem and check if the clearly separated examples
	 * are recognized as such
	 */
	@Test
	public void testTrainPredict() {
		final Problem prob = new Problem();
		prob.bias = -1;
		prob.l = 4;
		prob.n = 4;
		prob.x = new double[4][4];

		prob.x[0][0] = 1;
		prob.x[0][1] = 1;

		prob.x[1][2] = 1;
		prob.x[2][2] = 1;

		prob.x[3][0] = 2;
		prob.x[3][1] = 1;
		prob.x[3][3] = 1;

		prob.y = new double[4];
		prob.y[0] = 0;
		prob.y[1] = 1;
		prob.y[2] = 1;
		prob.y[3] = 0;

		for (final SolverType solver : SolverType.values()) {
			for (double C = 0.1; C <= 100.; C *= 1.2) {
				// compared the behavior with the C version
				if (C < 0.2)
					if (solver == SolverType.L1R_L2LOSS_SVC)
						continue;
				if (C < 0.7)
					if (solver == SolverType.L1R_LR)
						continue;

				if (solver.isSupportVectorRegression()) {
					continue;
				}

				final Parameter param = new Parameter(solver, C, 0.1, 0.1);
				final Model model = Linear.train(prob, param);

				final double[] featureWeights = model.getFeatureWeights();
				if (solver == SolverType.MCSVM_CS) {
					assertThat(featureWeights.length).isEqualTo(8);
				} else {
					assertThat(featureWeights.length).isEqualTo(4);
				}

				int i = 0;
				for (final double value : prob.y) {
					final double prediction = Linear.predict(model, prob.x[i]);
					assertThat(prediction).as("prediction with solver " + solver).isEqualTo(value);
					if (model.isProbabilityModel()) {
						final double[] estimates = new double[model.getNrClass()];
						final double probabilityPrediction = Linear.predictProbability(model, prob.x[i], estimates);
						assertThat(probabilityPrediction).isEqualTo(prediction);
						assertThat(estimates[(int) probabilityPrediction]).isGreaterThanOrEqualTo(
								1.0 / model.getNrClass());
						double estimationSum = 0;
						for (final double estimate : estimates) {
							estimationSum += estimate;
						}
						assertThat(estimationSum).isEqualTo(1.0, Delta.delta(0.001));
					}
					i++;
				}
			}
		}
	}

	@Test
	public void testCrossValidation() throws Exception {

		final int numClasses = random.nextInt(10) + 1;

		final Problem prob = createRandomProblem(numClasses);

		final Parameter param = new Parameter(SolverType.L2R_LR, 10, 0.01);
		final int nr_fold = 10;
		final double[] target = new double[prob.l];
		Linear.crossValidation(prob, param, nr_fold, target);

		for (final double clazz : target) {
			assertThat(clazz).isGreaterThanOrEqualTo(0).isLessThan(numClasses);
		}
	}

	@Test
	public void testLoadSaveModel() throws Exception {

		Model model = null;
		for (final SolverType solverType : SolverType.values()) {
			model = createRandomModel();
			model.solverType = solverType;

			final File tempFile = File.createTempFile("liblinear", "modeltest");
			tempFile.deleteOnExit();
			Linear.saveModel(tempFile, model);

			final Model loadedModel = Linear.loadModel(tempFile);
			assertThat(loadedModel).isEqualTo(model);
		}
	}

	@Test
	public void testPredictProbabilityWrongSolver() throws Exception {
		final Problem prob = new Problem();
		prob.l = 1;
		prob.n = 1;
		prob.x = new double[prob.l][prob.n];
		prob.y = new double[prob.l];
		for (int i = 0; i < prob.l; i++) {
			prob.y[i] = i;
		}

		final SolverType solverType = SolverType.L2R_L1LOSS_SVC_DUAL;
		final Parameter param = new Parameter(solverType, 10, 0.1);
		final Model model = Linear.train(prob, param);
		try {
			Linear.predictProbability(model, prob.x[0], new double[1]);
			fail("IllegalArgumentException expected");
		} catch (final IllegalArgumentException e) {
			assertThat(e.getMessage()).isEqualTo("probability output is only supported for logistic regression." //
					+ " This is currently only supported by the following solvers:" //
					+ " L2R_LR, L1R_LR, L2R_LR_DUAL");
		}
	}

	@Test
	public void testRealloc() {

		int[] f = new int[] { 1, 2, 3 };
		f = Linear.copyOf(f, 5);
		f[3] = 4;
		f[4] = 5;
		assertThat(f).isEqualTo(new int[] { 1, 2, 3, 4, 5 });
	}

	@Test
	public void testAtoi() {
		assertThat(Linear.atoi("+25")).isEqualTo(25);
		assertThat(Linear.atoi("-345345")).isEqualTo(-345345);
		assertThat(Linear.atoi("+0")).isEqualTo(0);
		assertThat(Linear.atoi("0")).isEqualTo(0);
		assertThat(Linear.atoi("2147483647")).isEqualTo(Integer.MAX_VALUE);
		assertThat(Linear.atoi("-2147483648")).isEqualTo(Integer.MIN_VALUE);
	}

	@Test(expected = NumberFormatException.class)
	public void testAtoiInvalidData() {
		Linear.atoi("+");
	}

	@Test(expected = NumberFormatException.class)
	public void testAtoiInvalidData2() {
		Linear.atoi("abc");
	}

	@Test(expected = NumberFormatException.class)
	public void testAtoiInvalidData3() {
		Linear.atoi(" ");
	}

	@Test
	public void testAtof() {
		assertThat(Linear.atof("+25")).isEqualTo(25);
		assertThat(Linear.atof("-25.12345678")).isEqualTo(-25.12345678);
		assertThat(Linear.atof("0.345345299")).isEqualTo(0.345345299);
	}

	@Test(expected = NumberFormatException.class)
	public void testAtofInvalidData() {
		Linear.atof("0.5t");
	}

	@Test
	public void testSaveModelWithIOException() throws Exception {
		final Model model = createRandomModel();

		final Writer out = PowerMockito.mock(Writer.class);

		final IOException ioException = new IOException("some reason");

		doThrow(ioException).when(out).flush();

		try {
			Linear.saveModel(out, model);
			fail("IOException expected");
		} catch (final IOException e) {
			assertThat(e).isEqualTo(ioException);
		}

		verify(out).flush();
		verify(out, times(1)).close();
	}

	/**
	 * compared input/output values with the C version (1.51)
	 * 
	 * <pre>
	 * IN:
	 * res prob.l = 4
	 * res prob.n = 4
	 * 0: (2,1) (4,1)
	 * 1: (1,1)
	 * 2: (3,1)
	 * 3: (2,2) (3,1) (4,1)
	 * 
	 * TRANSPOSED:
	 * 
	 * res prob.l = 4
	 * res prob.n = 4
	 * 0: (2,1)
	 * 1: (1,1) (4,2)
	 * 2: (3,1) (4,1)
	 * 3: (1,1) (4,1)
	 * </pre>
	 */
	@Test
	public void testTranspose() throws Exception {
		final Problem prob = new Problem();
		prob.bias = -1;
		prob.l = 4;
		prob.n = 4;
		prob.x = new double[4][4];

		prob.x[0][1] = 1;
		prob.x[0][3] = 1;

		prob.x[1][0] = 1;
		prob.x[2][2] = 1;

		prob.x[3][1] = 2;
		prob.x[3][2] = 1;
		prob.x[3][3] = 1;

		prob.y = new double[4];
		prob.y[0] = 0;
		prob.y[1] = 1;
		prob.y[2] = 1;
		prob.y[3] = 0;

		final Problem transposed = Linear.transpose(prob);

		assertThat(transposed.x[0].length).isEqualTo(4);
		assertThat(transposed.x[1].length).isEqualTo(4);
		assertThat(transposed.x[2].length).isEqualTo(4);
		assertThat(transposed.x[3].length).isEqualTo(4);

		assertThat(transposed.x[0][1]).isEqualTo(1);

		assertThat(transposed.x[1][0]).isEqualTo(1);
		assertThat(transposed.x[1][3]).isEqualTo(2);

		assertThat(transposed.x[2][2]).isEqualTo(1);
		assertThat(transposed.x[2][3]).isEqualTo(1);

		assertThat(transposed.x[3][0]).isEqualTo(1);
		assertThat(transposed.x[3][3]).isEqualTo(1);

		assertThat(transposed.y).isEqualTo(prob.y);
	}

	/**
	 * 
	 * compared input/output values with the C version (1.51)
	 * 
	 * <pre>
	 * IN:
	 * res prob.l = 5
	 * res prob.n = 10
	 * 0: (1,7) (3,3) (5,2)
	 * 1: (2,1) (4,5) (5,3) (7,4) (8,2)
	 * 2: (1,9) (3,1) (5,1) (10,7)
	 * 3: (1,2) (2,2) (3,9) (4,7) (5,8) (6,1) (7,5) (8,4)
	 * 4: (3,1) (10,3)
	 * 
	 * TRANSPOSED:
	 * 
	 * res prob.l = 5
	 * res prob.n = 10
	 * 0: (1,7) (3,9) (4,2)
	 * 1: (2,1) (4,2)
	 * 2: (1,3) (3,1) (4,9) (5,1)
	 * 3: (2,5) (4,7)
	 * 4: (1,2) (2,3) (3,1) (4,8)
	 * 5: (4,1)
	 * 6: (2,4) (4,5)
	 * 7: (2,2) (4,4)
	 * 8:
	 * 9: (3,7) (5,3)
	 * </pre>
	 */
	@Test
	public void testTranspose2() throws Exception {
		final Problem prob = new Problem();
		prob.bias = -1;
		prob.l = 5;
		prob.n = 10;
		prob.x = new double[5][10];

		prob.x[0][0] = 7;
		prob.x[0][2] = 3;
		prob.x[0][4] = 2;

		prob.x[1][1] = 1;
		prob.x[1][3] = 5;
		prob.x[1][4] = 3;
		prob.x[1][6] = 4;
		prob.x[1][7] = 2;

		prob.x[2][0] = 9;
		prob.x[2][2] = 1;
		prob.x[2][4] = 1;
		prob.x[2][9] = 7;

		prob.x[3][0] = 2;
		prob.x[3][1] = 2;
		prob.x[3][2] = 9;
		prob.x[3][3] = 7;
		prob.x[3][4] = 8;
		prob.x[3][5] = 1;
		prob.x[3][6] = 5;
		prob.x[3][7] = 4;

		prob.x[4][2] = 1;
		prob.x[4][9] = 3;

		prob.y = new double[5];
		prob.y[0] = 0;
		prob.y[1] = 1;
		prob.y[2] = 1;
		prob.y[3] = 0;
		prob.y[4] = 1;

		final Problem transposed = Linear.transpose(prob);

		assertThat(transposed.x[0]).hasSize(5);
		assertThat(transposed.x[1]).hasSize(5);
		assertThat(transposed.x[2]).hasSize(5);
		assertThat(transposed.x[3]).hasSize(5);
		assertThat(transposed.x[4]).hasSize(5);
		assertThat(transposed.x[5]).hasSize(5);
		assertThat(transposed.x[7]).hasSize(5);
		assertThat(transposed.x[7]).hasSize(5);
		assertThat(transposed.x[8]).hasSize(5);
		assertThat(transposed.x[9]).hasSize(5);

		assertThat(transposed.x[0][0]).isEqualTo(7);
		assertThat(transposed.x[0][2]).isEqualTo(9);
		assertThat(transposed.x[0][3]).isEqualTo(2);

		assertThat(transposed.x[1][1]).isEqualTo(1);
		assertThat(transposed.x[1][3]).isEqualTo(2);

		assertThat(transposed.x[2][0]).isEqualTo(3);
		assertThat(transposed.x[2][2]).isEqualTo(1);
		assertThat(transposed.x[2][3]).isEqualTo(9);
		assertThat(transposed.x[2][4]).isEqualTo(1);

		assertThat(transposed.x[3][1]).isEqualTo(5);
		assertThat(transposed.x[3][3]).isEqualTo(7);

		assertThat(transposed.x[4][0]).isEqualTo(2);
		assertThat(transposed.x[4][1]).isEqualTo(3);
		assertThat(transposed.x[4][2]).isEqualTo(1);
		assertThat(transposed.x[4][3]).isEqualTo(8);

		assertThat(transposed.x[5][3]).isEqualTo(1);

		assertThat(transposed.x[6][1]).isEqualTo(4);
		assertThat(transposed.x[6][3]).isEqualTo(5);

		assertThat(transposed.x[7][1]).isEqualTo(2);
		assertThat(transposed.x[7][3]).isEqualTo(4);

		assertThat(transposed.x[9][2]).isEqualTo(7);
		assertThat(transposed.x[9][4]).isEqualTo(3);

		assertThat(transposed.y).isEqualTo(prob.y);
	}

	/**
	 * compared input/output values with the C version (1.51)
	 * 
	 * IN: res prob.l = 3 res prob.n = 4 0: (1,2) (3,1) (4,3) 1: (1,9) (2,7)
	 * (3,3) (4,3) 2: (2,1)
	 * 
	 * TRANSPOSED:
	 * 
	 * res prob.l = 3 * res prob.n = 4 0: (1,2) (2,9) 1: (2,7) (3,1) 2: (1,1)
	 * (2,3) 3: (1,3) (2,3)
	 * 
	 */
	@Test
	public void testTranspose3() throws Exception {

		final Problem prob = new Problem();
		prob.l = 3;
		prob.n = 4;
		prob.y = new double[3];
		prob.x = new double[3][4];

		prob.x[0][0] = 2;
		prob.x[0][2] = 1;
		prob.x[0][3] = 3;
		prob.x[1][0] = 9;
		prob.x[1][1] = 7;
		prob.x[1][2] = 3;
		prob.x[1][3] = 3;

		prob.x[2][1] = 1;

		final Problem transposed = Linear.transpose(prob);
		assertThat(transposed.x).hasSize(4);
		assertThat(transposed.x[0]).hasSize(3);
		assertThat(transposed.x[1]).hasSize(3);
		assertThat(transposed.x[2]).hasSize(3);
		assertThat(transposed.x[3]).hasSize(3);

		assertThat(transposed.x[0][0]).isEqualTo(2);
		assertThat(transposed.x[0][1]).isEqualTo(9);

		assertThat(transposed.x[1][1]).isEqualTo(7);
		assertThat(transposed.x[1][2]).isEqualTo(1);

		assertThat(transposed.x[2][0]).isEqualTo(1);
		assertThat(transposed.x[2][1]).isEqualTo(3);

		assertThat(transposed.x[3][0]).isEqualTo(3);
		assertThat(transposed.x[3][1]).isEqualTo(3);
	}
}
