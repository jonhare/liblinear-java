package de.bwaldvogel.denseliblinear;

import static org.fest.assertions.Assertions.assertThat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;

import org.junit.Test;

public class TrainTest {

	@Test
	public void testParseCommandLine() {
		final Train train = new Train();

		for (final SolverType solver : SolverType.values()) {
			train.parse_command_line(new String[] { "-B", "5.3", "-s", "" + solver.getId(), "-p", "0.01", "model-filename" });
			final Parameter param = train.getParameter();
			assertThat(param.solverType).isEqualTo(solver);
			// check default eps
			if (solver.getId() == 0 || solver.getId() == 2 //
					|| solver.getId() == 5 || solver.getId() == 6)
			{
				assertThat(param.eps).isEqualTo(0.01);
			} else if (solver.getId() == 7) {
				assertThat(param.eps).isEqualTo(0.1);
			} else if (solver.getId() == 11) {
				assertThat(param.eps).isEqualTo(0.001);
			} else {
				assertThat(param.eps).isEqualTo(0.1);
			}
			// check if bias is set
			assertThat(train.getBias()).isEqualTo(5.3);
			assertThat(param.p).isEqualTo(0.01);
		}
	}

	@Test
	// https://github.com/bwaldvogel/liblinear-java/issues/4
	public void
			testParseWeights() throws Exception
	{
		final Train train = new Train();
		train.parse_command_line(new String[] { "-v", "10", "-c", "10", "-w1", "1.234", "model-filename" });
		Parameter parameter = train.getParameter();
		assertThat(parameter.weightLabel).isEqualTo(new int[] { 1 });
		assertThat(parameter.weight).isEqualTo(new double[] { 1.234 });

		train.parse_command_line(new String[] { "-w1", "1.234", "-w2", "0.12", "-w3", "7", "model-filename" });
		parameter = train.getParameter();
		assertThat(parameter.weightLabel).isEqualTo(new int[] { 1, 2, 3 });
		assertThat(parameter.weight).isEqualTo(new double[] { 1.234, 0.12, 7 });
	}

	@Test
	public void testReadProblem() throws Exception {

		final File file = File.createTempFile("svm", "test");
		file.deleteOnExit();

		final Collection<String> lines = new ArrayList<String>();
		lines.add("1 1:1  3:1  4:1   6:1");
		lines.add("2 2:1  3:1  5:1   7:1");
		lines.add("1 3:1  5:1");
		lines.add("1 1:1  4:1  7:1");
		lines.add("2 4:1  5:1  7:1");
		final BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		try {
			for (final String line : lines)
				writer.append(line).append("\n");
		} finally {
			writer.close();
		}

		final Train train = new Train();
		train.readProblem(file.getAbsolutePath());

		final Problem prob = train.getProblem();
		assertThat(prob.bias).isEqualTo(1);
		assertThat(prob.y).hasSize(lines.size());
		assertThat(prob.y).isEqualTo(new double[] { 1, 2, 1, 1, 2 });
		assertThat(prob.n).isEqualTo(8);
		assertThat(prob.l).isEqualTo(prob.y.length);
		assertThat(prob.x).hasSize(prob.y.length);

		for (final double[] nodes : prob.x) {

			assertThat(nodes.length).isLessThanOrEqualTo(prob.n);
			for (int ind = 0; ind < prob.n; ind++) {
				// bias term
				if (prob.bias >= 0 && ind == prob.n - 1) {
					// assertThat(ind).isEqualTo(prob.n);
					assertThat(nodes[ind]).isEqualTo(prob.bias);
				} else {
					assertThat(ind).isLessThan(prob.n);
				}
			}
		}
	}

	/**
	 * unit-test for Issue #1
	 * (http://github.com/bwaldvogel/liblinear-java/issues#issue/1)
	 */
	@Test
	public void testReadProblemEmptyLine() throws Exception {

		final File file = File.createTempFile("svm", "test");
		file.deleteOnExit();

		final Collection<String> lines = new ArrayList<String>();
		lines.add("1 1:1  3:1  4:1   6:1");
		lines.add("2 ");
		final BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		try {
			for (final String line : lines)
				writer.append(line).append("\n");
		} finally {
			writer.close();
		}

		final Problem prob = Train.readProblem(file, -1.0);
		assertThat(prob.bias).isEqualTo(-1);
		assertThat(prob.y).hasSize(lines.size());
		assertThat(prob.y).isEqualTo(new double[] { 1, 2 });
		assertThat(prob.n).isEqualTo(6);
		assertThat(prob.l).isEqualTo(prob.y.length);
		assertThat(prob.x).hasSize(prob.y.length);

		assertThat(prob.x[0]).hasSize(6);
		assertThat(prob.x[1]).hasSize(6);
	}

	@Test(expected = InvalidInputDataException.class)
	public void testReadUnsortedProblem() throws Exception {
		final File file = File.createTempFile("svm", "test");
		file.deleteOnExit();

		final Collection<String> lines = new ArrayList<String>();
		lines.add("1 1:1  3:1  4:1   6:1");
		lines.add("2 2:1  3:1  5:1   7:1");
		lines.add("1 3:1  5:1  4:1"); // here's the mistake: not correctly
										// sorted

		final BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		try {
			for (final String line : lines)
				writer.append(line).append("\n");
		} finally {
			writer.close();
		}

		final Train train = new Train();
		train.readProblem(file.getAbsolutePath());
	}

	@Test(expected = InvalidInputDataException.class)
	public void testReadProblemWithInvalidIndex() throws Exception {
		final File file = File.createTempFile("svm", "test");
		file.deleteOnExit();

		final Collection<String> lines = new ArrayList<String>();
		lines.add("1 1:1  3:1  4:1   6:1");
		lines.add("2 2:1  3:1  5:1  -4:1");

		final BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		try {
			for (final String line : lines)
				writer.append(line).append("\n");
		} finally {
			writer.close();
		}

		final Train train = new Train();
		try {
			train.readProblem(file.getAbsolutePath());
		} catch (final InvalidInputDataException e) {
			throw e;
		}
	}

	@Test(expected = InvalidInputDataException.class)
	public void testReadWrongProblem() throws Exception {
		final File file = File.createTempFile("svm", "test");
		file.deleteOnExit();

		final Collection<String> lines = new ArrayList<String>();
		lines.add("1 1:1  3:1  4:1   6:1");
		lines.add("2 2:1  3:1  5:1   7:1");
		lines.add("1 3:1  5:a"); // here's the mistake: incomplete line

		final BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		try {
			for (final String line : lines)
				writer.append(line).append("\n");
		} finally {
			writer.close();
		}

		final Train train = new Train();
		try {
			train.readProblem(file.getAbsolutePath());
		} catch (final InvalidInputDataException e) {
			throw e;
		}
	}
}
