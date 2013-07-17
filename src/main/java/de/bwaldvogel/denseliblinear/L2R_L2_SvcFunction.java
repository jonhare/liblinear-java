package de.bwaldvogel.denseliblinear;

class L2R_L2_SvcFunction implements Function {

	protected final Problem prob;
	protected final double[] C;
	protected final int[] I;
	protected final double[] z;

	protected int sizeI;

	public L2R_L2_SvcFunction(Problem prob, double[] C) {
		final int l = prob.l;

		this.prob = prob;

		z = new double[l];
		I = new int[l];
		this.C = C;
	}

	@Override
	public double fun(double[] w) {
		int i;
		double f = 0;
		final double[] y = prob.y;
		final int l = prob.l;
		final int w_size = get_nr_variable();

		Xv(w, z);

		for (i = 0; i < w_size; i++)
			f += w[i] * w[i];
		f /= 2.0;
		for (i = 0; i < l; i++) {
			z[i] = y[i] * z[i];
			final double d = 1 - z[i];
			if (d > 0)
				f += C[i] * d * d;
		}

		return (f);
	}

	@Override
	public int get_nr_variable() {
		return prob.n;
	}

	@Override
	public void grad(double[] w, double[] g) {
		final double[] y = prob.y;
		final int l = prob.l;
		final int w_size = get_nr_variable();

		sizeI = 0;
		for (int i = 0; i < l; i++) {
			if (z[i] < 1) {
				z[sizeI] = C[i] * y[i] * (z[i] - 1);
				I[sizeI] = i;
				sizeI++;
			}
		}
		subXTv(z, g);

		for (int i = 0; i < w_size; i++)
			g[i] = w[i] + 2 * g[i];
	}

	@Override
	public void Hv(double[] s, double[] Hs) {
		int i;
		final int w_size = get_nr_variable();
		final double[] wa = new double[sizeI];

		subXv(s, wa);
		for (i = 0; i < sizeI; i++)
			wa[i] = C[I[i]] * wa[i];

		subXTv(wa, Hs);
		for (i = 0; i < w_size; i++)
			Hs[i] = s[i] + 2 * Hs[i];
	}

	protected void subXTv(double[] v, double[] XTv) {
		int i;
		final int w_size = get_nr_variable();

		for (i = 0; i < w_size; i++)
			XTv[i] = 0;

		for (i = 0; i < sizeI; i++) {
			for (int j = 0; j < prob.x[I[i]].length; j++) {
				XTv[j] += v[i] * prob.x[I[i]][j];
			}
		}
	}

	private void subXv(double[] v, double[] Xv) {
		for (int i = 0; i < sizeI; i++) {
			Xv[i] = 0;

			for (int j = 0; j < prob.x[I[i]].length; j++) {
				Xv[i] += v[j] * prob.x[I[i]][j];
			}
		}
	}

	protected void Xv(double[] v, double[] Xv) {
		for (int i = 0; i < prob.l; i++) {
			Xv[i] = 0;
			for (int j = 0; j < prob.x[i].length; j++) {
				Xv[i] += v[j] * prob.x[i][j];
			}
		}
	}
}
