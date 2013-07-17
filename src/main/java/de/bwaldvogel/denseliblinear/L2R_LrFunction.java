package de.bwaldvogel.denseliblinear;

class L2R_LrFunction implements Function {

	private final double[] C;
	private final double[] z;
	private final double[] D;
	private final Problem prob;

	public L2R_LrFunction(Problem prob, double[] C) {
		final int l = prob.l;

		this.prob = prob;

		z = new double[l];
		D = new double[l];
		this.C = C;
	}

	private void Xv(double[] v, double[] Xv) {
		for (int i = 0; i < prob.l; i++) {
			Xv[i] = 0;
			for (int j = 0; j < prob.x[i].length; j++) {
				Xv[i] += v[j] * prob.x[i][j];
			}
		}
	}

	private void XTv(double[] v, double[] XTv) {
		final int l = prob.l;
		final int w_size = get_nr_variable();
		final double[][] x = prob.x;

		for (int i = 0; i < w_size; i++)
			XTv[i] = 0;

		for (int i = 0; i < l; i++) {
			for (int j = 0; j < prob.x[i].length; j++) {
				XTv[j] += v[i] * x[i][j];
			}
		}
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
			final double yz = y[i] * z[i];
			if (yz >= 0)
				f += C[i] * Math.log(1 + Math.exp(-yz));
			else
				f += C[i] * (-yz + Math.log(1 + Math.exp(yz)));
		}

		return (f);
	}

	@Override
	public void grad(double[] w, double[] g) {
		int i;
		final double[] y = prob.y;
		final int l = prob.l;
		final int w_size = get_nr_variable();

		for (i = 0; i < l; i++) {
			z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
			D[i] = z[i] * (1 - z[i]);
			z[i] = C[i] * (z[i] - 1) * y[i];
		}
		XTv(z, g);

		for (i = 0; i < w_size; i++)
			g[i] = w[i] + g[i];
	}

	@Override
	public void Hv(double[] s, double[] Hs) {
		int i;
		final int l = prob.l;
		final int w_size = get_nr_variable();
		final double[] wa = new double[l];

		Xv(s, wa);
		for (i = 0; i < l; i++)
			wa[i] = C[i] * D[i] * wa[i];

		XTv(wa, Hs);
		for (i = 0; i < w_size; i++)
			Hs[i] = s[i] + Hs[i];
		// delete[] wa;
	}

	@Override
	public int get_nr_variable() {
		return prob.n;
	}

}
