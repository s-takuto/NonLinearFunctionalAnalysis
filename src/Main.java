package src;

import java.util.Scanner;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

import java.lang.Math;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Main {
	// 停止条件ε
	public final double EPSILON = 0.000001;

	// constant 
	public RealVector a;

	// matrix(32 * 32)
	public RealMatrix m;

	// mの逆行列
	public RealMatrix inv;

	// 最小値を保存しておく配列
	public List<Double> ans;


	// mの逆行列を計算するメソッド
	public void inverse(RealMatrix m) {
		inv = m.transpose().scalarMultiply(4);
	}

	// f(x) := 1 / 2 * ||x - a|| ^ 2
	public double f(ArrayRealVector x) {
		ArrayRealVector tmp = x.subtract(a);
		double norm = tmp.getNorm();
		return (norm * norm * 0.5d);
	}

	// g(x) := 1 / 2 * ||Ax - a|| ^ 2
	public double g(ArrayRealVector x) {
		RealVector rv = inv.operate(x);
		double norm = (rv.subtract(a)).getNorm();
		return (norm * norm * 0.5d);
	}

	// minimize する関数
	public double objectiveF(ArrayRealVector x) {
		return f(x) + x.getL1Norm();
	}

	// minimize する関数(行列ver)
	public double objectiveG(ArrayRealVector x) {
		return g(x) + x.getL1Norm();
	}

	// 関数fの勾配(gradient)
	public ArrayRealVector nablaF(ArrayRealVector x) {
		return x.subtract(a);
	}

	// 関数gの勾配(gradient)
	public RealVector nablaG(ArrayRealVector x) {
		RealVector rv = (m.operate(x)).subtract(a);
		return (m.transpose()).operate(rv);
	}

	// リゾルベント(resolvent)
	public ArrayRealVector resolvent(ArrayRealVector x) {
		int n = x.getDimension();
		ArrayRealVector tmp = new ArrayRealVector(n);

		for(int i = 0; i < n; i++) {
			double v = x.getEntry(i);
			if(v < -1) {
				tmp.setEntry(i, v + 1);
			}
			else if(v > 1) {
				tmp.setEntry(i, v - 1);
			}
			else {
				tmp.setEntry(i, 0);
			}
		}

		return tmp;
	}

	// Douglas-Rachford algorithm
	public void douglasRachfordAlgorithm(ArrayRealVector x, ArrayRealVector alpha) {
		ans = new ArrayList<>();
		ans.add(objectiveF(x));
		ArrayRealVector pre = x;
		ArrayRealVector next = new ArrayRealVector();

		while(true) {
			next = resolvent(pre.subtract(alpha.ebeMultiply(nablaF(pre))));
			ans.add(objectiveF(next));
			double diff = objectiveF(next) - objectiveF(pre);
			System.out.println("pre:" + pre + ", next:" + next + ", diff:" + diff);
			if(Math.abs(diff) < EPSILON) {
				break;
			}
			pre = next;
		}
	}

	// Douglas-Rachford algorithm(行列)
	public void douglasRachfordAlgorithm1(ArrayRealVector x, ArrayRealVector alpha) {
		ans = new ArrayList<>();
		ans.add(objectiveG(x));
		ArrayRealVector pre = x;
		ArrayRealVector next = new ArrayRealVector();

		while(true) {
			next = resolvent(pre.subtract(alpha.ebeMultiply(nablaG(pre))));
			ans.add(objectiveG(next));
			double diff = objectiveG(next) - objectiveG(pre);
			//	System.out.println("pre:" + pre + ", next:" + next + ", diff:" + diff);
			//	System.out.println("g:" + objectiveG(next) + ", x:" + x);
			if(Math.abs(diff) < EPSILON) {
				break;
			}
			pre = next;
		}
	}

	public void run() {
		Scanner sc;
		// 次元数
		int n = 1024;

		// ある定数a, 初期点x, パラメータα, 停止条件ε の入力
		// ある定数a
		try {
			String name = "instance/constant.txt";
			File file = new File(name);
			sc = new Scanner(file);

			a = new ArrayRealVector(n);
			for(int i = 0; i < n; i++) {
				double val = sc.nextDouble();
				a.setEntry(i, val);
			}

		} catch (FileNotFoundException e) {
			System.out.println(e);
		}

		// 行列m
		try {
			String name = "instance/M.txt";
			File file = new File(name);
			sc = new Scanner(file);

			int N = 32;
			m = MatrixUtils.createRealMatrix(n, n);
			for(int i = 0; i < N; i++) {
				for(int j = 0; j < N; j++) {
					double val = sc.nextDouble();
					m.setEntry(i, j, val);
				}
			}

		} catch (FileNotFoundException e) {
			System.out.println(e);
		}

		// 行列mの逆行列
		inverse(m);

		sc = new Scanner(System.in);

		// 初期点
		ArrayRealVector x = new ArrayRealVector(n);
		for(int i = 0; i < n; i++) {
			x.setEntry(i, sc.nextDouble());
		}

		// パラメータα
		// 書き換える
		ArrayRealVector alpha = new ArrayRealVector(n);
		alpha.set(0.75);

		// algorithmを走らせる
		long start = System.currentTimeMillis();
		douglasRachfordAlgorithm1(x, alpha);
		long end = System.currentTimeMillis();

		System.out.println();
		System.out.println("answer");
		System.out.println("time:" + (end - start) + "ms");
		int count = 0;
		for(Double d : ans) {
			count++;
			System.out.println(count + "," + d.toString());
		}
	}

	public static void main(String[] args) {
		new Main().run();
	}
}
