#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <map>
#include <mutex>
#include <time.h>
#include <string>
#include"gurobi_c++.h"
using namespace std;


void ModelXOR2(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& o)
{
	model.addConstr(a + b + (1 - o) >= 1);
	model.addConstr(a + (1 - b) + o >= 1);
	model.addConstr((1 - a) + b + o >= 1);
	model.addConstr((1 - a) + (1 - b) + (1 - o) >= 1);
}
void ModelXOR3(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& c, GRBVar& o)
{
	model.addConstr(a + b + c + (1 - o) >= 1);
	model.addConstr(a + b + (1 - c) + o >= 1);
	model.addConstr(a + (1 - b) + c + o >= 1);
	model.addConstr(a + (1 - b) + (1 - c) + (1 - o) >= 1);
	model.addConstr((1 - a) + b + c + o >= 1);
	model.addConstr((1 - a) + b + (1 - c) + (1 - o) >= 1);
	model.addConstr((1 - a) + (1 - b) + c + (1 - o) >= 1);
	model.addConstr((1 - a) + (1 - b) + (1 - c) + o >= 1);
}
void ModelXOR4(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& c, GRBVar& d, GRBVar& o)
{
	model.addConstr(a + b + c + d + (1 - o) >= 1);
	model.addConstr(a + b + c + (1 - d) + o >= 1);
	model.addConstr(a + b + (1 - c) + d + o >= 1);
	model.addConstr(a + b + (1 - c) + (1 - d) + (1 - o) >= 1);
	model.addConstr(a + (1 - b) + c + d + o >= 1);
	model.addConstr(a + (1 - b) + c + (1 - d) + (1 - o) >= 1);
	model.addConstr(a + (1 - b) + (1 - c) + d + (1 - o) >= 1);
	model.addConstr(a + (1 - b) + (1 - c) + (1 - d) + o >= 1);
	model.addConstr((1 - a) + b + c + d + o >= 1);
	model.addConstr((1 - a) + b + c + (1 - d) + (1 - o) >= 1);
	model.addConstr((1 - a) + b + (1 - c) + d + (1 - o) >= 1);
	model.addConstr((1 - a) + b + (1 - c) + (1 - d) + o >= 1);
	model.addConstr((1 - a) + (1 - b) + c + d + (1 - o) >= 1);
	model.addConstr((1 - a) + (1 - b) + c + (1 - d) + o >= 1);
	model.addConstr((1 - a) + (1 - b) + (1 - c) + d + o >= 1);
	model.addConstr((1 - a) + (1 - b) + (1 - c) + (1 - d) + (1 - o) >= 1);
}

// sum(\Delta x_i \vee \Delta x_{i+1} )
void ModelProb(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& o)
{
	model.addConstr(o - a >= 0);
	model.addConstr(o - b >= 0);
	model.addConstr(a + b - o >= 0);
}

//  the number of occurrences of the (1, 0, 1) in the input difference
void ModelProb2(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& c, GRBVar& d)
{
	model.addConstr(b + d <= 1);
	model.addConstr(a - d >= 0);
	model.addConstr(c - d >= 0);
	model.addConstr(1 - a + b - c + d >= 0);
}


void sbox_n(GRBModel& model, int n, GRBVar* s, GRBVar* t, GRBVar* active, GRBVar* active2)
{
	int i;
	int m = n / 2;


	GRBVar* y = model.addVars(n, GRB_BINARY);

	for (i = 0; i < m - 3; i++)
	{
		model.addConstr(s[i + 1] + s[i + 2] - y[i] >= 0);
		ModelXOR3(model, s[i], s[i + 2], y[i], t[i]);
		ModelProb(model, s[i + 1], s[i + 2], active[i]);
	}

	model.addConstr(s[m - 2] + s[0] - y[m - 3] >= 0);
	ModelXOR3(model, s[m], s[0], y[m - 3], t[m - 3]);
	ModelProb(model, s[m - 2], s[0], active[m - 3]);

	model.addConstr(s[0] + s[1] - y[m - 2] >= 0);
	ModelXOR3(model, s[m - 1], s[1], y[m - 2], t[m - 2]);
	ModelProb(model, s[0], s[1], active[m - 2]);

	model.addConstr(s[m] + s[m + 1] - y[m - 1] >= 0);
	ModelXOR4(model, s[m - 3], s[m], s[m + 1], y[m - 1], t[m - 1]);
	ModelProb(model, s[m], s[m + 1], active[m - 1]);

	model.addConstr(s[m + 1] + s[m + 2] - y[m] >= 0);
	ModelXOR3(model, s[m - 2], s[m + 2], y[m], t[m]);
	ModelProb(model, s[m + 1], s[m + 2], active[m]);

	for (i = m + 1; i < n - 2; i++)
	{
		model.addConstr(s[i + 1] + s[i + 2] - y[i] >= 0);
		ModelXOR3(model, s[i], s[i + 2], y[i], t[i]);
		ModelProb(model, s[i + 1], s[i + 2], active[i]);
	}

	model.addConstr(s[n - 1] + s[m - 1] - y[n - 2] >= 0);
	ModelXOR3(model, s[n - 2], s[m - 1], y[n - 2], t[n - 2]);
	ModelProb(model, s[n - 1], s[m - 1], active[n - 2]);

	model.addConstr(s[m - 1] + s[m] - y[n - 1] >= 0);
	ModelXOR3(model, s[n - 1], s[m], y[n - 1], t[n - 1]);
	ModelProb(model, s[m - 1], s[m], active[n - 1]);





	for (i = 0; i < m - 4; i++)
	{
		model.addConstr(s[i + 1] - s[i + 2] + s[i + 3] - y[i] + y[i + 1] <= 2);
		model.addConstr(s[i + 1] - s[i + 2] + s[i + 3] + y[i] - y[i + 1] <= 2);
		ModelProb2(model, s[i + 1], s[i + 2], s[i + 3], active2[i]);
	}

	model.addConstr(s[m - 3] - s[m - 2] + s[0] - y[m - 4] + y[m - 3] <= 2);
	model.addConstr(s[m - 3] - s[m - 2] + s[0] + y[m - 4] - y[m - 3] <= 2);
	ModelProb2(model, s[m - 3], s[m - 2], s[0], active2[m - 4]);

	model.addConstr(s[m - 2] - s[0] + s[1] - y[m - 3] + y[m - 2] <= 2);
	model.addConstr(s[m - 2] - s[0] + s[1] + y[m - 3] - y[m - 2] <= 2);
	ModelProb2(model, s[m - 2], s[0], s[1], active2[m - 3]);

	model.addConstr(s[0] - s[1] + s[2] - y[m - 2] + y[0] <= 2);
	model.addConstr(s[0] - s[1] + s[2] + y[m - 2] - y[0] <= 2);
	ModelProb2(model, s[0], s[1], s[2], active2[m - 2]);

	model.addConstr(s[m] - s[m + 1] + s[m + 2] - y[m - 1] + y[m] <= 2);
	model.addConstr(s[m] - s[m + 1] + s[m + 2] + y[m - 1] - y[m] <= 2);
	ModelProb2(model, s[m], s[m + 1], s[m + 2], active2[m - 1]);

	model.addConstr(s[m + 1] - s[m + 2] + s[m + 3] - y[m] + y[m + 1] <= 2);
	model.addConstr(s[m + 1] - s[m + 2] + s[m + 3] + y[m] - y[m + 1] <= 2);
	ModelProb2(model, s[m + 1], s[m + 2], s[m + 3], active2[m]);

	for (i = m + 1; i < n - 3; i++)
	{
		model.addConstr(s[i + 1] - s[i + 2] + s[i + 3] - y[i] + y[i + 1] <= 2);
		model.addConstr(s[i + 1] - s[i + 2] + s[i + 3] + y[i] - y[i + 1] <= 2);
		ModelProb2(model, s[i + 1], s[i + 2], s[i + 3], active2[i]);
	}

	model.addConstr(s[n - 2] - s[n - 1] + s[m - 1] - y[n - 3] + y[n - 2] <= 2);
	model.addConstr(s[n - 2] - s[n - 1] + s[m - 1] + y[n - 3] - y[n - 2] <= 2);
	ModelProb2(model, s[n - 2], s[n - 1], s[m - 1], active2[n - 3]);

	model.addConstr(s[n - 1] - s[m - 1] + s[m] - y[n - 2] + y[n - 1] <= 2);
	model.addConstr(s[n - 1] - s[m - 1] + s[m] + y[n - 2] - y[n - 1] <= 2);
	ModelProb2(model, s[n - 1], s[m - 1], s[m], active2[n - 2]);

	model.addConstr(s[m - 1] - s[m] + s[m + 1] - y[n - 1] + y[m - 1] <= 2);
	model.addConstr(s[m - 1] - s[m] + s[m + 1] + y[n - 1] - y[m - 1] <= 2);
	ModelProb2(model, s[m - 1], s[m], s[m + 1], active2[n - 1]);

}



// Linear layer parameters
struct LinearParams {
	int alpha;
	int beta0;
	int beta1;
	int beta2;
};

LinearParams L32_params = { 11, 5, 9, 12 };
LinearParams L32_prime_params = { 11, 1, 26, 30 };
LinearParams L40_params = { 17, 1, 9, 30 };
LinearParams L64_params = { 3, 1, 26, 50 };
LinearParams L128_params = { 17, 7, 11, 14 };



void LT(GRBModel& model, int n, GRBVar* t, GRBVar* s, const LinearParams& params)
{
	for (int i = 0; i < n; i++)
	{
		int idx0 = (params.alpha * i + params.beta0) % n;
		int idx1 = (params.alpha * i + params.beta1) % n;
		int idx2 = (params.alpha * i + params.beta2) % n;

		ModelXOR3(model, t[idx0], t[idx1], t[idx2], s[i]);
	}
}



void search_X(int round, int N)
{
	int i, j;

	GRBEnv env = GRBEnv();
	env.set(GRB_IntParam_LogToConsole, 0);

	GRBModel model = GRBModel(env);

	GRBVar** s = new GRBVar * [round + 1];
	GRBVar** t = new GRBVar * [round];

	for (i = 0; i < round; i++)
	{
		s[i] = model.addVars(N, GRB_BINARY);
		t[i] = model.addVars(N, GRB_BINARY);
	}
	s[round] = model.addVars(N, GRB_BINARY);

	GRBVar** active = new GRBVar * [round];
	for (i = 0; i < round; i++)
		active[i] = model.addVars(N, GRB_BINARY);

	GRBVar** active2 = new GRBVar * [round];
	for (i = 0; i < round; i++)
		active2[i] = model.addVars(N, GRB_BINARY);

	for (int loc = 0; loc < round; loc++)
	{
		sbox_n(model, N, s[loc], t[loc], active[loc], active2[loc]);
		if (N == 32)
		{
			LT(model, N, t[loc], s[loc + 1], L32_params);
		}
		else if (N == 40)
		{
			LT(model, N, t[loc], s[loc + 1], L40_params);
		}
		else if (N == 64)
		{
			LT(model, N, t[loc], s[loc + 1], L64_params);
		}


	}


	GRBLinExpr ttt = 0;
	for (i = 0; i < round; i++)
		for (j = 0; j < N; j++)
			ttt += (active[i][j] - active2[i][j]);

	model.addConstr(ttt >= 1);


	model.setObjective(ttt, GRB_MINIMIZE);
	model.update();
	model.optimize();





	int ret = int(model.get(GRB_DoubleAttr_ObjVal));
	cout << "P: " << ret << endl << endl;


	for (i = 0; i < round; i++)
	{
		for (j = 0; j < N; j++)
		{
			if (s[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "S" << endl;


		for (j = 0; j < N; j++)
		{
			if (t[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "L" << endl;

	}
	for (j = 0; j < N; j++)
	{
		if (s[round][j].get(GRB_DoubleAttr_Xn) > 0.6)
			cout << 1;
		else
			cout << 0;
	}
	cout << endl << endl;

}


void search_XT(int round, int N)
{
	int i, j;

	GRBEnv env = GRBEnv();
	env.set(GRB_IntParam_LogToConsole, 0);

	GRBModel model = GRBModel(env);

	GRBVar** T_s = new GRBVar * [round + 1];
	GRBVar** T_t = new GRBVar * [round];

	for (i = 0; i < round; i++)
	{
		T_s[i] = model.addVars(64, GRB_BINARY);
		T_t[i] = model.addVars(64, GRB_BINARY);
	}
	T_s[round] = model.addVars(64, GRB_BINARY);

	GRBVar** T_active = new GRBVar * [round];
	for (i = 0; i < round; i++)
		T_active[i] = model.addVars(64, GRB_BINARY);

	GRBVar** T_active2 = new GRBVar * [round];
	for (i = 0; i < round; i++)
		T_active2[i] = model.addVars(64, GRB_BINARY);



	GRBVar** D_s = new GRBVar * [round + 1];
	GRBVar** D_t = new GRBVar * [round];
	GRBVar** D_r = new GRBVar * [round];


	for (i = 0; i < round; i++)
	{
		D_s[i] = model.addVars(N, GRB_BINARY);
		D_t[i] = model.addVars(N, GRB_BINARY);
		D_r[i] = model.addVars(N, GRB_BINARY);

	}
	D_s[round] = model.addVars(N, GRB_BINARY);

	GRBVar** D_active = new GRBVar * [round];
	for (i = 0; i < round; i++)
		D_active[i] = model.addVars(N, GRB_BINARY);

	GRBVar** D_active2 = new GRBVar * [round];
	for (i = 0; i < round; i++)
		D_active2[i] = model.addVars(N, GRB_BINARY);


	for (int loc = 0; loc < round; loc++)
	{
		sbox_n(model, N, D_s[loc], D_t[loc], D_active[loc], D_active2[loc]);
		sbox_n(model, 64, T_s[loc], T_t[loc], T_active[loc], T_active2[loc]);


		if (loc < round)
		{
			if (N == 32)
				LT(model, N, D_t[loc], D_r[loc], L32_params);
			else if (N == 40)
				LT(model, N, D_t[loc], D_r[loc], L40_params);

			LT(model, 64, T_t[loc], T_s[loc + 1], L64_params);

			for (i = 0; i < N; i++)
				ModelXOR2(model, T_s[loc + 1][i], D_r[loc][i], D_s[loc + 1][i]);
		}

	}



	GRBLinExpr ttt = 0;
	for (i = 0; i < round; i++)
		for (j = 0; j < N; j++)
			ttt += (D_active[i][j] - D_active2[i][j]);

	for (i = 0; i < round; i++)
		for (j = 0; j < 64; j++)
			ttt += (T_active[i][j] - T_active2[i][j]);

	GRBLinExpr sss = 0;
	for (j = 0; j < 64; j++)
		sss += T_s[0][j];
	model.addConstr(sss >= 1);



	model.setObjective(ttt, GRB_MINIMIZE);
	model.update();
	model.optimize();



	int ret = int(model.get(GRB_DoubleAttr_ObjVal));
	cout << "P: " << ret << endl << endl;


	for (i = 0; i < round; i++)
	{
		for (j = 0; j < 64; j++)
		{
			if (T_s[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "S" << endl;


		for (j = 0; j < 64; j++)
		{
			if (T_t[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "L" << endl;

	}
	for (j = 0; j < 64; j++)
	{
		if (T_s[round][j].get(GRB_DoubleAttr_Xn) > 0.6)
			cout << 1;
		else
			cout << 0;
	}
	cout << endl << endl;


	for (i = 0; i < round; i++)
	{
		for (j = 0; j < N; j++)
		{
			if (D_s[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "S" << endl;

		for (j = 0; j < N; j++)
		{
			if (D_t[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "L" << endl;

		for (j = 0; j < N; j++)
		{
			if (D_r[i][j].get(GRB_DoubleAttr_Xn) > 0.6)
				cout << 1;
			else
				cout << 0;
		}
		cout << endl << "Xor" << endl;
	}

	for (j = 0; j < N; j++)
	{
		if (D_s[round][j].get(GRB_DoubleAttr_Xn) > 0.6)
			cout << 1;
		else
			cout << 0;
	}
	cout << endl << endl;

}


int main()
{

	cout << "Search for 1-round differential trail..." << endl;
	search_X(1, 32);   //D32
	search_X(1, 40);   //D40
	search_X(1, 64);   //T

	cout << "Search for 2-round differential trail..." << endl;
	search_X(2, 32);   //D32
	search_X(2, 40);   //D40
	search_X(2, 64);   //T

	cout << "Search for 3-round differential trail..." << endl;
	search_X(3, 32);   //D32
	search_X(3, 40);   //D40
	search_X(3, 64);   //T



	cout << "Search for 1-round differential trail..." << endl;
	search_XT(1, 32);   //D32
	search_XT(1, 40);   //D40

	cout << "Search for 2-round differential trail..." << endl;
	search_XT(2, 32);   //D32
	search_XT(2, 40);   //D40

	cout << "Search for 3-round differential trail..." << endl;
	search_XT(3, 32);   //D32
	search_XT(3, 40);   //D40

	return 0;
}



