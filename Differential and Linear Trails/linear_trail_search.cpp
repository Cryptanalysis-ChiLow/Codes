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


const int N = 32;

const int VARIABLE_COUNT = N + 1;
const int loc_1 = N;
using PolyTerm = bitset<VARIABLE_COUNT>;
using Poly = vector<PolyTerm>;


void printPoly(Poly poly)
{
    int i;
    for (auto& term : poly)
    {
        for (i = 0; i < N; i++)
            if (term[i] == 1)   cout << "v" << i;
        if (term[loc_1] == 1)   cout << "1";
        cout << "+";
    }
    cout << endl;
}


bool MySort(PolyTerm a, PolyTerm b)
{
    for (int i = 0; i < VARIABLE_COUNT; i++)
    {
        if (a[i] > b[i])
            return true;
        else if (a[i] < b[i])
            return false;
    }
    return false;
}
Poly add(Poly a, Poly b)
{
    Poly result;
    unordered_map<PolyTerm, int> termCount;

    for (auto& term : a)
        termCount[term]++;
    for (auto& term : b)
        termCount[term]++;

    for (const auto& pair : termCount)
        if (pair.second % 2 == 1)
            result.push_back(pair.first);

    sort(result.begin(), result.end(), MySort);
    return result;
}
Poly addmore(vector<Poly> polys)
{
    Poly result;
    unordered_map<PolyTerm, int> termCount;

    for (auto& p : polys)
        for (auto& term : p)
            termCount[term]++;

    for (const auto& pair : termCount)
        if (pair.second % 2 == 1)
            result.push_back(pair.first);

    sort(result.begin(), result.end(), MySort);
    return result;
}
Poly add_one(Poly a)
{
    sort(a.begin(), a.end(), MySort);

    PolyTerm term;
    term.set(loc_1);
    if (a.size() == 0)
    {
        Poly one;
        one.push_back(term);
        return one;
    }
    else
    {
        if (a[a.size() - 1][loc_1] == 1)
            a.pop_back();
        else
            a.push_back(term);
        return a;
    }
}
Poly multiply(Poly a, Poly b)
{
    Poly result;
    unordered_map<PolyTerm, int> occurrences;
    for (auto& termA : a)
    {
        for (auto& termB : b)
        {
            PolyTerm newTerm = termA | termB;
            if (newTerm[loc_1] == 1 && newTerm.count() > 1)
                newTerm[loc_1] = 0;
            occurrences[newTerm]++;
        }
    }
    for (auto& pair : occurrences)
        if (pair.second % 2 == 1)
            result.push_back(pair.first);
    sort(result.begin(), result.end(), MySort);
    return result;
}

vector<Poly> ChiChi(vector<Poly>& x)
{
    const int N = x.size();
    const int m = N / 2;

    vector<Poly> y(N);

    for (int i = 0; i < m - 3; i++)
        y[i] = add(x[i], multiply(add_one(x[i + 1]), x[i + 2]));
    for (int i = m + 1; i < N - 2; i++)
        y[i] = add(x[i], multiply(add_one(x[i + 1]), x[i + 2]));

    y[m - 3] = add(x[m], multiply(add_one(x[m - 2]), x[0]));
    y[m - 2] = add(x[m - 1], multiply(add_one(x[0]), x[1]));
    y[m - 1] = add(add_one(x[m - 3]), multiply(add_one(x[m]), add_one(x[m + 1])));
    y[m] = add(x[m - 2], multiply(add_one(x[m + 1]), x[m + 2]));
    y[N - 2] = add(x[N - 2], multiply(add_one(x[N - 1]), x[m - 1]));
    y[N - 1] = add(x[N - 1], multiply(add_one(x[m - 1]), x[m]));
    return y;
}

vector<Poly> calculateSeparate(Poly poly)
{
    vector<Poly> Separate;
    int i;
    Poly poly1;
    PolyTerm variable;

    int size = poly.size();
    int* flag = new int[size];
    for (i = 0; i < size; i++)
        flag[i] = 0;

    for (i = 0; i < size + 1; i++)
    {
        if (i == size)
        {
            if (poly1.size() == 0)
                break;
            Separate.push_back(poly1);
            poly1.clear();
            i = 0;
        }
        if (flag[i] == 0)
        {
            if (poly1.size() == 0)
            {
                poly1.push_back(poly[i]);
                variable = poly[i];
                flag[i] = 1;
            }
            else
            {
                if ((variable & poly[i]).any())
                {
                    poly1.push_back(poly[i]);
                    variable = variable | poly[i];
                    flag[i] = 1;
                    i = 0;
                }
            }
        }
    }
    delete[] flag;

    return Separate;
}

int calculate_weight(Poly poly)
{
    int i, j;

    PolyTerm variable;
    for (auto& term : poly)
        variable = variable | term;
    variable[loc_1] = 0;

    vector<int> variables;
    for (i = 0; i < VARIABLE_COUNT; i++)
        if (variable[i] == 1)
            variables.push_back(i);
    int size = variables.size();

    PolyTerm term;
    int flag, value;
    int count = 0;
    for (i = 0; i < pow(2, size); i++)
    {
        term.reset();
        for (j = 0; j < size; j++)
            term[variables[j]] = (i >> j) & 1;

        value = 0;
        for (auto& tttt : poly)
        {
            if (tttt[loc_1] == 1)
                value = value ^ 1;
            else
            {
                flag = 0;
                for (j = 0; j < size; j++)
                {
                    if (tttt[variables[j]] == 1)
                    {
                        if (term[variables[j]] == 1)
                            flag = 1;
                        else
                        {
                            flag = 0;
                            break;
                        }
                    }
                }
                if (flag == 1)
                    value = value ^ 1;
            }
        }
        count = count + value;
    }
    return count;
}

double Probability_1(Poly poly)
{
    PolyTerm variable;
    for (auto& term : poly)
        variable = variable | term;
    variable[loc_1] = 0;
    int count = variable.count();
    return (1.0 * calculate_weight(poly) / pow(2, count));
}

double calculatebias(Poly poly)
{
    int count;
    double bias;
    PolyTerm variable;
    vector<Poly> Separate = calculateSeparate(poly);

    double epsilon = 0.5;
    for (int i = 0; i < Separate.size(); i++)
    {
        variable.reset();
        for (auto& term : Separate[i])
            variable = variable | term;
        variable[loc_1] = 0;

        count = variable.count();

        bias = 0.5 - Probability_1(Separate[i]);
        epsilon = 2 * epsilon * bias;

        if (epsilon == 0)
            return 0;
    }
    return epsilon;
}


double isSolutionValid(bitset<2 * N> solution)
{
    int n = N;
    vector<Poly> T_state(n);

    PolyTerm term;
    for (int i = 0; i < n; i++)
    {
        term.reset();
        term.set(i);
        T_state[i].push_back(term);
    }

    vector<Poly> T;

    for (int i = 0; i < n; i++)
    {
        if (solution[i] == 1)
            T.push_back(T_state[i]);
    }

    T_state = ChiChi(T_state);

    for (int i = 0; i < n; i++)
    {
        if (solution[n + i] == 1)
            T.push_back(T_state[i]);
    }

    Poly poly = addmore(T);
    double bias = calculatebias(poly);

    return bias;
}




void ModelCopy3(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& c, GRBVar& o)
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
void ModelCopy4(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& c, GRBVar& d, GRBVar& o)
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
void ModelCopy5(GRBModel& model, GRBVar& a, GRBVar& b, GRBVar& c, GRBVar& d, GRBVar& e, GRBVar& o)
{
    model.addConstr(a + b + c + d + e + (1 - o) >= 1);
    model.addConstr(a + b + c + d + (1 - e) + o >= 1);
    model.addConstr(a + b + c + (1 - d) + e + o >= 1);
    model.addConstr(a + b + c + (1 - d) + (1 - e) + (1 - o) >= 1);
    model.addConstr(a + b + (1 - c) + d + e + o >= 1);
    model.addConstr(a + b + (1 - c) + d + (1 - e) + (1 - o) >= 1);
    model.addConstr(a + b + (1 - c) + (1 - d) + e + (1 - o) >= 1);
    model.addConstr(a + b + (1 - c) + (1 - d) + (1 - e) + o >= 1);
    model.addConstr(a + (1 - b) + c + d + e + o >= 1);
    model.addConstr(a + (1 - b) + c + d + (1 - e) + (1 - o) >= 1);
    model.addConstr(a + (1 - b) + c + (1 - d) + e + (1 - o) >= 1);
    model.addConstr(a + (1 - b) + c + (1 - d) + (1 - e) + o >= 1);
    model.addConstr(a + (1 - b) + (1 - c) + d + e + (1 - o) >= 1);
    model.addConstr(a + (1 - b) + (1 - c) + d + (1 - e) + o >= 1);
    model.addConstr(a + (1 - b) + (1 - c) + (1 - d) + e + o >= 1);
    model.addConstr(a + (1 - b) + (1 - c) + (1 - d) + (1 - e) + (1 - o) >= 1);
    model.addConstr((1 - a) + b + c + d + e + o >= 1);
    model.addConstr((1 - a) + b + c + d + (1 - e) + (1 - o) >= 1);
    model.addConstr((1 - a) + b + c + (1 - d) + e + (1 - o) >= 1);
    model.addConstr((1 - a) + b + c + (1 - d) + (1 - e) + o >= 1);
    model.addConstr((1 - a) + b + (1 - c) + d + e + (1 - o) >= 1);
    model.addConstr((1 - a) + b + (1 - c) + d + (1 - e) + o >= 1);
    model.addConstr((1 - a) + b + (1 - c) + (1 - d) + e + o >= 1);
    model.addConstr((1 - a) + b + (1 - c) + (1 - d) + (1 - e) + (1 - o) >= 1);
    model.addConstr((1 - a) + (1 - b) + c + d + e + (1 - o) >= 1);
    model.addConstr((1 - a) + (1 - b) + c + d + (1 - e) + o >= 1);
    model.addConstr((1 - a) + (1 - b) + c + (1 - d) + e + o >= 1);
    model.addConstr((1 - a) + (1 - b) + c + (1 - d) + (1 - e) + (1 - o) >= 1);
    model.addConstr((1 - a) + (1 - b) + (1 - c) + d + e + o >= 1);
    model.addConstr((1 - a) + (1 - b) + (1 - c) + d + (1 - e) + (1 - o) >= 1);
    model.addConstr((1 - a) + (1 - b) + (1 - c) + (1 - d) + e + (1 - o) >= 1);
    model.addConstr((1 - a) + (1 - b) + (1 - c) + (1 - d) + (1 - e) + o >= 1);
}



void sbox_Conservative(GRBModel& model, GRBVar* s, GRBVar* t)
{
    int i;
    int n = N;
    int m = n / 2;

    GRBVar** st = new GRBVar * [n];
    for (i = 0; i < n; i++)
    {
        if (i != m)
        {
            st[i] = model.addVars(4, GRB_BINARY);
            ModelCopy4(model, s[i], st[i][0], st[i][1], st[i][2], st[i][3]);
        }
        else
        {
            st[i] = model.addVars(5, GRB_BINARY);
            ModelCopy5(model, s[i], st[i][0], st[i][1], st[i][2], st[i][3], st[i][4]);
        }
    }

    for (i = 0; i < m - 1; i++)
    {
        model.addConstr(t[i] >= st[(i + 1) % (m - 1)][0]);
        model.addConstr(t[i] >= st[(i + 2) % (m - 1)][1]);
    }

    for (int i = (m - 1); i < n; i++)
    {
        model.addConstr(t[i] >= st[(i + 1) % n + (m - 1) * ((i + 1) / n)][0]);
        model.addConstr(t[i] >= st[(i + 2) % n + (m - 1) * ((i + 2) / n)][1]);
    }

    for (i = 0; i < m - 3; i++)
    {
        model.addConstr(t[i] == st[i][2]);
        model.addConstr(t[i] == st[i + 2][3]);
    }

    model.addConstr(t[m - 3] == st[m][2]);
    model.addConstr(t[m - 3] == st[0][3]);

    model.addConstr(t[m - 2] == st[m - 1][2]);
    model.addConstr(t[m - 2] == st[1][3]);

    model.addConstr(t[m - 1] == st[m - 3][2]);
    model.addConstr(t[m - 1] == st[m + 1][3]);
    model.addConstr(t[m - 1] == st[m][4]);

    model.addConstr(t[m] == st[m - 2][2]);
    model.addConstr(t[m] == st[m + 2][3]);

    for (i = m + 1; i < n - 2; i++)
    {
        model.addConstr(t[i] == st[i][2]);
        model.addConstr(t[i] == st[i + 2][3]);
    }

    model.addConstr(t[n - 2] == st[n - 2][2]);
    model.addConstr(t[n - 2] == st[m - 1][3]);

    model.addConstr(t[n - 1] == st[n - 1][2]);
    model.addConstr(t[n - 1] == st[m][3]);


}


void Modelremove3(GRBModel& model, GRBVar& a, GRBVar& c, GRBVar& ab, GRBVar& bc)
{
    model.addConstr(a + ab + bc - c <= 2);
    model.addConstr(c + ab + bc - a <= 2);
}

void sbox_Aggressive(GRBModel& model, GRBVar* s, GRBVar* t)
{
    int i;
    int n = N;
    int m = n / 2;


    GRBVar** st = new GRBVar * [n];
    for (i = 0; i < n; i++)
    {
        if (i != m)
        {
            st[i] = model.addVars(3, GRB_BINARY);
            ModelCopy3(model, s[i], st[i][0], st[i][1], st[i][2]);
        }
        else
        {
            st[i] = model.addVars(4, GRB_BINARY);
            ModelCopy4(model, s[i], st[i][0], st[i][1], st[i][2], st[i][3]);
        }
    }


    model.addConstr(t[m - 3] + t[m - 2] >= st[0][0]);
    Modelremove3(model, st[1][0], st[m - 2][0], t[m - 3], t[m - 2]);

    model.addConstr(t[m - 2] + t[0] >= st[1][0]);
    Modelremove3(model, st[0][0], st[2][0], t[m - 2], t[0]);

    for (i = 0; i < m - 4; i++)
    {
        model.addConstr(t[i] + t[i + 1] >= st[i + 2][0]);
        Modelremove3(model, st[i + 1][0], st[i + 3][0], t[i], t[i + 1]);
    }

    model.addConstr(t[m - 3] + t[m - 1] >= st[m - 2][0]);
    Modelremove3(model, st[m - 3][0], st[0][0], t[m - 2], t[m - 3]);

    model.addConstr(t[n - 2] + t[n - 1] >= st[m - 1][0]);
    Modelremove3(model, st[n - 1][0], st[m][0], t[n - 2], t[n - 1]);

    model.addConstr(t[n - 1] + t[m - 1] >= st[m][0]);
    Modelremove3(model, st[m - 1][0], st[m + 1][0], t[n - 1], t[m - 1]);

    for (i = m - 1; i < n - 3; i++)
    {
        model.addConstr(t[i] + t[i + 1] >= st[i + 2][0]);
        Modelremove3(model, st[i + 1][0], st[i + 3][0], t[i], t[i + 1]);
    }

    model.addConstr(t[n - 3] + t[n - 2] >= st[n - 1][0]);
    Modelremove3(model, st[n - 2][0], st[m - 1][0], t[n - 3], t[n - 2]);



    for (i = 0; i < m - 3; i++)
    {
        model.addConstr(t[i] == st[i][1]);
        model.addConstr(t[i] == st[i + 2][2]);
    }

    model.addConstr(t[m - 3] == st[m][1]);
    model.addConstr(t[m - 3] == st[0][2]);

    model.addConstr(t[m - 2] == st[m - 1][1]);
    model.addConstr(t[m - 2] == st[1][2]);

    model.addConstr(t[m - 1] == st[m - 3][1]);
    model.addConstr(t[m - 1] == st[m + 1][2]);
    model.addConstr(t[m - 1] == st[m][3]);

    model.addConstr(t[m] == st[m - 2][1]);
    model.addConstr(t[m] == st[m + 2][2]);

    for (i = m + 1; i < n - 2; i++)
    {
        model.addConstr(t[i] == st[i][1]);
        model.addConstr(t[i] == st[i + 2][2]);
    }

    model.addConstr(t[n - 2] == st[n - 2][1]);
    model.addConstr(t[n - 2] == st[m - 1][2]);

    model.addConstr(t[n - 1] == st[n - 1][1]);
    model.addConstr(t[n - 1] == st[m][2]);
}


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

void LT(GRBModel& model, GRBVar* t, GRBVar* s, const LinearParams& params)
{
    GRBVar** tt = new GRBVar * [N];
    for (int i = 0; i < N; i++)
    {
        tt[i] = model.addVars(3, GRB_BINARY);
        ModelCopy3(model, t[i], tt[i][0], tt[i][1], tt[i][2]);
    }

    for (int i = 0; i < N; i++)
    {
        int idx0 = (params.alpha * i + params.beta0) % N;
        int idx1 = (params.alpha * i + params.beta1) % N;
        int idx2 = (params.alpha * i + params.beta2) % N;

        model.addConstr(s[i] == tt[idx0][0]);
        model.addConstr(s[i] == tt[idx1][1]);
        model.addConstr(s[i] == tt[idx2][2]);
    }
}




void ModelProb(GRBModel& model, GRBVar& A, GRBVar& B, GRBVar& C, GRBVar& D, GRBVar& E)
{
    model.addConstr(1 - D - E >= 0);
    model.addConstr(B - E >= 0);
    model.addConstr(B - D >= 0);
    model.addConstr(C - E >= 0);
    model.addConstr(A - D >= 0);
}

//the number of occurrences of (1,1) in the output mask, and "1" is not recorded repeatedly
void set_odd(GRBModel& model, GRBVar* y, GRBVar* pr)
{
    int m = N / 2;
    for (int i = 0; i < m - 1; i++)
    {
        ModelProb(model, y[i], y[(i + 1) % (m - 1)], y[(i + 2) % (m - 1)], pr[i], pr[(i + 1) % (m - 1)]);
    }


    for (int i = (m - 1); i < N; i++)
    {
        ModelProb(model, y[i], y[(i + 1) % N + (m - 1) * ((i + 1) / N)], y[(i + 2) % N + (m - 1) * ((i + 2) / N)], pr[i], pr[(i + 1) % N + (m - 1) * ((i + 1) / N)]);
    }
}



int Rough_boundary(int round, vector<int> lamda)
{

    int i, j;

    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_LogToConsole, 0);

    GRBModel model = GRBModel(env);
    GRBVar** s = new GRBVar * [round];
    GRBVar** t = new GRBVar * [round];

    for (i = 0; i < round; i++)
    {
        s[i] = model.addVars(N, GRB_BINARY);
        t[i] = model.addVars(N, GRB_BINARY);
    }

    if (lamda.size() == 0)
    {
        GRBLinExpr sss = 0;
        for (j = 0; j < N; j++)
            sss += s[0][j];
        model.addConstr(sss >= 1);
    }
    else
    {
        bitset<N> IV(0);
        for (int t : lamda)
            IV.set(t);

        for (j = 0; j < N; j++)
            if (IV[j])
                model.addConstr(s[0][j] == 1);
            else
                model.addConstr(s[0][j] == 0);
    }





    GRBVar** pr = new GRBVar * [round];
    for (i = 0; i < round; i++)
        pr[i] = model.addVars(N, GRB_BINARY);


    for (int loc = 0; loc < round; loc++)
    {
        sbox_Aggressive(model, s[loc], t[loc]);
        if (loc < round - 1)
        {
            if (N == 32)
            {
                LT(model, t[loc], s[loc + 1], L32_params);
            }
            else if (N == 40)
            {
                LT(model, t[loc], s[loc + 1], L40_params);
            }
        }

        set_odd(model, t[loc], pr[loc]);
    }



    GRBLinExpr ttt = 0;
    for (i = 0; i < round; i++)
        for (j = 0; j < N; j++)
            ttt += t[i][j] - pr[i][j];

    model.setObjective(ttt, GRB_MINIMIZE);

    // int iteration = 0;

     //double totalGurobiTime = 0.0;
    while (true)
    {
        // iteration++;

        model.update();
        model.optimize();
        //double thisTime = model.get(GRB_DoubleAttr_Runtime);
       // totalGurobiTime += thisTime;

        bitset<2 * N> solution1(0), solution2(0), solution3(0);
        for (i = 0; i < N; i++)
        {
            if (s[0][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution1.set(i);
            if (t[0][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution1.set(N + i);
            if (s[1][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution2.set(i);
            if (t[1][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution2.set(N + i);
            if (s[2][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution3.set(i);
            if (t[2][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution3.set(N + i);
        }

        int flag = 1;
        double bias1 = isSolutionValid(solution1);
        double bias2 = isSolutionValid(solution2);
        double bias3 = isSolutionValid(solution3);
        if (bias1 == 0)
        {
            flag = 0;
            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution1[i])
                    cut += (1 - s[0][i]);
                else
                    cut += s[0][i];
                if (solution1[i + N])
                    cut += (1 - t[0][i]);
                else
                    cut += t[0][i];
            }
            model.addConstr(cut >= 1);
        }

        if (bias2 == 0)
        {
            flag = 0;
            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution2[i])
                    cut += (1 - s[1][i]);
                else
                    cut += s[1][i];
                if (solution2[i + N])
                    cut += (1 - t[1][i]);
                else
                    cut += t[1][i];
            }
            model.addConstr(cut >= 1);
        }

        if (bias3 == 0)
        {
            flag = 0;
            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution3[i])
                    cut += (1 - s[2][i]);
                else
                    cut += s[2][i];
                if (solution3[i + N])
                    cut += (1 - t[2][i]);
                else
                    cut += t[2][i];
            }
            model.addConstr(cut >= 1);
        }
        if (flag == 1)
        {
            return -2 * log2(fabs(8 * bias1 * bias2 * bias3));


            int ret = int(model.get(GRB_DoubleAttr_ObjVal));
            if (ret == -2 * log2(fabs(8 * bias1 * bias2 * bias3)))
            {
                return ret;
            }
            else
            {
                cout << "Error!   Error!  Error!  Error!  " << endl;
            }
            cout << "First" << endl;
            cout << "ObjVal: " << ret << endl;
            cout << "Cor: " << fabs(8 * bias1 * bias2 * bias3) << "   log2(cor):" << log2(fabs(8 * bias1 * bias2 * bias3)) << endl;
            //cout << "Time: " << totalGurobiTime << "s,  Iteration:" << iteration << endl;
        }
    }
}



int search(int round, vector<int> lamda)
{

    //int boundary = Rough_boundary(round, lamda);



    int i, j;

    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_LogToConsole, 0);

    GRBModel model = GRBModel(env);
    GRBVar** s = new GRBVar * [round];
    GRBVar** t = new GRBVar * [round];

    for (i = 0; i < round; i++)
    {
        s[i] = model.addVars(N, GRB_BINARY);
        t[i] = model.addVars(N, GRB_BINARY);
    }

    if (lamda.size() == 0)
    {
        GRBLinExpr sss = 0;
        for (j = 0; j < N; j++)
            sss += s[0][j];
        model.addConstr(sss >= 1);
    }
    else
    {
        bitset<N> IV(0);
        for (int t : lamda)
            IV.set(t);

        for (j = 0; j < N; j++)
            if (IV[j])
                model.addConstr(s[0][j] == 1);
            else
                model.addConstr(s[0][j] == 0);
    }




    GRBVar** pr = new GRBVar * [round];
    for (i = 0; i < round; i++)
        pr[i] = model.addVars(N, GRB_BINARY);


    for (int loc = 0; loc < round; loc++)
    {
        sbox_Conservative(model, s[loc], t[loc]);
        if (loc < round - 1)
        {
            if (N == 32)
            {
                LT(model, t[loc], s[loc + 1], L32_params);
            }
            else if (N == 40)
            {
                LT(model, t[loc], s[loc + 1], L40_params);
            }
        }

        set_odd(model, t[loc], pr[loc]);
    }



    GRBLinExpr ttt = 0;
    for (i = 0; i < round; i++)
        for (j = 0; j < N; j++)
            ttt += t[i][j] - pr[i][j];

    //model.addConstr(ttt <= 13);
    model.setObjective(ttt, GRB_MINIMIZE);


    while (true)
    {

        model.update();
        model.optimize();

        //  int status = model.get(GRB_IntAttr_Status);
        //  if (status == GRB_INFEASIBLE)
        //  {
        //      cout << "Model is infeasible. No more solutions." << endl;
        //      return -1;
        //      break;
        //  }

        int objVal = int(model.get(GRB_DoubleAttr_ObjVal));
        //cout << "Objective value: " << objVal << endl;

        //if (2*objVal >= boundary)
       // {
            //cout << "The rough boundary is tight." << endl;
       //     cout << "ObjVal1: " << boundary << endl;
        //    return boundary;
       //     break;
       // }

        model.addConstr(ttt >= objVal);

        bitset<2 * N> solution1(0), solution2(0), solution3(0);
        for (i = 0; i < N; i++)
        {
            if (s[0][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution1.set(i);
            if (t[0][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution1.set(N + i);
            if (s[1][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution2.set(i);
            if (t[1][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution2.set(N + i);
            if (s[2][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution3.set(i);
            if (t[2][i].get(GRB_DoubleAttr_Xn) > 0.6)
                solution3.set(N + i);
        }

        int flag = 1;
        double bias1 = isSolutionValid(solution1);
        double bias2 = isSolutionValid(solution2);
        double bias3 = isSolutionValid(solution3);
        if (bias1 == 0)
        {
            flag = 0;
            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution1[i])
                    cut += (1 - s[0][i]);
                else
                    cut += s[0][i];
                if (solution1[i + N])
                    cut += (1 - t[0][i]);
                else
                    cut += t[0][i];
            }
            model.addConstr(cut >= 1);
        }

        if (bias2 == 0)
        {
            flag = 0;
            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution2[i])
                    cut += (1 - s[1][i]);
                else
                    cut += s[1][i];
                if (solution2[i + N])
                    cut += (1 - t[1][i]);
                else
                    cut += t[1][i];
            }
            model.addConstr(cut >= 1);
        }

        if (bias3 == 0)
        {
            flag = 0;
            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution3[i])
                    cut += (1 - s[2][i]);
                else
                    cut += s[2][i];
                if (solution3[i + N])
                    cut += (1 - t[2][i]);
                else
                    cut += t[2][i];
            }
            model.addConstr(cut >= 1);
        }
        if (flag == 1)
        {
            //cout << "ObjVal2: " << -2 * log2(fabs(8 * bias1 * bias2 * bias3)) << endl;
           // return -2 * log2(fabs(8 * bias1 * bias2 * bias3));

           // break;
            //cout << "Solution is valid!" << endl;
            int ret = int(model.get(GRB_DoubleAttr_ObjVal));


            //cout << "Second" << endl;
            cout << "ObjVal2: " << ret << endl;
            cout << "Bias: " << 4 * bias1 * bias2 * bias3 << "   log2(bias):" << log2(fabs(4 * bias1 * bias2 * bias3)) << endl;
            cout << "Cor: " << fabs(8 * bias1 * bias2 * bias3) << "   log2(cor):" << log2(fabs(8 * bias1 * bias2 * bias3)) << endl;


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
            cout << endl << endl;




            GRBLinExpr cut;
            for (i = 0; i < N; i++)
            {
                if (solution2[i])
                    cut += (1 - s[1][i]);
                else
                    cut += s[1][i];
                if (solution2[i + N])
                    cut += (1 - t[1][i]);
                else
                    cut += t[1][i];
            }

            for (i = 0; i < N; i++)
            {
                if (solution3[i])
                    cut += (1 - s[2][i]);
                else
                    cut += s[2][i];
                if (solution3[i + N])
                    cut += (1 - t[2][i]);
                else
                    cut += t[2][i];
            }

            model.addConstr(cut >= 1);


        }

        //cout << "Searching for feasible solutions..." << endl;
    }

    // cout << "Time: " << totalGurobiTime << "s,  Iteration:" << iteration << endl;

}


int main()
{
    vector<int> data = {};
    search(3, data);

    return 0;
}

