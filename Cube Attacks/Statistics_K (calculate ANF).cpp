#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <map>
#include <time.h>
#include <string>
#define round 3
using namespace std;



const int VARIABLE_COUNT = 128 + 64 + 40 + 1 + 1;       // key(128) + IV(64) + C(32) + x + const 1 
const int loc_x = 232;
const int loc_1 = 233;
using PolyTerm = bitset<VARIABLE_COUNT>;
using Poly = vector<PolyTerm>;


bool MySort(PolyTerm a, PolyTerm b)
{
    int i;
    for (i = 128; i < loc_x; i++)
    {
        if (a[i] > b[i])
            return true;
        else if (a[i] < b[i])
            return false;
    }

    for (i = 0; i < 128; i++)
    {
        if (a[i] > b[i])
            return true;
        else if (a[i] < b[i])
            return false;
    }
    if (a[loc_x] > b[loc_x])
        return true;
    else if (a[loc_x] < b[loc_x])
        return false;

    if (a[loc_1] > b[loc_1])
        return true;
    else if (a[loc_1] < b[loc_1])
        return false;

    return false;
}

void printPoly(Poly poly)
{

    int i;
    for (auto& term : poly)
    {
        for (i = 0; i < 128; i++)
            if (term[i] == 1)   cout << "k" << i;
        for (i = 0; i < 64; i++)
            if (term[i + 128] == 1) cout << "v" << i;
        for (i = 0; i < 40; i++)
            if (term[i + 192] == 1) cout << "c" << i;

        if (term[loc_x] == 1)   cout << "x";
        if (term[loc_1] == 1)   cout << "1";
        cout << "+";
    }
    cout << endl;
}

void Statistics_K(Poly poly, bitset<128>& Statistics_K)
{
    int i;
    PolyTerm mask;
    for (i = 128; i < VARIABLE_COUNT; i++)
        mask.set(i);
    
    for (auto& term : poly)
    {
        if ((term & mask).any())
        {
            for (i = 0; i < 128; i++)
                if (term[i] == 1)
                {
                    Statistics_K.set(i);
                }
        }
    }
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


// Linear layer parameters
struct LinearParams {
    int alpha;
    int beta0;
    int beta1;
    int beta2;
};

LinearParams L32_params = { 11, 5, 9, 12 };
LinearParams L40_params = { 17, 1, 9, 30 };
LinearParams L64_params = { 3, 1, 26, 50 };
LinearParams L128_params = { 17, 7, 11, 14 };

vector<Poly> linear_layer(vector<Poly>& x, const LinearParams& params)
{
    const int N = x.size();
    vector<Poly> y(N);
    vector<Poly> T;
    for (size_t i = 0; i < N; i++)
    {
        T.clear();
        T.push_back(x[(params.alpha * i + params.beta0) % N]);
        T.push_back(x[(params.alpha * i + params.beta1) % N]);
        T.push_back(x[(params.alpha * i + params.beta2) % N]);
        y[i] = addmore(T);
    }
    return y;
}


Poly calculatedifference(Poly poly)
{
    Poly difference;
    for (auto& term : poly)
    {
        if (term[loc_x] == 1)
        {
            term.reset(loc_x);
            if (term.count() == 0)
                term.set(loc_1);
            difference.push_back(term);
        }
    }
    return difference;
}


int mainD32()   //D32
{

    int ROUNDS = 1;
    vector<int> delta = { 0,13,20,22,23 };
    vector<Poly> K_state(128), T_state(64), C_state(32);


    PolyTerm term;

    //   VARIABLE_COUNT = 128 + 64 + 32 + 1 + 1;   key(128) + IV(64) + ciphertext(32)+ x + const 1

    for (int i = 0; i < 128; i++)
    {
        term.reset();
        term.set(i);
        K_state[i].push_back(term);
    };

    for (int i = 0; i < 64; i++)
    {
        term.reset();
        term.set(i + 128);
        T_state[i].push_back(term);
    };

    for (int i = 0; i < 32; i++)
    {
        term.reset();
        term.set(i + 128 + 64);
        C_state[i].push_back(term);
    };

    
    /*
    vector <int> set0 = { 3,6,8,10,28 };
    vector <int> set1 = { 1,26 };

    for (auto m : set0)
    {
        C_state[m].clear();
        term.reset();
        term.set(m+64);
        C_state[m].push_back(term);
    }


    for (auto m : set1)
    {
        C_state[m].clear();
        term.reset();
        term.set(m + 64);
        C_state[m].push_back(term);
        term.reset();
        term.set(loc_1);
        C_state[m].push_back(term);
    }
    */

    term.reset();
    term.set(loc_x);
    for (auto& t : delta)
    {
        C_state[t].push_back(term);
    };
 



    for (int j = 0; j < 32; j++)
        C_state[j] = add(C_state[j], K_state[64 + j]);

    for (int j = 0; j < 64; j++)
        T_state[j] = add(T_state[j], K_state[j]);


    for (int i = 0; i < ROUNDS; i++)
    {
        if (i == 0)
        {
            K_state[123] = add_one(K_state[123]);
        }
        if (i == 1)
        {
            K_state[127] = add_one(K_state[127]);
            K_state[122] = add_one(K_state[122]);
        }
        if (i == 2)
        {
            K_state[126] = add_one(K_state[126]);
            K_state[121] = add_one(K_state[121]);
        }


        C_state = ChiChi(C_state);
        T_state = ChiChi(T_state);
        K_state = ChiChi(K_state);



        C_state = linear_layer(C_state, L32_params);
        T_state = linear_layer(T_state, L64_params);
        K_state = linear_layer(K_state, L128_params);


        for (int j = 0; j < 32; j++)
            C_state[j] = add(C_state[j], T_state[j]);

        for (int j = 0; j < 64; j++)
            T_state[j] = add(T_state[j], K_state[j]);


        for (int j = 0; j < 32; j++)
        {
            printPoly(calculatedifference(C_state[j]));
        }
        cout << "-----------------------------------------------------------------------------------" << endl;
    }




    system("pause");
    return 0;
}




int mainD40()  //D40
{

    int ROUNDS = 2;
    vector<int> delta = { 9,10,19,31 };
    vector<Poly> K_state(128), T_state(64), C_state(40);


    PolyTerm term;

    //   VARIABLE_COUNT = 128 + 64 + 32 + 1 + 1;   key(128) + IV(64) + ciphertext(32)+ x + const 1

    for (int i = 0; i < 128; i++)
    {
        term.reset();
        term.set(i);
        K_state[i].push_back(term);
    };

    for (int i = 0; i < 64; i++)
    {
        term.reset();
        term.set(i + 128);
        T_state[i].push_back(term);
    };

    for (int i = 0; i < 40; i++)
    {
        term.reset();
        term.set(i + 128 + 64);
        C_state[i].push_back(term);
    };


    /*
    vector <int> set0 = { 3,6,8,10,28 };
    vector <int> set1 = { 1,26 };

    for (auto m : set0)
    {
        C_state[m].clear();
        term.reset();
        term.set(m + 64);
        C_state[m].push_back(term);
    }


    for (auto m : set1)
    {
        C_state[m].clear();
        term.reset();
        term.set(m + 64);
        C_state[m].push_back(term);
        term.reset();
        term.set(loc_1);
        C_state[m].push_back(term);
    }
    */

    term.reset();
    term.set(loc_x);
    for (auto& t : delta)
    {
        C_state[t].push_back(term);
    };




    for (int j = 0; j < 40; j++)
        C_state[j] = add(C_state[j], K_state[64 + j]);

    for (int j = 0; j < 64; j++)
        T_state[j] = add(T_state[j], K_state[j]);


    for (int i = 0; i < ROUNDS; i++)
    {
        if (i == 0)
        {
            K_state[96] = add_one(K_state[96]);
            K_state[123] = add_one(K_state[123]);
        }
        if (i == 1)
        {
            K_state[96] = add_one(K_state[96]);
            K_state[127] = add_one(K_state[127]);
            K_state[122] = add_one(K_state[122]);
        }
        if (i == 2)
        {
            K_state[126] = add_one(K_state[126]);
            K_state[121] = add_one(K_state[121]);
        }


        C_state = ChiChi(C_state);
        T_state = ChiChi(T_state);
        K_state = ChiChi(K_state);



        C_state = linear_layer(C_state, L40_params);
        T_state = linear_layer(T_state, L64_params);
        K_state = linear_layer(K_state, L128_params);


        for (int j = 0; j < 40; j++)
            C_state[j] = add(C_state[j], T_state[j]);

        for (int j = 0; j < 64; j++)
            T_state[j] = add(T_state[j], K_state[j]);


        for (int j = 0; j < 40; j++)
        {
            printPoly(calculatedifference(C_state[j]));
        }
        cout << "-----------------------------------------------------------------------------------" << endl;
    }




    system("pause");
    return 0;
}





int main()
{

    int ROUNDS = 1;
    vector<Poly> K_state(128), T_state(64), C_state(32);


    vector<int> cube_T = { 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62 };
    vector<int> cube_C = { 0 };
    // Statistics_K: 35
    // k0, k1, k3, k5, k7, k9, k11, k13, k15, k17, k19, k21, k23, k25, k27, k29, k31, k33, k35, k37, k39, k41, k43, k45, k47, k49, k51, k53, k55, k57, k59, k61, k63, k65, k78,

    // vector<int> cube_T = { 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61 };
    // vector<int> cube_C = { 1 };
    // Statistics_K: 35
    // k0, k2, k4, k6, k8, k10, k12, k14, k16, k18, k20, k22, k24, k26, k28, k30, k32, k34, k36, k38, k40, k42, k44, k46, k48, k50, k52, k54, k56, k58, k60, k62, k63, k64, k66,

    PolyTerm term;

    //   VARIABLE_COUNT = 128 + 64 + 32 + 1 + 1;   key(128) + IV(64) + ciphertext(32)+ x + const 1

    for (int i = 0; i < 128; i++)
    {
        term.reset();
        term.set(i);
        K_state[i].push_back(term);
    };

    for (auto& m : cube_T)
    {
        term.reset();
        term.set(m + 128);
        T_state[m].push_back(term);
    };

    for (auto& m : cube_C)
    {
        term.reset();
        term.set(m + 128 + 64);
        C_state[m].push_back(term);
    };

    /*
    cout << "---------------------  initialization  --------------------" << endl;
    cout << " K: " << endl;
    for (int j = 0; j < 128; j++)
        printPoly(K_state[j]);

    cout << " T: " << endl;
    for (int j = 0; j < 64; j++)
        printPoly(T_state[j]);

    cout << " C: " << endl;
    for (int j = 0; j < 32; j++)
        printPoly(C_state[j]);
    */





    for (int j = 0; j < 32; j++)
        C_state[j] = add(C_state[j], K_state[64 + j]);

    for (int j = 0; j < 64; j++)
        T_state[j] = add(T_state[j], K_state[j]);


    for (int i = 0; i < ROUNDS; i++)
    {
        if (i == 0)
        {
            K_state[123] = add_one(K_state[123]);
        }
        if (i == 1)
        {
            K_state[127] = add_one(K_state[127]);
            K_state[122] = add_one(K_state[122]);
        }
        if (i == 2)
        {
            K_state[126] = add_one(K_state[126]);
            K_state[121] = add_one(K_state[121]);
        }

        C_state = ChiChi(C_state);
        T_state = ChiChi(T_state);
        K_state = ChiChi(K_state);


        C_state = linear_layer(C_state, L32_params);
        T_state = linear_layer(T_state, L64_params);
        K_state = linear_layer(K_state, L128_params);


        for (int j = 0; j < 32; j++)
            C_state[j] = add(C_state[j], T_state[j]);

        for (int j = 0; j < 64; j++)
            T_state[j] = add(T_state[j], K_state[j]);



        cout << " C: " << endl;
        for (int j = 0; j < 32; j++)
            printPoly(C_state[j]);

        cout << " T: " << endl;
        for (int j = 0; j < 64; j++)
            printPoly(T_state[j]);




        bitset<128> K_count(0);
        for (int j = 0; j < 32; j++)
            Statistics_K(C_state[j], K_count);
        for (int j = 0; j < 64; j++)
            Statistics_K(T_state[j], K_count);



        cout << endl << "Statistics_K: " << K_count.count() << endl;
        for (int j = 0; j < 128; j++)
            if (K_count[j])  cout << 'k' << j << ",";
        cout << endl;

    }




    system("pause");
    return 0;
}


