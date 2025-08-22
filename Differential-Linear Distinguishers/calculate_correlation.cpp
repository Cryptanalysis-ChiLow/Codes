#include <iostream>
#include <bitset>
#include <vector>
#include <random>
#include <cassert>
#include <ctime>
#include <omp.h> 
#include <fstream>
using namespace std;
thread_local std::mt19937_64 rng(std::random_device{}());



const int KEY_SIZE = 128;
const int TWEAK_SIZE = 64;
const int BLOCK32_SIZE = 32;
const int BLOCK40_SIZE = 40;



// Helper function to extract bits from large bitset
template<size_t N>
uint64_t extract_bits(const bitset<N>& bs, size_t start, size_t len)
{
    if (start + len > N)
        throw out_of_range("Bit range out of bounds");
    uint64_t value = 0;
    for (size_t i = 0; i < len; i++)
        value |= static_cast<uint64_t>(bs[start + i]) << i;
    return value;
}

// Helper to create bitset from uint64_t
template<size_t N>
bitset<N> create_bitset(uint64_t value, size_t bits = N)
{
    bitset<N> bs;
    for (size_t i = 0; i < min<size_t>(bits, 64); i++)
        bs[i] = (value >> i) & 1;
    return bs;
}

template<size_t N>
void print_hex(bitset<N> K)
{
    string bin_str = K.to_string();
    const char hex_chars[] = "0123456789ABCDEF";
    string hex_str;
    hex_str.reserve(N / 4); // 128 bits -> 32 hex digits

    for (size_t i = 0; i < bin_str.size(); i += 4)
    {
        string four_bits = bin_str.substr(i, 4);
        unsigned long val = bitset<4>(four_bits).to_ulong();
        hex_str += hex_chars[val];
    }
    string result = "0x" + hex_str;
    cout << result << endl;
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

// Non-linear χ function for arbitrary size
template<size_t N>
bitset<N> chi(const bitset<N>& x) {
    bitset<N> y;
    for (size_t i = 0; i < N; i++) {
        bool b = x[(i + 1) % N];
        bool c = x[(i + 2) % N];
        y[i] = x[i] ^ (!b & c);
    }
    return y;
}

// ChiChi (𝕏) function for even dimensions (Definition 2)
template<size_t N>
bitset<N> ChiChi(const bitset<N>& x)
{
    const size_t m = N / 2;

    bitset<N> y;
    for (size_t i = 0; i < m - 3; i++)
        y[i] = x[i] ^ (~x[i + 1] & x[i + 2]);
    for (size_t i = m + 1; i < N - 2; i++)
        y[i] = x[i] ^ (~x[i + 1] & x[i + 2]);

    y[m - 3] = x[m] ^ (~x[m - 2] & x[0]);
    y[m - 2] = x[m - 1] ^ (~x[0] & x[1]);
    y[m - 1] = ~x[m - 3] ^ (~x[m] & ~x[m + 1]);
    y[m] = x[m - 2] ^ (~x[m + 1] & x[m + 2]);
    y[N - 2] = x[N - 2] ^ (~x[N - 1] & x[m - 1]);
    y[N - 1] = x[N - 1] ^ (~x[m - 1] & x[m]);
    return y;
}

// Linear layer function
template<size_t N>
bitset<N> linear_layer(const bitset<N>& x, const LinearParams& params)
{
    bitset<N> y;
    for (size_t i = 0; i < N; i++)
    {
        size_t idx0 = (params.alpha * i + params.beta0) % N;
        size_t idx1 = (params.alpha * i + params.beta1) % N;
        size_t idx2 = (params.alpha * i + params.beta2) % N;
        y[i] = x[idx0] ^ x[idx1] ^ x[idx2];
    }
    return y;
}

// Generate round constant
bitset<32> get_round_constant(int round, bool for_Chilow40 = false)
{
    uint32_t c = round;
    c ^= (1 << (round + 4));
    if (for_Chilow40)     // Domain separation for Chilow-40
        c ^= (1 << 31);
    return bitset<32>(c);
}

// ChiLow-(32 + τ) Decryption
bitset<32> Chilow32_decrypt(int ROUNDS, bitset<TWEAK_SIZE> T_state, bitset<BLOCK32_SIZE> X1, bitset<BLOCK32_SIZE> X2)
{

    for (int i = 0; i < ROUNDS - 1; i++)
    {
        X1 = ChiChi<BLOCK32_SIZE>(X1);
        X2 = ChiChi<BLOCK32_SIZE>(X2);
        T_state = ChiChi<TWEAK_SIZE>(T_state);

        X1 = linear_layer<BLOCK32_SIZE>(X1, L32_params);
        X2 = linear_layer<BLOCK32_SIZE>(X2, L32_params);
        T_state = linear_layer<TWEAK_SIZE>(T_state, L64_params);

        X1 ^= create_bitset<BLOCK32_SIZE>(extract_bits(T_state, 0, 32));
        X2 ^= create_bitset<BLOCK32_SIZE>(extract_bits(T_state, 0, 32));
    }

    X1 = ChiChi<BLOCK32_SIZE>(X1);
    X2 = ChiChi<BLOCK32_SIZE>(X2);

    bitset<32> X;
    for (int i = 0; i < 32; i++)
        X[i] = X1[i] ^ X2[i];

    X = linear_layer<BLOCK32_SIZE>(X, L32_params);

    return X;
}



// ChiLow-40 Decryption
bitset<40> Chilow40_decrypt(int ROUNDS, bitset<TWEAK_SIZE> T_state, bitset<40> X1, bitset<40> X2)
{

    for (int i = 0; i < ROUNDS - 1; i++)
    {
        X1 = ChiChi<40>(X1);
        X2 = ChiChi<40>(X2);
        T_state = ChiChi<TWEAK_SIZE>(T_state);

        X1 = linear_layer<40>(X1, L40_params);
        X2 = linear_layer<40>(X2, L40_params);
        T_state = linear_layer<TWEAK_SIZE>(T_state, L64_params);

        X1 ^= create_bitset<40>(extract_bits(T_state, 0, 40));
        X2 ^= create_bitset<40>(extract_bits(T_state, 0, 40));
    }

    X1 = ChiChi<40>(X1);
    X2 = ChiChi<40>(X2);

    bitset<40> X;
    for (int i = 0; i < 40; i++)
        X[i] = X1[i] ^ X2[i];

    X = linear_layer<40>(X, L40_params);

    return X;
}





void D32(int count, vector<int> Delta, vector<int> lambda)
{
    uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);  // 0 ~ 2^64 - 1
    uniform_int_distribution<uint32_t> dist32(0, UINT32_MAX);  // 0 ~ 2^32 - 1


    bitset<32> IV(0);
    for (auto m : Delta)
        IV.set(m);


    int counts = 0;
    clock_t start = clock();
#pragma omp parallel for
    for (int i = 0; i < count; i++)
    {
        uint64_t val64 = dist64(rng);
        uint32_t val32 = dist32(rng);
        bitset<64> T(val64);
        bitset<32> C(val32);
        bitset<32> C2;

        for (int j = 0; j < 32; j++)
            C2[j] = C[j] ^ IV[j];

        bitset<32> plaintext = Chilow32_decrypt(3, T, C, C2);
#pragma omp critical
        {
            char t = 0;
            for (auto m : lambda)
                t = t ^ plaintext[m];
            counts += t;
        }
    }
    clock_t end = clock();

    double elapsed_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_seconds << " s\n";

    double correlation = 1 - (2.0 * counts / count);
    cout << "Correlation: " << correlation << endl;
};


void D40(int count, vector<int> Delta, vector<int> lambda)
{
    uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);  // 0 ~ 2^64 - 1
    uniform_int_distribution<uint64_t> dist40(0, 1099511627775);  // 0 ~ 2^40 - 1

    bitset<40> IV(0);
    for (auto m : Delta)
        IV.set(m);


    int counts = 0;
    clock_t start = clock();
#pragma omp parallel for
    for (int i = 0; i < count; i++)
    {

        uint64_t val64 = dist64(rng);
        uint64_t val40 = dist40(rng);
        bitset<64> T(val64);
        bitset<40> C(val40);
        bitset<40> C2;

        for (int j = 0; j < 40; j++)
            C2[j] = C[j] ^ IV[j];

        bitset<40> plaintext = Chilow40_decrypt(3, T, C, C2);
#pragma omp critical
        {
            char t = 0;
            for (auto m : lambda)
                t = t ^ plaintext[m];
            counts += t;
        }
    }
    clock_t end = clock();

    double elapsed_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_seconds << " s\n";

    double correlation = 1 - (2.0 * counts / count);
    cout << "Correlation: " << correlation << endl;
};




int main()
{
    int count = 1073741824;


    //  Delta:  0x01403000   lambda:  0x00000804    r = 2^{-10.43}
    cout << "Delta:  0x01403000     lambda:  0x00000804" << endl;
    vector<int> Delta1 = { 7,9,18,19 };
    vector<int> lambda1 = { 20,29 };

    D32(count, Delta1, lambda1);



    //  Delta:  0x840000a0   lambda:  0x00000804    r = 2^{-11.98}
    cout << "Delta:  0x840000a0   lambda:  0x00000804" << endl;
    vector<int> Delta2 = { 0,5,24,26 };
    vector<int> lambda2 = { 20,29 };
    D32(count, Delta2, lambda2);



    //  Delta:  0x8000242000   lambda:  0x0900000100    r = 2^{-7.00}
    cout << "Delta:  0x8000242000   lambda:  0x0900000100" << endl;
    vector<int> Delta3 = { 0,18,21,26 };
    vector<int> lambda3 = { 4,7,31 };
    D40(count, Delta3, lambda3);



    //  Delta:  0x0410000004   lambda:  0x0004000400    r = 2^{-11.38}
    cout << "Delta:  0x0410000004   lambda:  0x0004000400" << endl;
    vector<int> Delta4 = { 5,11,37 };
    vector<int> lambda4 = { 13,29 };
    D40(count, Delta4, lambda4);


    //  Delta:  0x0004000800   lambda:  0x0004000400    r = 2^{-8.67}
    cout << "Delta:  0x0004000800   lambda:  0x0004000400" << endl;
    vector<int> Delta5 = { 13,28 };
    vector<int> lambda5 = { 13,29 };
    D40(count, Delta5, lambda5);


    system("pause");
    return 0;
}

