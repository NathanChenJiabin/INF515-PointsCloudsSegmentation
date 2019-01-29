//
// Created by jiabin on 2019/1/22.
//

#ifndef BILATERALNN_PERMUTOHEDRALLATTICECPU_H
#define BILATERALNN_PERMUTOHEDRALLATTICECPU_H

#include <cstddef>
#include <vector>
#include <math.h>
#include <memory>
// #include <tclDecls.h>

using namespace std;
using std::vector;


// Hash table implementation for permutohedral lattice.
//
// The lattice points are stored sparsely using a hash table.
// The key for each point is its spatial location in the (d+1)-
// dimensional space.
template <typename T>
class HashTablePermutohedral {
public:
    // Hash table constructor
    // kd : the dimensionality of the position vectors
    // vd : the dimensionality of the value vectors

    HashTablePermutohedral(int kd, int vd) : kd(kd), vd(vd) {
        filled = 0;
        entries.resize(1 << 15);
        keys.resize(kd*entries.size()/2);
        values.resize(vd*entries.size()/2, 0);
    }

    // Returns the number of vectors stored.
    int get_size() { return filled; }
    // Returns a pointer to the keys array.
    vector<short> &getKeys() { return keys; }
    // Returns a pointer to the values array.
    vector<T> &getValues() { return values; }
    // Looks up the value vector associated with a given key. May or
    // may not create a new entry if that key doesn’t exist.
    T *lookup(const vector<short> &key, bool create){
        // Double hash table size if necessary
        if (create && filled >= entries.size()/2 - 1) { grow(); }

        // Hash the key
        size_t h = hash(&key[0]) % entries.size();

        // Find the entry with the given key
        while (true) {
            Entry e = entries[h];
            // Check if the cell is empty
            if (e.keyIdx == -1) { // -1 means cell is empty
                if (!create) return nullptr; // Not found if we don't need to create

                // Need to create an entry. Store the given key.
                for (int i = 0; i < kd; i++) {
                    keys[filled*kd+i] = key[i];
                }
                e.keyIdx = static_cast<int>(filled * kd);
                e.valueIdx = static_cast<int>(filled * vd);
                entries[h] = e;
                filled++;
                return &values[e.valueIdx];
            }
            // check if the cell has a matching key
            bool match = true;

            for (int i = 0; i < kd && match; i++) {
                match = keys[e.keyIdx+i] == key[i];
            }
            if (match) {
                return &values[e.valueIdx];
            }
            // increment the bucket with wraparound
            h++;
            if (h == entries.size()) { h = 0; }
        }

    }

    // Hash function used in this implementation. A simple base conversion.
    size_t hash(const short *key) {
        size_t h = 0;
        for (int i = 0; i < kd; i++) {
            h += key[i];
            h *= 2531011;
        }
        return h;
    }


private:
    // Grows the hash table when it runs out of space
    void grow() {
        // Grow the arrays
        values.resize(vd*entries.size(), 0);
        keys.resize(kd*entries.size());
        vector<Entry> newEntries(entries.size()*2);
        // Rehash all the entries
        for (size_t i = 0; i < entries.size(); i++) {
            if (entries[i].keyIdx == -1) { continue; }

            size_t h = hash(&keys[entries[i].keyIdx]) % newEntries.size();

            while (newEntries[h].keyIdx != -1) {
                h++;
                if (h == newEntries.size()) { h = 0; }
            }
            newEntries[h] = entries[i];
        }
        entries.swap(newEntries);
    }

    // Private struct for the hash table entries.
    struct Entry {
        Entry(): keyIdx(-1), valueIdx(-1) {}
        int keyIdx;
        int valueIdx;
    };
    vector<short> keys;
    vector<T> values;
    vector<Entry> entries;
    size_t filled;
    int kd, vd;
};

template <typename T>
class PermutohedralLattice {
public:
    // Constructor
    PermutohedralLattice(int pd_, int vd_, int n_) :
            d(pd_), vd(vd_), n(n_), hashTable(pd_, vd_) {
        // Allocate storage for various arrays
        scale = 1.0f/(d+1);
        elevated.resize(d+1);
        scaleFactor.resize(d);
        greedy.resize(d+1);
        rank.resize(d+1);
        barycentric.resize(d+2);
        canonical.resize((d+1)*(d+1));
        key.resize(d+1);
        replay.resize(n*(d+1));
        nReplay = 0;
        // compute the coordinates of the canonical simplex, in which
        // the difference between a contained point and the zero
        // remainder vertex is always in ascending order. (see pg.4 in paper)
        for (int i = 0; i <= d; i++) {
            for (int j = 0; j <= d-i; j++) {
                canonical[i*(d+1)+j] = static_cast<short>(i);
            }
            for (int j = d-i+1; j <= d; j++) {
                canonical[i*(d+1)+j] = static_cast<short>(i - (d + 1));
            }
        }
        // Compute part of the rotation matrix E that elevates a
        // position vector into the hyperplane (see pg.4-5 in paper)
        for (int i = 0; i < d; i++) {
            // the diagonal entries for normalization
            scaleFactor[i] = 1.0/(sqrt((i+1)*(i+2)));
            scaleFactor[i] *= (d+1)*sqrt(2.0/3);
        }
    }


    // Performs splatting with given position and value vectors
    void splat(const T *position, const T *value) {

        // Elevate position into the (d+1)-dimensional hyperplane
        elevated[d] = -d*position[d-1]*scaleFactor[d-1];
        for (int i = d-1; i > 0; i--)
            elevated[i] = (elevated[i+1] -
                           i*position[i-1]*scaleFactor[i-1] +
                           (i+2)*position[i]*scaleFactor[i]);
        elevated[0] = elevated[1] + 2*position[0]*scaleFactor[0];

        find_enclosing_simplex();

        // Compute barycentric coordinates
        compute_barycentric_coordinates();

        // Splat the value into each vertex of the simplex, with
        // barycentric weights
        for (int remainder = 0; remainder <= d; remainder++) {
            // Compute the location of the lattice point explicitly
            // (all but the last coordinate - it’s redundant because
            // they sum to zero)
            for (int i = 0; i < d; i++) {
                key[i] = greedy[i] + canonical[remainder*(d+1) + rank[i]];
            }

            // Retrieve pointer to the value at this vertex
            T *val = hashTable.lookup(key, true);

            // Accumulate values with barycentric weight
            for (int i = 0; i < vd; i++) {
                val[i] += barycentric[remainder]*value[i];
            }

            // Record this interaction to use later when slicing
            replay[nReplay].offset = static_cast<int>(val - &(hashTable.getValues()[0]));
            replay[nReplay].weight = barycentric[remainder];
            nReplay++;
        }

    }


    // Performs a Gaussian blur along each projected axis in the hyperplane.
    void blur() {
        // Prepare temporary arrays
        vector<short> neighbor1(d+1), neighbor2(d+1);
        vector<T> zero(vd, 0);
        vector<T> newValue(hashTable.get_size()*vd);
        vector<T> &oldValue = hashTable.getValues();
        // For each of d+1 axes,
        for (int j = 0; j <= d; j++) {
            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.get_size(); i++) { // blur point i along axis j
                // Blur point i in dimension j
                short *key = &(hashTable.getKeys()[i*d]);
                for (int k = 0; k < d; k++) {
                    neighbor1[k] = static_cast<short>(key[k] + 1);
                    neighbor2[k] = static_cast<short>(key[k] - 1);
                }
                neighbor1[j] = static_cast<short>(key[j] - d);
                neighbor2[j] = static_cast<short>(key[j] + d);
                T *oldVal = &oldValue[i*vd];
                T *newVal = &newValue[i*vd];

                T *v1 = hashTable.lookup(neighbor1, false); // look up first neighbor
                if (v1 == nullptr) v1 = &zero[0];

                T *v2 = hashTable.lookup(neighbor2, false); // look up second neighbor
                if (v2 == nullptr) v2 = &zero[0];

                // Mix values of the three vertices
                for (int k = 0; k < vd; k++) {
                    newVal[k] = 0.25 * v1[k] + 0.5 * oldVal[k] + 0.25 * v2[k] ;
                }
            }

            std::swap(oldValue, newValue);
            // newValue.swap(oldValue);
        }

    }


    // Prepare for slicing
    void beginSlice() {
        nReplay = 0;
    }

    // Performs slicing out of position vectors. The barycentric
    // weights and the simplex containing each position vector were
    // calculated and stored in the splatting step.
    void slice(T *col){
        const vector<T> &vals = hashTable.getValues();
        for (int j = 0; j < vd; j++) { col[j] = 0; } // set 0

        for (int i = 0; i <= d; i++) {
            ReplayEntry r = replay[nReplay++];
            for (int j = 0; j < vd; j++) {
                col[j] += r.weight*vals[r.offset + j];
            }
        }
    }


private:
    int d, vd, n;
    float scale;
    vector<T> elevated, scaleFactor, barycentric;
    vector<short> canonical, key, greedy;
    vector<char> rank;
    struct ReplayEntry {
        int offset;
        T weight;
    };
    vector<ReplayEntry> replay;
    int nReplay;
    HashTablePermutohedral<T> hashTable;

    void find_enclosing_simplex(){
        // Prepare to find the closest lattice points
        // float scale = 1.0f/(d+1);
        // Greedily search for the closest remainder-zero lattice point
        int sum = 0;
        for (int i = 0; i <= d; i++) {
            T v = elevated[i]*scale;
            T up = ceil(v) * (d+1);
            T down = floor(v) * (d+1);
            if (up - elevated[i] < elevated[i] - down) {
                greedy[i] = (short)up;
            } else {
                greedy[i] = (short)down;
            }
            sum += greedy[i];
        }
        sum /= d + 1;

        // Rank differential to find the permutation between this
        // simplex and the canonical one. (see pg. 3-4 in paper)
        for (int i = 0; i < d+1; i++) rank[i] = 0; // reset rank

        for (int i = 0; i < d; i++) {
            T di = elevated[i] - greedy[i];
            for (int j = i+1; j <= d; j++) {
                if ( di < elevated[j] - greedy[j]) {
                    rank[i]++;
                } else {
                    rank[j]++;
                }
            }
        }

        if (sum > 0) {
            // Sum too large - the point is off the hyperplane. We
            // need to bring down the ones with the smallest
            // differential
            for (int i = 0; i <= d; i++) {
                if (rank[i] >= d + 1 - sum) {
                    greedy[i] -= d+1;
                    rank[i] += sum - (d+1);
                }else {
                    rank[i] += sum;
                }
            }
        } else if (sum < 0) {
            // Sum too small - the point is off the hyperplane. We
            // need to bring up the ones with largest differential
            for (int i = 0; i <= d; i++) {
                if (rank[i] < -sum) {
                    greedy[i] += d+1;
                    rank[i] += (d+1) + sum;
                }else {
                    rank[i] += sum;
                }
            }
        }
    }

    void compute_barycentric_coordinates(){
        for (int i = 0; i < d+2; i++) { barycentric[i] = 0; } // reset barycentric

        for (int i = 0; i <= d; i++) {
            T delta = (elevated[i] - greedy[i]) * scale;
            barycentric[d - rank[i]] += delta ;
            barycentric[d + 1 - rank[i]] -= delta ;
        }
        barycentric[0] += 1.0 + barycentric[d+1];
    }

};

// Performs a Gauss transform
// pos : position vectors
// pd : position dimensions
// val : value vectors
// vd : value dimensions
// n : number of items to filter
// out : place to store the output
template <typename T>
static void filter(const T *pos,
                   const T *val,
                   T *out,
                   int pd,
                   int vd,
                   int n,
                   bool reverse= false){
    // Create lattice
    PermutohedralLattice<T> lattice(pd, vd, n);
    // Splat
    for (int i = 0; i < n; i++) {
        lattice.splat(pos + i*pd, val + i*vd);
    }
    // Blur
    lattice.blur();
    // Slice
    lattice.beginSlice();
    for (int i = 0; i < n; i++) {
        lattice.slice(out + i*vd);
    }
}
template void filter<float>(const float *pos,
                            const float *val,
                            float *out,
                            int pd,
                            int vd,
                            int n,
                            bool reverse= false);
template void filter<double>(const double *pos,
                            const double *val,
                            double *out,
                            int pd,
                            int vd,
                            int n,
                            bool reverse= false);


#endif //BILATERALNN_PERMUTOHEDRALLATTICECPU_H
