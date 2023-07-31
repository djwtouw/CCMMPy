#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <algorithm>
#include <vector>
#include <map>

#include "graphs.h"


struct DisjointSet {
    std::vector<int> id;
    std::vector<int> sz;

    DisjointSet(int N)
    {
        // Set the id of each object to itself and set the sizes to one
        id.resize(N);
        sz.resize(N);

        for (int i = 0; i < N; i++) {
            id[i] = i;
            sz[i] = 1;
        }
    }

    int root(int i)
    {
        // Ascend through the tree until the root is found and apply path
        // compression on the way up
        while(i != id[i]) {
            id[i] = id[id[i]];
            i = id[i];
        }

        return i;
    }

    bool connected(int p, int q)
    {
        // Check if p and q have the same root
        return root(p) == root(q);
    }

    void merge(int p, int q)
    {
        // Change the parent of the root of p into the root of q
        int i = root(p);
        int j = root(q);

        // Return if the roots are the same
        if (i == j) return;

        // Otherwise link the root of the smaller tree to the root of the larger
        // tree
        if (sz[i] < sz[j]) {
            id[i] = j;
            sz[j] += sz[i];
        } else {
            id[j] = i;
            sz[i] += sz[j];
        }
    }
};


struct Edges {
    std::vector<int> a;
    std::vector<int> b;
    std::vector<double> w;

    Edges(const Eigen::MatrixXd& G)
    {
        // Number of edges
        int n = (G.cols() * G.cols() - G.cols()) >> 1;

        a.resize(n);
        b.resize(n);
        w.resize(n);

        int index = 0;

        for (int j = 0; j < G.cols(); j++) {
            for (int i = 0; i < j; i++) {
                a[index] = i;
                b[index] = j;
                w[index] = G(i, j);

                index++;
            }
        }
    }

    void sort()
    {
        int n = a.size();

        // Vector of indices
        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;

        // Sort based on the weights
        std::sort(
            indices.begin(), indices.end(),
            [&](int i, int j) { return w[i] < w[j]; }
        );

        // New versions of a, b, and w
        std::vector<int> a_new(n);
        std::vector<int> b_new(n);
        std::vector<double> w_new(n);

        for (int i = 0; i < n; i++) {
            a_new[i] = a[indices[i]];
            b_new[i] = b[indices[i]];
            w_new[i] = w[indices[i]];
        }

        // Assign
        a = a_new;
        b = b_new;
        w = w_new;
    }

    int size() const
    {
        return a.size();
    }

    int u(int index) const
    {
        return a[index];
    }

    int v(int index) const
    {
        return b[index];
    }
};


Eigen::MatrixXi find_mst(const Eigen::MatrixXd& G)
{
    // Initialize a disjoint set
    DisjointSet djs(G.cols());

    // Gather edges from the graph and sort them based on their weights
    Edges E(G);
    E.sort();

    // Initialize minimum spanning tree as a matrix of integers
    Eigen::MatrixXi mst(2, G.cols() - 1);
    int mst_index = 0;

    // Apply the remainder of Kruskal's algorithm, adding edges with the
    // smallest weight unless they cause a loop
    for (int i = 0; i < E.size(); i++) {
        if (!djs.connected(E.u(i), E.v(i))) {
            mst(0, mst_index) = E.u(i);
            mst(1, mst_index) = E.v(i);
            mst_index++;

            djs.merge(E.u(i), E.v(i));
        }
    }

    return mst.transpose();
}


Eigen::VectorXi find_subgraphs(const Eigen::MatrixXi& E, int n)
{
    // Initialize a disjoint set
    DisjointSet djs(n);

    // Fill the disjoint set
    for (int i = 0; i < E.cols(); i++) {
        int u = E(0, i);
        int v = E(1, i);
        djs.merge(u, v);
    }

    // Initialize vector of cluster IDs
    Eigen::VectorXi id(n);

    // The roots are random values, we want consecutive cluster IDs, so we make
    // a map for that
    std::map<int, int> id_dict;

    // Initialize the cluster id
    int c = 0;

    for (int i = 0; i < n; i++) {
        int root = djs.root(i);

        // If the root is not present in the dictionary, add it and give it a
        // new cluster id
        auto it = id_dict.find(root);
        if (it == id_dict.end()) {
            id_dict[root] = c;
            c++;
        }

        // Assign the object the correct id
        id(i) = id_dict[root];
    }

    return id;
}
