#pragma once

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

using namespace std;

struct TreeNode {
    int64_t parent_idx;
    int value;
    TreeNode() { parent_idx = -1; value = -1; }
    TreeNode(int64_t p, int v) : parent_idx(p), value(v) { }
};

struct TreeLevel {
    int64_t start;
    int64_t nnodes;
    TreeLevel(int64_t start_, int64_t nnodes_)
        : start(start_), nnodes(nnodes_) { }

    int64_t end() const { return start + nnodes; }
};

struct Tree {
    vector<TreeLevel> *levels;
    TreeNode *slab;
    int64_t tree_sz;

    typedef vector<TreeLevel>::iterator LevelIter;

    Tree(int64_t tree_sz_) : tree_sz(tree_sz_) {
        slab = new TreeNode[tree_sz];
        levels = new vector<TreeLevel>();
    }

    int64_t nbytes() const { return sizeof(TreeNode) * tree_sz; }

    TreeNode *copy_to_gpu();
    Tree copy_from_gpu(TreeNode *gpu_ptr);

    void print() {
        for(LevelIter it = levels->begin(); it != levels->end(); it++) {
            TreeLevel lvl = *it;
            for (int i = lvl.start; i < lvl.end(); i++) {
                if (i != lvl.start) {
                    cout << ", " << endl << "    ";
                }
                TreeNode n = slab[i];
                cout << "Node(parent=" << n.parent_idx <<
                    ", value=" << n.value << ")";
            }
            cout << endl;
        }
    }
};

Tree generate_tree(int64_t tree_sz);
