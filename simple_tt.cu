// includes, system
#include "datatypes.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

#include <shrUtils.h>
#include <cutil_inline.h>

using namespace std;


/** sample function: sum up path from root to node */
__device__ void computation(TreeNode *parent, TreeNode *curr) {
    curr->value = curr->value + parent->value;
}

/** one step of the traversal */
__global__ void tt_step(
    int64_t lvl_start,
    int64_t lvl_end,
    int64_t nnodes_blk,
    TreeNode *slab
    )
{
    // range that this block will work on
    TreeNode *start = &slab[lvl_start + nnodes_blk * blockIdx.x];
    TreeNode *end = &start[nnodes_blk];
    if (end > &slab[lvl_end]) {
        end = &slab[lvl_end];
    }

    // loop over all of the nodes
    for (TreeNode *n = start + threadIdx.x; n < end; n += blockDim.x) {
        computation(&slab[n->parent_idx], n);
    }
}


TreeNode *Tree::copy_to_gpu() {
    TreeNode *res = NULL;
    cutilSafeCall(cudaMalloc((void**) &res, nbytes()));
    cutilSafeCall(cudaMemcpy(res, slab, nbytes(),
        cudaMemcpyHostToDevice)); 
    return res;
}

/** make sure to sync before this! */
Tree Tree::copy_from_gpu(TreeNode *gpu_ptr) {
    Tree res(tree_sz);
    res.levels = levels;
    cutilSafeCall( cudaMemcpy( res.slab, gpu_ptr,
        nbytes(), cudaMemcpyDeviceToHost));
    return res;
}

int64_t div_ceil(int64_t x, int64_t y) {
    return (x / y) + ((x % y != 0) ? 1 : 0);
}

/** NOTE: a bit of copied code */
int main(int argc, char** argv) {
    Tree t = generate_tree(20);
    t.print();

    // use command-line specified CUDA device,
    // otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    TreeNode *gpu_tree = t.copy_to_gpu();
    cutilSafeCall( cudaThreadSynchronize() );
    shrLog("*** Running test on %d bytes of input\n", t.nbytes());

    Tree::LevelIter it = t.levels->begin();
    ++it; // skip the root
    for (; it != t.levels->end(); it++) {
        TreeLevel lvl = *it;
        int64_t nthreads_blk = 2;
        // NOTE: can make each thread process more than one node
        int64_t nnodes_blk = nthreads_blk;
        int64_t nblocks = div_ceil(lvl.nnodes, nnodes_blk);

        cout << "running with " << nblocks <<
            " blocks of " << nthreads_blk << " threads each" << endl;
        tt_step<<<nblocks, nthreads_blk>>>(lvl.start, lvl.end(),
            nnodes_blk, gpu_tree);
    }

    cutilSafeCall( cudaThreadSynchronize() );
    Tree t_res = t.copy_from_gpu(gpu_tree);

    cout << endl << endl << "=== After computation" << endl;
    t_res.print();

    return 0;
}
