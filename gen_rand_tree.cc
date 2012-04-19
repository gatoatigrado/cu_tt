#include "datatypes.h"

#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

typedef boost::mt19937 RNGType; 

TreeLevel generate_level(
    boost::variate_generator< RNGType, boost::uniform_int<> > *rng,
    int64_t max_nodes,
    TreeLevel *prev,
    TreeNode *next)
{
    int child_idx = 0;
    // for each parent, generate a random number of children,
    // put them in the slab in increasing order
    for (int64_t pidx = prev->start;
         // second expression: cut off generation if we've reached the max # of nodes
         pidx < prev->end() && child_idx < max_nodes;
         pidx++)
    {
        int nchildren = (*rng)();
        for (int i = 0; i < nchildren && child_idx < max_nodes; i++) {
            next[child_idx] = TreeNode(pidx, (*rng)());
            child_idx += 1;
        }
    }

    return TreeLevel(prev->end(), child_idx);
}

Tree generate_tree(int64_t tree_sz) {
    // slab of tree nodes
    // organization of slab to make other code easier
    Tree tree(tree_sz);

    // add a root level
    tree.slab[0] = TreeNode(-1, 1);
    TreeLevel prev_level = TreeLevel(0, 1);
    tree.levels->push_back(prev_level);

    // generate the rest of the tree using a random
    // number generator
    RNGType rng;
    boost::uniform_int<> one_to_six(1, 6);    
    boost::variate_generator< RNGType, boost::uniform_int<> >
        dice(rng, one_to_six);

    for (int64_t lvl_idx = 1; lvl_idx < tree_sz;) {
        cout << "generating new level at " << lvl_idx << endl;
        prev_level = generate_level(
            &dice, tree_sz - lvl_idx, &prev_level, &tree.slab[lvl_idx]);
        lvl_idx += prev_level.nnodes;
        tree.levels->push_back(prev_level);
    }

    return tree;
}
