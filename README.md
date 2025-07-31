# FM-DJ
Fast Multipoles Done with Jax


# Performance Considerations:
Which parts of the algorithm could still be sped up?
* octree.find_previous_and_next_lower could likely be made much faster. However, it is not a bottlenecks
* multipoles.com_via_levels and multipoles.multipoles_via_levels are in principle doing a lot of redundant work. However, they are also nowhere close to being bottlenecks
* Lexsort with 3i32 seems about two times slower than argsort with i32 and stable=False for ~8e6 particles

# Other questions?
* Should I always open leaf-node interactions?
* I can skip the first 4 terms in the multipole update
* How to treat softening consistently in the Node interactions?
* Is Plummer softening the right choice?