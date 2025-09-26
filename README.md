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

# ToDo KNN:
 - Test Query Points != Tree Points
 - Check optimizations for reduced query points (most simple and promising: larger leafsizes)


 - Learn to deal with infinite nodes ? 


 - Consider possibility of building interaction list partially to reduce memory usage
 - make double inputs work
 - remove radii fromn PosR?

 - noradii output option
 - subset option

 - Consider recursive summarization for large initial max_size or large rfac
 - Fix tiny bug for rfac = 2

- test non-uniform distributions

# ToDo:
* Think about how to do force
* Name and structure all evaluation code consistently
* Get an understanding of which compile times are very slow?
* Measure interaction list potential performance
* Think about universal way to deal with memory parameters (e.g. a variable indicating max temporary size or so)
* Dynamic handling of interaction-overflow