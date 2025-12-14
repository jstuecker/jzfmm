# FM-DJ
Fast Multipoles Done with Jax

# ToDo:
* Add a unit test that compares the tree structure for different coarsening
* Adapt code to use new tree hierarchy efficiently / Don't materialize tree-planes
* Improve dense interaction allocation handling
* Think about ztree in multi GPU
* Put tree hierarchy construction into a CUDA kernel (?)
* Make some config parameters differentible
* Test forces in Hernquist potential
* Improve opening criterion
* Try out differently transposed multipoles
* Other softening

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