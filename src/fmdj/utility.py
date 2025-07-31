import time
import jax
from .octree import BinaryTree, Octree

class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

class Timer():
    def __init__(self, verbose=True):
        self.dts = {}
        self.verbose = verbose

        self.tstart = self.tlast = time.time()

    def add_time(self, name, dt):
        self.dts[name] = dt
        if self.verbose:
            self.print_time_of(name)

    def print_time_of(self, name=None):
        if name is None: name = list(self.dts.keys())[-1]
        print(f"Time for {name}: {(self.dts[name])*1000.:.1f} ms")
    
    def measure(self, array, name="step"):
        array.block_until_ready()
        newt = time.time()
        self.dts[name] = newt - self.tlast
        self.tlast = newt
        
        if self.verbose:
            self.print_time_of(name)

    def timeit_jit(self, func, *args, name="step", loops=40, static_argnames=None, only_print_run=False,  **kwargs):
        func_jit = jax.jit(func, static_argnames=static_argnames)
        def call_func():
            val = func_jit(*args, **kwargs)
            res = val
            if isinstance(val, tuple):
                val = val[0]
            if isinstance(val, BinaryTree):
                val.lchild.block_until_ready()
            elif isinstance(val, Octree):
                val.lchild.block_until_ready()
            else:
                val.block_until_ready()
            return res

        t0 = time.time()
        func_jit.lower(*args, **kwargs).compile()
        t1 = time.time()
        res = call_func() # warmup
        t2 = time.time()
        for _ in range(loops):
            call_func()
        t3 = time.time()

        if not only_print_run:    
            self.add_time(name + "_compile", t1 - t0)
            self.add_time(name + "_warmup", t2 - t1)
        self.add_time(name + "_run[%d]" % loops, (t3 - t2) / loops)

        return res