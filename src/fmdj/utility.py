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
    def __init__(self, verbose=True, print_compile=True, print_warmup=True, print_run=True, loops=40):
        self.dts = {}
        self.verbose = verbose

        self.print_compile = print_compile
        self.print_warmup = print_warmup
        self.print_run = print_run
        self.loops = loops

    def add_time(self, name, dt):
        self.dts[name] = dt
        if self.verbose:
            self.print_time_of(name)

    def print_time_of(self, name=None):
        if name is None: name = list(self.dts.keys())[-1]
        print(f"Time for {name}: {(self.dts[name])*1000.:.1f} ms")

    def timeit_jit(self, func, *args, name=None, loops=None, static_argnames=None, **kwargs):
        loops = loops if loops is not None else self.loops

        if hasattr(func, "lower"): # Function is already jitted
            func_jit = func
        else:
            func_jit = jax.jit(func, static_argnames=static_argnames)
        if name is None:
            name = func.__name__

        def call_func():
            val = func_jit(*args, **kwargs)
            res = val
            while isinstance(val, tuple):
                val = val[1]
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

        if self.print_compile:
            self.add_time(name + "_compile", t1 - t0)
        if self.print_warmup:
            self.add_time(name + "_warmup", t2 - t1)
        if self.print_run:
            self.add_time(name + "_run[%d]" % loops, (t3 - t2) / loops)

        return res