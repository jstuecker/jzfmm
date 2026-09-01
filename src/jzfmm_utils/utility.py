import time
import jax
import numpy as np

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
        self.verbose = verbose

        self.print_compile = print_compile
        self.print_warmup = print_warmup
        self.print_run = print_run
        self.loops = loops

        self.dts = [{}]
        self.group_tags = [{}]

    def set_tag(self, **kwargs):
        self.group_tags.append(kwargs)
        self.dts.append({})
        print("------- Starting new group: %s -------" % kwargs)

    def add_time(self, name, dt):
        self.dts[-1][name] = dt
        if self.verbose:
            self.print_time_of(name)

    def print_time_of(self, name=None, tagid=-1):
        if name is None: name = list(self.dts[tagid].keys())[-1]
        print(f"{(self.dts[tagid][name])*1000.:6.1f} ms for {name}")

    def timeit_jit(self, func, *args, name=None, loops=None, static_argnames=None, **kwargs):
        loops = loops if loops is not None else self.loops

        if hasattr(func, "lower"): # Function is already jitted
            func_jit = func
        else:
            func_jit = jax.jit(func, static_argnames=static_argnames)
        if name is None:
            name = func.__name__

        def call_func():
            return jax.block_until_ready(func_jit(*args, **kwargs))

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
            self.add_time(name + "_run", (t3 - t2) / loops)

        return res
    
    def transpose_timings(self, only_runs=True, simplify_names=True):
        unique_vars = set().union(*(group.keys() for group in self.group_tags))
        # res_names = tuple(unique_vars)
        # res_vals = {name: [] for name in res_names}
        dts = {}
        vars = {}

        for dtgroup, tag in zip(self.dts, self.group_tags):
            for name in dtgroup:
                if name not in dts:
                    dts[name] = []
                    vars[name] = {var:[] for var in unique_vars}
                dts[name].append(dtgroup[name])

                for var in unique_vars:
                    vars[name][var].append(tag.get(var, np.nan))

        for name in dts:
            dts[name] = np.array(dts[name])
            for var in unique_vars:
                vars[name][var] = np.array(vars[name][var])

        if only_runs and simplify_names:
            dts = {name[:-4]: dt for name, dt in dts.items() if name.endswith("_run")}
            vars = {name[:-4]: var for name, var in vars.items() if name.endswith("_run")}
        elif only_runs:
            dts = {name: dt for name, dt in dts.items() if name.endswith("_run")}
            vars = {name: var for name, var in vars.items() if name.endswith("_run")}

        return dts, vars
    
    def plot_timings(self, key, ax=None, logx=True, logy=True, save=None):
        import matplotlib.pyplot as plt
        dts, vars = self.transpose_timings()

        if ax is None:
            ax = plt.gca()

        for name in dts:
            if np.all(np.isnan(vars[name][key])):
                continue
            
            ax.plot(vars[name][key], dts[name]*1e3, label=name, marker="o")
        ax.set_xlabel(key)
        ax.set_ylabel("Time (ms)")

        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")

        ax.grid("on")
        
        ax.legend()

        if save is not None:
            plt.savefig(save, bbox_inches="tight")

        return ax

def bytes_str(bytes):
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.1f} kB"
    elif bytes < 1024**3:
        return f"{bytes / 1024**2:.1f} MB"
    else:
        return f"{bytes / 1024**3:.1f} GB"

def print_memory_usage(fcompiled):
    """Use with fcompiled=jax.jit(f).lower(args).compile()"""
    m = fcompiled.memory_analysis()

    print(f"temp bytes: {bytes_str(m.temp_size_in_bytes)}")
    print(f"arg  bytes: {bytes_str(m.argument_size_in_bytes)}")
    print(f"output bytes: {bytes_str(m.output_size_in_bytes)}")
    print(f"alias bytes: {bytes_str(m.alias_size_in_bytes)}")
    print(f"EST. peak bytes: {bytes_str(m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes)}")