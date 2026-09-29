#!/usr/bin/env python3
"""CPU 基准：单线程 / 多线程 / 加速比 / 浮点与整数吞吐"""
import time, os, multiprocessing as mp

def int_work(n):
    s = 0
    for i in range(n):
        s = (s + i * i) % 1000000007
    return s

def fp_work(n):
    s = 0.0
    for i in range(n):
        s += (i ** 0.5) * 1.000001
    return s

def bench_one(n_int=3_000_000, n_fp=3_000_000):
    t = time.time(); int_work(n_int); ti = time.time() - t
    t = time.time(); fp_work(n_fp);   tf = time.time() - t
    return ti, tf

def _worker(_):
    return bench_one()

if __name__ == "__main__":
    n = os.cpu_count()
    bench_one(300_000)  # warmup
    ti, tf = bench_one()
    print("单线程  整数循环 300万次: %.3f s" % ti)
    print("单线程  浮点循环 300万次: %.3f s" % tf)
    with mp.Pool(n) as p:
        t = time.time(); p.map(_worker, range(n)); multi = time.time() - t
    seq = ti + tf
    print()
    print("%d 线程并发 %d 份任务: %.3f s" % (n, n, multi))
    print("实测并行加速比: %.2fx   (理想 %d.0x，线性度 %.0f%%)" % (seq * n / multi, n, 100 * (seq * n / multi) / n))
