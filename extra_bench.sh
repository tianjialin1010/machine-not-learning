#!/usr/bin/env bash
###############################################################################
#  第二轮补充测评：内存带宽(STREAM) + CPU 多核 + 包源 + 端口映射服务
###############################################################################
set -uo pipefail
PY=/usr/bin/python3

echo "═══ A. 安装 numpy（阿里云镜像）═══"
pip3 install --user --break-system-packages -q -i https://mirrors.aliyun.com/pypi/simple/ numpy 2>&1 | tail -3
$PY -c "import numpy; print('numpy', numpy.__version__, '✓')" 2>&1 | tail -1

echo
echo "═══ B. 内存带宽实测 ═══"
$PY - <<'PY'
import numpy as np, time
def timeit(fn, n=8, warm=2):
    for _ in range(warm): fn()
    t=time.time()
    for _ in range(n): fn()
    return (time.time()-t)/n

SIZE = 512*1024*1024              # 512 MiB
n = SIZE//8
a = np.ones(n, dtype=np.float64)
b = np.empty(n, dtype=np.float64)

def f_read():   a.sum()
def f_write():  b.fill(1.0)
def f_copy():   b[:] = a
def f_triad():  np.add(a, b, out=b)   # b = a + b

GB = SIZE/1024**3
r = timeit(f_read);   print("  只读    : %6.1f GB/s" % (GB/r))
w = timeit(f_write);  print("  只写    : %6.1f GB/s" % (GB/w))
c = timeit(f_copy);   print("  拷贝    : %6.1f GB/s  (读+写各 %.1f GB/s)" % (2*GB/c, GB/c))
t = timeit(f_triad);  print("  Triad   : %6.1f GB/s  (读2+写1 = %0.1f GB/s)" % (3*GB/t, 3*GB/t))
print()
print("  厂商标称带宽 273 GB/s（LPDDR5x 统一内存，CPU 与 GPU 共享）")
PY

echo
echo "═══ C. CPU 多核基准 ═══"
cat > /tmp/cpu_bench.py <<'PY'
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
def bench_one(wi=2_000_000, wf=2_000_000):
    t=time.time(); int_work(wi); ti=time.time()-t
    t=time.time(); fp_work(wf);  tf=time.time()-t
    return ti, tf
def _w(_): return bench_one()
if __name__ == "__main__":
    n = os.cpu_count()
    bench_one(200_000)
    ti, tf = bench_one()
    seq = ti + tf
    print("  单线程: 整数 200万次 %.3f s | 浮点 200万次 %.3f s" % (ti, tf))
    with mp.Pool(n) as p:
        t=time.time(); p.map(_w, range(n)); multi=time.time()-t
    sp = seq*n/multi
    print("  %d 线程并发: %.3f s" % (n, multi))
    print("  并行加速比: %.2fx  (理想 %d.0x，线性度 %.0f%%)" % (sp, n, 100*sp/n))
PY
$PY /tmp/cpu_bench.py

echo
echo "═══ D. 补充包源连通性 ═══"
for u in https://mirrors.aliyun.com/pypi/simple/ https://mirror.nju.edu.cn/pypi/web/simple/ https://nvcr.io/v2/ https://registry-1.docker.io/v2/ https://modelscope.cn; do
  printf "  %-42s " "$u"
  curl -s -o /dev/null -m 10 -w "HTTP %{http_code}  %{time_total}s\n" "$u" || echo "不可达"
done

echo
echo "═══ E. 启动测试服务绑定 0.0.0.0:8888（用于验证端口映射）═══"
pkill -f "http.server 8888" 2>/dev/null
cd /tmp && nohup $PY -m http.server 8888 --bind 0.0.0.0 >/tmp/httpd.log 2>&1 &
sleep 2
echo "  监听状态:"
ss -tlnp 2>/dev/null | grep 8888 || echo "  (未监听)"
echo "  PID: $(pgrep -f 'http.server 8888')"

echo
echo "═══ F. GPU 空闲状态复查 ═══"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,utilization.gpu --format=csv
