#!/usr/bin/env bash
###############################################################################
#  NVIDIA DGX Spark (GB10 Grace Blackwell) 一键测评脚本
#  用途：登录节点后直接执行，产出完整性能报告
#  用法：bash spark_bench.sh            # 结果写到 ~/spark_bench_<时间戳>/
#        bash spark_bench.sh ~/myout   # 指定输出目录
#  说明：不依赖任何需要联网安装的软件；缺什么就跳过什么，不会中断整体流程
###############################################################################
set -uo pipefail

OUT="${1:-$HOME/spark_bench_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
REPORT="$OUT/report.txt"
: > "$REPORT"

say(){ printf '\n\033[1;36m══ %s ══\033[0m\n' "$*" | tee -a "$REPORT"; }
sub(){ printf '\n--- %s ---\n' "$*" | tee -a "$REPORT"; }
run(){ printf '\n$ %s\n' "$*" | tee -a "$REPORT"; eval "$@" 2>&1 | tee -a "$REPORT"; }

say "0. 基础环境"
run "date; hostname; whoami"
run "uname -a"
run "cat /etc/os-release | head -4"
run "uptime"

say "1. CPU"
run "lscpu | sed -n '1,25p'"
run "nproc"
sub "短时 CPU 满载（10 秒，观察频率与降频）"
if command -v sysbench >/dev/null; then
  run "sysbench cpu --cpu-max-prime=20000 --threads=\$(nproc) --time=10 run"
else
  run "echo 'sysbench 未安装，使用 Python 多进程基准替代'"
  run "$(command -v python3 || echo /usr/bin/python3) - <<'PY'
import time,multiprocessing as mp,os
def work(n):
    s=0.0
    for i in range(n): s+=i**0.5
    return s
def bench():
    t=time.time(); work(6_000_000); return time.time()-t
if __name__=='__main__':
    t=time.time(); bench(); one=time.time()-t
    print('单线程耗时: %.2f s'%one)
    n=os.cpu_count() or 4
    with mp.Pool(n) as p:
        t=time.time(); p.map(lambda _: bench(), range(n)); multi=time.time()-t
    print('%d 线程耗时: %.2f s'%(n,multi))
    print('并行加速比: %.2fx  (理想 %.1fx)'%(one*n/multi, n))
PY"
fi

say "2. 内存"
run "free -h"
run "lscpu | grep -iE 'L3|NUMA node\(s\)|CPU\(s\)|Thread|Core' | head -10"
run "cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|SwapTotal'"
sub "内存带宽实测（纯 Python 内存拷贝，参考值）"
run "$(command -v python3 || echo /usr/bin/python3) - <<'PY'
import time
N=512*1024*1024          # 512 MiB
try:
    import numpy as np
    a=np.ones(N//8,dtype=np.float64); b=np.empty_like(a)
    for _ in range(2): b[:]=a          # warmup
    t=time.time()
    for _ in range(10): b[:]=a
    dt=time.time()-t
    gb=10*(N/1024**3)*2
    print('numpy 拷贝带宽: %.1f GB/s (读+写)'%(gb/dt))
except ImportError:
    a=bytearray(N); b=bytearray(N)
    t=time.time()
    for _ in range(5): b[:]=a
    dt=time.time()-t
    gb=5*(N/1024**3)*2
    print('bytearray 拷贝带宽: %.1f GB/s (读+写) [numpy 未安装，数值偏低属正常]'%(gb/dt))
PY"

say "3. GPU"
run "nvidia-smi"
run "nvidia-smi -q | grep -E 'Product Name|CUDA Version|Driver Version|Total *:|Memory Bus|Max Power|Compute Cap' | head -20"
run "nvidia-smi topo -m 2>/dev/null || echo '(topo 不可用)'"

say "4. 磁盘 IO"
run "df -hT | grep -vE 'tmpfs|overlay'"
sub "顺序写 / 顺序读（1 GiB）"
run "cd $OUT && dd if=/dev/zero of=iotest.bin bs=1M count=1024 oflag=direct 2>&1 | tail -1"
run "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; dd if=$OUT/iotest.bin of=/dev/null bs=1M iflag=direct 2>&1 | tail -1"
run "rm -f $OUT/iotest.bin"
if command -v fio >/dev/null; then
  sub "fio 随机 4K 读写 (QD32)"
  run "fio --name=randrw --ioengine=libaio --rw=randrw --bs=4k --numjobs=4 --size=1G --runtime=20 --group_reporting --directory=$OUT 2>&1 | grep -E 'read:|write:|IOPS|BW' | head -10"
fi

say "5. 网络"
sub "节点出口测速（下载 100MB，注意 50 台共用一条公网出口）"
run "curl -sL --max-time 60 -o /dev/null -w '  目标: %{remote_ip}\n  下载: %{speed_download} B/s\n  耗时: %{time_total} s\n' 'https://speed.cloudflare.com/__down?bytes=104857600'"
sub "延迟（国内公共 DNS）"
run "ping -c 5 -W 2 223.5.5.5 2>&1 | tail -3"

say "6. AI 软件栈"
run "nvcc --version 2>/dev/null | tail -2 || echo 'nvcc: 未安装'"
run "python3 -c 'import torch;print(\"PyTorch\",torch.__version__,\"| CUDA\",torch.version.cuda,\"| 设备\",torch.cuda.get_device_name(0),\"| capability\",torch.cuda.get_device_capability(0))' 2>&1 | tail -2"
run "docker --version 2>/dev/null || echo 'docker: 未安装'"
run "which ollama vllm sglang 2>/dev/null || echo '(未发现 ollama/vllm/sglang)'"
run "ollama list 2>/dev/null || true"
run "ls ~/.cache/huggingface/hub 2>/dev/null || echo '(HuggingFace 缓存为空)'"

say "7. GPU 算力实测（PyTorch）"
run "$(command -v python3 || echo /usr/bin/python3) - <<'PY'
try:
    import torch, time
except Exception as e:
    print('PyTorch 不可用，跳过:', e); raise SystemExit
if not torch.cuda.is_available():
    print('CUDA 不可用'); raise SystemExit
dev=torch.cuda.get_device_name(0)
print('设备:', dev)
props=torch.cuda.get_device_properties(0)
print('显存: %.1f GB | SM 数: %d | 计算能力: %s'%(props.total_memory/1024**3, props.multi_processor_count, props.major+props.minor/10))
torch.backends.cuda.matmul.allow_tf32=True

def bench(dtype,n=8192,iters=20,tag=''):
    try:
        a=torch.randn(n,n,dtype=dtype,device='cuda'); b=torch.randn(n,n,dtype=dtype,device='cuda')
    except Exception as e:
        print('  %s 不支持: %s'%(tag,e)); return
    for _ in range(5): c=a@b
    torch.cuda.synchronize()
    t=time.time()
    for _ in range(iters): c=a@b
    torch.cuda.synchronize()
    dt=(time.time()-t)/iters
    fl=2*n**3
    print('  %-6s %dx%d 矩阵乘: %.1f ms/tile  →  %.1f TFLOPS'%(tag,n,n,dt*1000,fl/dt/1e12))

print('矩阵乘法吞吐:')
bench(torch.float32,tag='FP32')
bench(torch.bfloat16,tag='BF16')
try: bench(torch.float8_e4m3fn,tag='FP8')
except Exception: pass

print('显存带宽:')
for dt,name in [(torch.float32,'FP32')]:
    a=torch.empty(256*1024*1024//4,dtype=dt,device='cuda'); b=torch.empty_like(a)
    for _ in range(3): b.copy_(a)
    torch.cuda.synchronize()
    t=time.time(); n=10
    for _ in range(n): b.copy_(a)
    torch.cuda.synchronize()
    dtm=(time.time()-t)/n
    print('  拷贝带宽: %.1f GB/s (读+写)  [%s]'%(2*a.numel()*4/dtm/1e9,name))
PY"

say "8. LLM 推理实测"
if command -v ollama >/dev/null; then
  run "ollama ps 2>/dev/null; ollama list 2>/dev/null | head -10"
  sub "若已有模型，执行： ollama run <模型> --verbose  观察 tokens/s"
else
  run "echo '未安装 ollama。可执行：curl -fsSL https://ollama.com/install.sh | sh'"
fi

say "9. 汇总"
{
  echo "报告生成时间: $(date -Is)"
  echo "主机: $(hostname)  用户: $(whoami)"
  echo "内核: $(uname -r)"
  echo "CPU: $(lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs)  核心: $(nproc)"
  echo "内存: $(free -h | awk '/^Mem:/{print $2}')  可用: $(free -h | awk '/^Mem:/{print $7}')"
  echo "GPU : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo N/A)"
  echo "磁盘: $(df -h / | awk 'NR==2{print $2" 总 / "$4" 可用"}')"
} | tee -a "$REPORT"

printf '\n\033[1;32m✅ 测评完成，报告: %s\033[0m\n' "$REPORT" | tee -a "$REPORT"
printf '取回本地:  scp -P <端口> %s@<IP>:%s ./   （注意 scp 的端口参数是大写 -P）\n' "$(whoami)" "$REPORT" | tee -a "$REPORT"
