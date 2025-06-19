import os
import signal
import subprocess


def kill_process_on_port(port):
    """查找并杀死占用指定端口的进程"""
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True)
        pids = output.decode().strip().split('\n')
        for pid in pids:
            print(f"🔪 杀死占用端口 {port} 的进程 PID: {pid}")
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        print(f"✅ 端口 {port} 没有被占用，无需处理。")
