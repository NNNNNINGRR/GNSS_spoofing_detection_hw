# -*- coding: utf-8 -*-
"""云服务器 SSH 辅助工具（密码认证）。

用法（需先设置环境变量）：
  $env:CLOUD_HOST='connect.westb.seetacloud.com'
  $env:CLOUD_PORT='23811'
  $env:CLOUD_USER='root'
  $env:CLOUD_SSH_PASS='<密码>'

  python cloud_helper.py run "ls /root/autodl-tmp"
  python cloud_helper.py put <本地文件> <远端路径>
  python cloud_helper.py mkdir /root/autodl-tmp/exp_gnss/results
"""
import os
import sys
import paramiko

HOST = os.environ["CLOUD_HOST"]
PORT = int(os.environ.get("CLOUD_PORT", "22"))
USER = os.environ["CLOUD_USER"]
PWD = os.environ["CLOUD_SSH_PASS"]


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    return c


def run(cmd):
    c = client()
    _, out, err = c.exec_command(cmd, timeout=1800)
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("=== STDERR ===")
        print(e)
    code = out.channel.recv_exit_status()
    c.close()
    sys.exit(code)


def put(local, remote):
    c = client()
    sftp = c.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    c.close()
    print(f"uploaded {local} -> {remote}")


def get(remote, local):
    c = client()
    sftp = c.open_sftp()
    sftp.get(remote, local)
    sftp.close()
    c.close()
    print(f"downloaded {remote} -> {local}")


def mkdir(path):
    run(f"mkdir -p {path}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2])
    elif cmd == "put":
        put(sys.argv[2], sys.argv[3])
    elif cmd == "get":
        get(sys.argv[2], sys.argv[3])
    elif cmd == "mkdir":
        mkdir(sys.argv[2])
    else:
        raise SystemExit("unknown command")
