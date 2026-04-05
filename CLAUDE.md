# Cliriux Agent — 项目说明

## Python 环境

使用 conda `agent` 环境。运行任何 Python 命令前先激活：

```bash
conda run -n agent <命令>
# 或
conda activate agent && <命令>
```

安装新依赖：
```bash
conda run -n agent pip install -r requirements.txt
```

## Git Worktrees

worktree 目录：`.worktrees/`（已在 .gitignore 中忽略）

## 测试

```bash
conda run -n agent python -m pytest tests/ -v
```
