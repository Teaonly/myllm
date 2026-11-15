#!/bin/sh
# feed.sh —— ndjson 喂食器：把文件逐行按节奏送上 stdout，收尾补 ExitMsg。
#
# 用场：backend 的 stdin 是 ndjson 协议信道；冒烟/调试要用一个文件按
# 时间表喂它——头延时等剧本装载完（~30s，发早了行会攒在管道里，节拍
# 全被搅进装载期），行间隔模拟前端节奏，尾延时留出 beat 拍窗口，最后
# ExitMsg 让后端走设计内干净退出（exit 1）。
#
# 用法：feed.sh <ndjson文件> [头延时s] [行间隔s] [尾延时s]
#   延时支持小数（GNU sleep）；缺省 头=30 行间=0.2 尾=5
#
# 取舍：
#   - 纯 sh + sleep：为个喂食器引入语言不值；节奏精度 0.1s 级足够
#   - 空行跳过：空行到后端是坏行，会按 front parse error 收场
#   - 无尾换行的末行照样送出（|| [ -n "$line" ] 兜住 read 的尾巴——
#     single.ndjson 栽过的坑，这里不再栽）
#   - 不回显已发行：后端 LOG 自会报到，重复回显徒增噪声
#   - 后端先退（文件自带 ExitMsg 等）时，后续写吃 SIGPIPE 静默死；
#     Makefile 管道看的退出码取自后端，无害

set -eu

usage() { echo "usage: $0 <ndjson文件> [头延时s] [行间隔s] [尾延时s]" >&2; exit 2; }

[ $# -ge 1 ] && [ $# -le 4 ] || usage
FILE=$1; HEAD=${2:-30}; GAP=${3:-0.2}; TAIL=${4:-5}

[ -f "$FILE" ] || { echo "$0: no such file: $FILE" >&2; exit 2; }

sleep "$HEAD"

FIRST=1
while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    if [ "$FIRST" -eq 1 ]; then FIRST=0; else sleep "$GAP"; fi
    printf '%s\n' "$line"
done < "$FILE"

sleep "$TAIL"
printf '{"__type__":"ExitMsg"}\n'
