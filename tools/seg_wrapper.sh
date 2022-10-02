#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"

ulimit -c unlimited
"$@"
if [[ $? -eq 139 ]]; then
    coredumpctl gdb -1 -A "--batch -x $SCRIPTPATH/backtrace"
fi
