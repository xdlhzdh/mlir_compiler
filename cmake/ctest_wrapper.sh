#!/bin/sh
# 兼容两种用途：
# 1) 修补 Makefile 的 test 目标：ctest_wrapper.sh --patch <makefile> <wrapper_path>
# 2) 运行测试（make test DOMAIN=ast|pass）：从环境变量 DOMAIN 读并按域调用 ctest

if [ "${1:-}" = "--patch" ]; then
  MAKEFILE="${2:-}"
  WRAPPER="${3:-$0}"
  [ -n "${MAKEFILE}" ] || exit 1
  grep -q "ctest_wrapper.sh" "${MAKEFILE}" 2>/dev/null && exit 0
  # 单引号内 \\$(DOMAIN)/\\$(ARGS) 为字面量，供 Make 展开
  sed -i 's|/usr/bin/ctest --force-new-ctest-process \$(ARGS)|env DOMAIN=$(DOMAIN) sh '"${WRAPPER}"' --force-new-ctest-process \$(ARGS)|' "${MAKEFILE}"
  exit 0
fi

DOMAIN="${DOMAIN:-}"
case "$DOMAIN" in
  ast) exec ctest -R "^V[1-4]_" --output-on-failure "$@";;
  pass) exec ctest -L pass --output-on-failure "$@";;
  "") exec ctest --output-on-failure "$@";;
  *) echo "Unsupported DOMAIN=$DOMAIN; use ast or pass" >&2; exit 1;;
esac
