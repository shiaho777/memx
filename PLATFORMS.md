# MemX 跨平台适配状态

核心引擎（`core/`）OS 无关：freestanding 编译、仅依赖 memcpy 族、页大小 4K/16K 可配、NEON 带标量回退。每个平台一个薄适配层，实现 EMBEDDING.md 契约（缺页投递 + 保护握手 + 压缩提交）。

## 状态矩阵（诚实标注验证程度）

| 平台 | 适配层 | 缺页机制 | 验证状态 |
|---|---|---|---|
| macOS (Apple Silicon) | `libmemx3.m` 全功能运行时 | sigaction + Mach 异常端口接管 | ✅ 完整门禁本地全绿（压缩/解压/capsule/store/真实生成） |
| macOS (POSIX 参考) | `platform/posix/` | sigaction + mprotect | ✅ `make posix-adapter` 16K/4K 本地通过 |
| Linux x86_64 | `platform/posix/` | sigaction + mprotect | ✅ CI `linux-core` 4K 页端到端通过 |
| Linux arm64 | `platform/posix/` | 同上 | ⚠️ 代码同 POSIX，未单独跑 arm64 Linux CI |
| Windows x64 | `platform/windows/`（新增） | VEH + VirtualProtect | ⚠️ 本地 mingw 交叉编译验证 + CI `windows-core` 真机运行（见下） |
| iOS arm64 | `platform/posix/` 复用 | sigaction + mprotect + 异常端口接管 | ⚠️ Xcode iPhoneOS SDK 交叉编译+链接通过；**真机运行未验证**（需 app 宿主） |
| Android arm64 | `platform/posix/` 复用 | sigaction + mprotect（bionic） | ⚠️ 构建脚本就绪（`platform/android/build_android.sh`，需 NDK）；**真机运行未验证** |

## 构建入口

```sh
make posix-adapter      # macOS/Linux 参考适配层（16K + 4K）
make ios-build          # iOS arm64 交叉编译（需 Xcode iPhoneOS SDK）
make android-build      # Android arm64（需 ANDROID_NDK 环境变量）
make windows-adapter    # Windows x64（需 x86_64-w64-mingw32-gcc，或 CI）
```

## 各平台要点

### Windows（`platform/windows/memx_windows.c`）
- 缺页投递：`AddVectoredExceptionHandler` 捕获 `EXCEPTION_ACCESS_VIOLATION`，`ExceptionInformation[0]` 区分读/写，`[1]` 是故障地址。
- 保护握手：`VirtualProtect`（PAGE_NOACCESS/READONLY/READWRITE），与 POSIX mprotect 同序。
- 区域：`VirtualAlloc(MEM_RESERVE|MEM_COMMIT, PAGE_NOACCESS)`。
- 线程：`CreateThread` + `CRITICAL_SECTION`（同线程可重入，VEH/压缩器交互安全）。
- 原子：`_InterlockedCompareExchange8`（state 为 uint8_t）。
- 硬件页 4K，16K 逻辑页是其整数倍，粒度兼容；init 返回 -2 的契约与 POSIX 一致。
- **验证**：`.github/workflows/windows.yml` 在 windows-latest（MINGW64 gcc）编译并运行 `test_windows_adapter`（48 页两代写入-压缩-缺页读回 bitexact）。

### iOS
- 直接复用 POSIX 适配层；`__APPLE__` 分支的 Mach 异常端口接管在 iOS 同样适用。
- 16K 逻辑页 ≥ iOS 硬件页（16K），粒度兼容。
- **限制**：iOS 不允许裸可执行文件启动，测试需嵌入 app bundle 在真机跑；动态库需签名。编译链接已验证，运行待真机。
- Metal 辅助压缩（macOS 运行时特性）未移植；iOS 走纯 CPU 编解码。

### Android
- 复用 POSIX 适配层；bionic 提供 sigaction/mprotect/pthread。
- 无 `__APPLE__` 分支，纯 POSIX 路径。
- 16K 逻辑页兼容 4K/16K 硬件页设备。
- **限制**：需 NDK 交叉编译 + adb 推送到真机运行；各厂商 ROM 的信号投递行为需实测。
- 推理引擎对接（llama.cpp/MNN/ExecuTorch 权重层）是后续工程，不在适配层范围。

## 未移植的能力（诚实边界）

- **capsule 快照存储 / memx_stored 服务**：目前仅 macOS 全功能运行时；POSIX/Windows/iOS/Android 适配层是最小参考实现（单区域、bump arena、无 pool 回收、无 capsule）。
- **Metal 辅助压缩**：macOS 专属。
- **Python 绑定 / 真实 LLM 生成**：依赖 macOS 全功能运行时 + torch，未跨平台。

跨平台的是**压缩内存内核 + 缺页解压契约**；上层产品能力（capsule/store/推理）的跨平台是后续工作。