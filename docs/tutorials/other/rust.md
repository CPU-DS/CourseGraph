# Rust 扩展

<ArticleMetadata/>

本项目使用 Rust 编写了部分 Python 函数, 目的是加速这些函数的执行速度。例如目标检测的后处理上。实际上因为大量调用大模型, 所以 Rust 带来的速度提升并不明显, 这部分只是为了体验混合编程。

> [!NOTE]
> 这部分扩展并不提供给用户, 只在项目内部进行调用。如果你不打算继续开发, 请忽略本章内容。

## 继续编写 Rust 扩展

### 编写 Rust 代码

Rust 扩展代码都应该放到 `src/lib.rs` 目录下, 具体实现可参考 [PyO3 指南](https://pyo3.rs/v0.15.1/)。

### 导出函数

在 `src/lib.rs` 中的 `_core` 函数中添加导出函数, 具体导出方式可参考已导出的函数部分。

### 编写函数接口

为了使得 IDE 获得更好的提示, 我们可以为这些函数编写 Python 接口, 但不用编写具体的实现。

在 `src/course_graph/_core.pyi` 文件中继续添加函数接口, 包含类型标注和函数注解等信息即可。

### 编译并安装

确保已安装 Rust 环境、Cargo, 然后执行：

```bash
source .venv/bin/activate
maturin develop --uv
```

所有编写的 Rust 扩展函数会安装到 `course_graph._core` 包中。