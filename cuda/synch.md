# CUDA 中的同步

## `__syncthreads()`

`__syncthreads()` 是一个 **CTA 内的同步原语**。它保证：
- CTA 内所有参与线程都到达；
- barrier 之前的相关内存访问，对 barrier 之后的 CTA 线程可见。

以下写法可能会死锁：

```cpp
if (threadIdx.x % 2 == 0) {
    // 仅偶数线程到达 barrier
    __syncthreads();
}
```

可以在完成 CTA barrier 的同时，执行一次 predicate 聚合：

```cpp
int count = __syncthreads_count(predicate);
int all   = __syncthreads_and(predicate);
int any   = __syncthreads_or(predicate);
```

## `mbarrier`

`mbarrier` 是一个**放在共享内存中的、可重复使用的异步屏障状态机**。

它可以等待：

- 某些线程到达；
- TMA 异步拷贝完成；
- Blackwell `tcgen05.mma/cp/shift` 等异步操作完成；
- CTA cluster 中其他 CTA 发来的单向到达信号。

一个 mbarrier 对象占用共享内存中的 8 字节，要求 8 字节对齐。

mbarrier 初始化：

```cpp
mbarrier.init(barrier, N);
```

意思是：每个 phase 预期收到总计 N 次 arrival。初始化完成后，可以把状态想象为：

```text
phase                  = 0
expected_arrival_count = N
pending_arrival_count  = N
tx_count               = 0
```

单个线程执行一次：

```cpp
mbarrier.arrive(barrier);
```

会将当前 phase 的 pending_arrival_count 减少 1。（如果使用带 `count` 的版本，则一次可以减少多个 arrival count）

当以下两个条件同时成立时：

```text
pending_arrival_count == 0
tx_count              == 0
```

当前 phase 完成。然后硬件原子地：

```text
phase += 1
pending_arrival_count = expected_arrival_count
```

因此，mbarrier 是**自动循环重置的 barrier**，通常不需要在每轮循环重新执行 `mbarrier.init`。

---

### arrive 和 wait 分离

`__syncthreads()` 可以粗略理解为 arrive + wait：每个线程到达之后立即停止，直到整个线程块到齐。

mbarrier 将它拆成：

```cpp
token = barrier.arrive();

// 做不依赖 barrier 完成的工作
independent_work();

barrier.wait(token);
```

线程调用 `arrive` 后不立即阻塞，可以继续执行独立计算，之后再 `wait`。这就是 split arrive/wait barrier，可以用来**隐藏同步延迟**。

另外，mbarrier 不要求整个 CTA 的所有线程参与。例如初始化 arrival count = 32 就可以只让一个 warp 的 32 个线程参与，而其他线程完全不访问这个 barrier。

---

### phase

一个 barrier 通常会被循环重复使用。phase 表明 barrier 当前的轮次。`wait` 必须明确指定等待哪一轮的完成。

```c++
int phase = 0;
for (int i = 0; i < 2; ++i) {
    launch_async_operation(barrier);
    while (!mbarrier_try_wait_parity(barrier, phase)) {
        // 等待本轮异步操作
    }
    phase ^= 1;
}
```

首轮，`wait` 等待 phase 0 的异步操作完成，然后 phase 变为 1，下一轮 `wait` 等待 phase 1 的异步操作完成。

这里 phase 只取 0 或 1. 只能可靠地区分当前 phase 和紧邻的上一 phase。此时程序不能让同一个 barrier 在等待者尚未观察到完成时连续前进两轮，否则会出现 ABA 问题。

另一种方法是使用 `arrive` 返回的 opaque state/token：

```cpp
state = mbarrier.arrive(barrier);

...

mbarrier.try_wait(barrier, state);
```

`state` 捕获了执行 `arrive` 之前 barrier 所处的 phase。其内部位布局由硬件实现决定，程序不能解析，只能把它传给对应的 wait。

---

### mbarrier 等待异步事务

mbarrier 除了 arrival count，还有 `tx-count`，表示当前 phase 尚未完成的异步事务数量

对于 TMA bulk copy，通常以**字节数**为单位。

例如准备异步搬运 4096 字节：

```text
pending arrivals = 1
tx-count         = 4096
```

执行 `arrive.expect_tx` 后：

```text
pending arrivals: 1 → 0
tx-count:         0 → 4096
```

此时 arrival 已经满足，但 barrier 不会完成，因为 TMA 还没有搬完。

TMA 完成时，硬件将这个 mbarrier 上的 `tx-count` 减少 4096：

```text
tx-count: 4096 → 0
```

此时：

```text
pending arrivals == 0
tx-count == 0
```

当前 phase 才真正完成。Hopper 及之后的架构从 `sm_90` 开始支持这种 tx-count 异步事务追踪。

这也是 mbarrier 与普通线程 barrier 最大的区别之一：完成一个 phase 的“参与者”不一定都是 CUDA 线程，也可以是 TMA、Tensor Core 或其他异步硬件单元。

## 弱内存模型与同步语义

### “执行到这里”不等于“此前写入已经可见”

同步需要同时考虑两个问题：

1. **执行位置同步**：其他线程是否已经执行到了某个位置？
2. **内存可见性同步**：其他线程此前产生的内存结果，现在是否保证可以被观察到？

在单线程程序中，通常可以按照源码顺序理解执行过程：

```cpp
data = 42;
ready = true;
```

但 GPU 是弱内存模型。只要不改变单个线程自身可观察到的结果，编译器和硬件可以调整部分操作的执行顺序；一个线程发出的写入也不一定立刻对其他线程、其他 CTA 或异步硬件单元可见。

因此，下面两个事实并不天然等价：

```text
线程 A 已经执行过 data = 42
线程 B 现在保证能够读取到 data == 42
```

程序需要通过具有合适内存语义的同步操作，在两个线程之间建立明确的顺序关系。

这里的“可见”是一种**内存模型保证**。它不应该简单理解成每次都物理清空某级 cache；硬件可以用不同方式实现，只要最终行为满足规定的顺序和可见性即可。

---

### 同步原语的两个维度

一个同步原语可以同时具有两类语义：

```text
控制语义：线程是否需要到达、等待
内存语义：同步点前后的内存访问如何排序、何时可见
```

例如 `__syncthreads()` 同时提供这两类保证：

- CTA 中的参与线程全部到达后才能继续；
- barrier 前的相关内存访问，对 barrier 后的 CTA 线程可见。

但是，split barrier 将控制同步拆成了 `arrive` 和 `wait`，内存语义还可以进一步选择 `release`、`acquire` 或 `relaxed`。

---

### `arrive` 和 `wait` 的控制语义

`arrive` 表示当前参与者已经到达同步点。它通常只更新 barrier 状态，不立即等待其他参与者：

```cpp
barrier.arrive();

// 可以继续执行不依赖 barrier 完成的工作
independent_work();
```

`wait` 则等待规定的参与者全部到达，或者等待当前 mbarrier phase 的 arrival count 和 tx-count 都满足完成条件：

```cpp
barrier.wait();

// barrier 完成后才能执行的工作
dependent_work();
```

只讨论控制流时，可以把它们理解为：

```text
arrive：我已经到达
wait：等所有人都到达
```

但这句话尚未说明 barrier 前后的内存访问如何排序。这个问题由 release/acquire 等内存语义解决。

---

### `release`：发布此前的结果

release 约束的是 release 操作**之前**的内存访问：

```cpp
shared_data = 42;
barrier.arrive.release();
```

它保证从内存模型上看，前面的相关访问不能被推迟到 release 之后：

```text
写入 shared_data
        ↓ 必须先于
arrive.release
```

如果另一个参与者之后通过匹配的 acquire 完成同步，那么它可以观察到该 release 之前发布的结果。

`arrive.release` 同时做了两件事：

1. 以 release 语义发布此前的结果；
2. 向 barrier 报告当前参与者已经到达。

---

### `acquire`：接收已经发布的结果

acquire 约束的是 acquire 操作**之后**的内存访问：

```cpp
barrier.wait.acquire();
int value = shared_data;
```

它保证后面的相关访问不能在内存语义上越过 acquire 提前发生：

```text
wait.acquire
        ↓ 必须先于
读取 shared_data
```

当 acquire 与另一个参与者的 release 成功同步后，完整关系是：

```text
生产者                                      消费者

shared_data = 42
        │
        ▼
arrive.release  ───── synchronizes-with ───► wait.acquire
                                                    │
                                                    ▼
                                           value = shared_data
```

于是：生产者在 release 前的写入 happens before 消费者在 acquire 后的读取

---

### 同步作用域

release/acquire 还必须考虑作用域。常见作用域包括：

```text
.cta      当前 CTA
.cluster  当前 CTA cluster
.gpu      当前 GPU
.sys      整个系统
```

例如：

```text
release.cluster
```

只承诺把结果发布给当前 cluster 范围内的参与者，不能用它与另一个 cluster 或 CPU 线程建立同步关系。

release 和 acquire 的作用域必须足以覆盖通信双方。作用域过小，即使两边分别写了 release 和 acquire，也不能建立所需的跨线程同步关系。

---

### `relaxed`：保留操作本身，不提供周围内存排序

relaxed 并不表示“这条指令什么都不做”。它表示：

> 操作自身的状态变化或控制同步仍然发生，但不借助这条操作为前后的普通内存访问提供额外的顺序和可见性保证。

例如：

```cpp
shared_data = 42;
barrier.cluster.arrive.relaxed;
```

这个 arrive 仍然会：

- 报告当前参与者已经到达；
- 更新 cluster barrier 的到达状态；
- 最终帮助其他参与者的 `wait` 满足控制流完成条件。

但它不负责发布前面的 `shared_data = 42`。因此可以简化理解为：

```text
arrive.relaxed：我执行到这里了
arrive.release：我执行到这里了，而且此前的结果已经发布
```

如果程序只依赖 barrier 的控制流效果，或者此前的结果已经由其他 release/fence 发布，就可以使用 relaxed 操作，避免重复施加内存排序约束。

反过来，如果后续线程需要读取 barrier 前写入的数据，却没有其他发布机制，就不能只依赖 relaxed arrive。

---

### `barrier.cluster` 中 arrive/wait 与 release/acquire 的组合

cluster barrier 的典型形式是：

```cpp
barrier.cluster.arrive.release;
barrier.cluster.wait.acquire;
```

其执行过程是：

1. 每个参与者在 `arrive.release` 前完成并发布自己的相关内存操作；
2. `arrive` 报告当前参与者已到达，但不等待其他 CTA；
3. `wait` 等待 cluster 中所有参与者完成 arrive；
4. `wait.acquire` 接收各参与者在 release 前发布的结果；
5. wait 之后的代码可以安全使用这些结果。

当前教程中的 helper 是：

```cpp
asm volatile("barrier.cluster.arrive.aligned;\n"
             "barrier.cluster.wait.aligned;\n" ::: "memory");
```

这里没有显式写 `.release` 和 `.acquire`，但 `barrier.cluster` 的默认语义分别是：

```cpp
barrier.cluster.arrive.release.aligned;
barrier.cluster.wait.acquire.aligned;
```

`.aligned` 与内存序无关。它要求同一 warp 中的非退出线程以一致方式执行这条 barrier 指令，不能把它理解成更强的 acquire/release。
