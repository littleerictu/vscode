import sys
import time
import copy
from heapq import heappush, heappop
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def format_duration(seconds):
    """格式化时间显示：秒 → 可读格式"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {seconds%60:.1f}s"
    else:
        return f"{int(seconds//3600)}h {int(seconds%3600//60)}m {seconds%60:.1f}s"

class LocalScheduler:
    """本地调度器（子进程实例）"""
    def __init__(self, params):
        # 初始化参数
        self.M = params['M']           # 设备数量
        self.N = params['N']           # 任务数量
        self.T = params['T']           # 生产时间列表（索引对应内部任务编号）
        self.S = params['S']           # 切换时间矩阵[prev][current]
        self.prune_k = params['prune_k'] # 剪枝保留状态数
        self.prev_tasks = params['prev_tasks']  # 前日任务实际编号列表
        self.day_tasks = params['day_tasks']    # 当日任务实际编号列表
        
        # 状态编码参数
        self.TASK_BITS = 5             # 每个任务编号占用位数
        self.DEVICE_MASK = (1 << self.TASK_BITS) - 1
        
        # 状态存储
        self.dp = defaultdict(dict)    # 状态记录器：dp[mask][encoded_state] = time
        self.parent = {}               # 父状态记录
        self.min_makespan = float('inf') # 当前最优解
        self.state_count = 0           # 状态计数器
        self.start_time = time.time()  # 计时器

    def safe_encode(self, tasks):
        """安全编码状态（防止越界）"""
        max_task = len(self.S) - 1
        return [min(t, max_task) for t in tasks]

    def encode_state(self, tasks):
        """将设备最后任务列表编码为整数"""
        validated = self.safe_encode(tasks)
        encoded = 0
        for task in reversed(validated):
            encoded = (encoded << self.TASK_BITS) | task
        return encoded

    def decode_state(self, encoded):
        """从整数解码设备最后任务列表"""
        tasks = []
        for _ in range(self.M):
            tasks.append(encoded & self.DEVICE_MASK)
            encoded >>= self.TASK_BITS
        return tasks

    def process_range(self, start_mask, end_mask):
        """处理指定掩码范围的状态转移"""
        pq = []  # 优先队列（时间, 掩码, 编码状态）
        initial_encoded = self.encode_state(self.prev_tasks)
        heappush(pq, (0, 0, initial_encoded))
        self.parent[(0, initial_encoded)] = None
        
        # 进度跟踪
        last_print = time.time()
        
        while pq:
            current_max, mask, encoded_last = heappop(pq)
            self.state_count += 1
            
            # 打印进度（每5秒）
            if time.time() - last_print > 5:
                elapsed = time.time() - self.start_time
                print(f"  ▶ 已处理状态: {self.state_count} | 当前最优: {self.min_makespan} | 用时: {format_duration(elapsed)}")
                last_print = time.time()
            
            # 剪枝条件：超过当前最优解或时间阈值
            if current_max >= self.min_makespan or current_max > 50:
                continue
                
            # 找到完整解时更新最优
            if mask == (1 << self.N) - 1:
                if current_max < self.min_makespan:
                    self.min_makespan = current_max
                continue
                
            # 状态转移处理
            last_tasks = self.decode_state(encoded_last)
            for t_idx in range(self.N):  # t_idx是当日任务的内部索引
                if (mask & (1 << t_idx)) != 0:
                    continue  # 任务已分配
                
                actual_task = self.day_tasks[t_idx]  # 实际任务编号
                if actual_task >= len(self.S) or actual_task < 0:
                    raise ValueError(f"非法任务编号: {actual_task}")
                
                for dev in range(self.M):
                    # 获取前一个实际任务编号
                    prev_actual = last_tasks[dev]
                    if prev_actual >= len(self.S):
                        prev_actual = len(self.S) - 1
                    
                    # 计算切换时间和新时间
                    setup = self.S[prev_actual][actual_task]
                    prod = self.T[t_idx]
                    new_time = current_max + setup + prod
                    
                    # 生成新状态
                    new_last = last_tasks.copy()
                    new_last[dev] = actual_task
                    new_encoded = self.encode_state(new_last)
                    new_mask = mask | (1 << t_idx)
                    
                    # 状态剪枝逻辑
                    if (new_encoded not in self.dp[new_mask] or 
                        new_time < self.dp[new_mask].get(new_encoded, float('inf'))):
                        
                        # 记录父状态
                        self.parent[(new_mask, new_encoded)] = (mask, encoded_last, dev, t_idx)
                        
                        # 维护每个mask最多保留prune_k个状态
                        if len(self.dp[new_mask]) < self.prune_k:
                            self.dp[new_mask][new_encoded] = new_time
                            heappush(pq, (new_time, new_mask, new_encoded))
                        else:
                            # 找到需要替换的最大时间状态
                            max_time = max(self.dp[new_mask].values())
                            candidates = [k for k,v in self.dp[new_mask].items() if v == max_time]
                            if candidates and new_time < max_time:
                                del self.dp[new_mask][candidates[0]]
                                self.dp[new_mask][new_encoded] = new_time
                                heappush(pq, (new_time, new_mask, new_encoded))
            
            # 内存优化：定期清理过期状态
            if mask % 1000 == 0:
                self.dp = defaultdict(dict, {k:v for k,v in self.dp.items() if k >= mask})
        
        return (self.min_makespan, self.parent, self.state_count)

class GlobalScheduler:
    """全局调度控制器"""
    def __init__(self, M, N, T, S, prev_tasks, day_tasks, prune_k=20, num_workers=4):
        self.M = M                      # 设备数量
        self.N = N                      # 任务数量
        self.T = copy.deepcopy(T)       # 生产时间列表
        self.S = copy.deepcopy(S)       # 切换时间矩阵
        self.prev_tasks = copy.deepcopy(prev_tasks)  # 前日任务
        self.day_tasks = day_tasks      # 当日任务实际编号列表
        self.prune_k = prune_k          # 剪枝参数
        self.num_workers = num_workers  # 并行工作进程数
        self.global_best = float('inf') # 全局最优解
        self.best_schedule = None       # 最优调度方案
        self.total_states = 0           # 总处理状态数

    def trace_schedule(self, parent, final_state):
        """回溯任务分配方案"""
        schedule = {i: [] for i in range(self.M)}
        current = final_state
        
        while current in parent and parent[current] is not None:
            mask, encoded_last, dev, t_idx = parent[current]
            actual_task = self.day_tasks[t_idx]
            schedule[dev].append(actual_task)
            current = (mask, encoded_last)
        
        # 反转顺序并过滤空设备
        for dev in schedule:
            schedule[dev].reverse()
        return {k: v for k, v in schedule.items() if v}

    def run(self):
        """执行并行调度计算"""
        total_masks = 1 << self.N
        chunk_size = total_masks // self.num_workers
        
        print("="*40)
        print(f"任务调度参数".center(40))
        print(f"设备数量: {self.M}")
        print(f"任务数量: {self.N}")
        print(f"剪枝参数: 保留每个mask前{self.prune_k}个状态")
        print(f"并行工作进程: {self.num_workers}")
        print("="*40 + "\n")

        # 准备并行参数
        params = {
            'M': self.M,
            'N': self.N,
            'T': copy.deepcopy(self.T),
            'S': copy.deepcopy(self.S),
            'prev_tasks': copy.deepcopy(self.prev_tasks),
            'day_tasks': self.day_tasks,
            'prune_k': self.prune_k
        }

        # 启动并行计算
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                start = i * chunk_size
                end = (i+1)*chunk_size if i != self.num_workers-1 else total_masks
                future = executor.submit(
                    self.worker_wrapper,
                    start, end, params
                )
                futures.append(future)
            
            # 收集结果
            self.total_states = 0
            for future in futures:
                local_best, local_parent, state_count = future.result()
                self.total_states += state_count
                if local_best < self.global_best:
                    self.global_best = local_best
                    # 回溯最优方案
                    final_mask = (1 << self.N) - 1
                    final_encoded = LocalScheduler(params).encode_state(self.prev_tasks)
                    self.best_schedule = self.trace_schedule(local_parent, (final_mask, final_encoded))

        return self.global_best

    def worker_wrapper(self, start, end, params):
        """工作进程包装函数"""
        scheduler = LocalScheduler(params)
        return scheduler.process_range(start, end)

def main():
    """验证通过的测试用例"""
    # ================= 参数配置 =================
    M = 4                   # 设备数量
    N = 8                   # 当日任务数量
    prev_tasks = [5, 6, 7, 8]  # 前日任务在当日任务范围内
    day_tasks = [5, 6, 7, 8, 9, 10, 11, 12]  # 当日任务实际编号
    T = [3, 2, 4, 1, 5, 6, 2, 3]  # 生产时间（索引0对应任务5）
    
    # 生成切换时间矩阵（示例：统一切换时间为1）
    S = [[1 if i != j else 0 for j in range(13)] for i in range(13)]

    # ================= 执行计算 =================
    print("🚀 启动任务调度优化...")
    start_time = time.time()
    
    scheduler = GlobalScheduler(
        M=M, N=N, T=T, S=S,
        prev_tasks=prev_tasks,
        day_tasks=day_tasks,
        prune_k=20,  # 增大剪枝参数
        num_workers=4
    )
    makespan = scheduler.run()
    
    # ================= 结果输出 =================
    print("\n" + "="*40)
    print(f"计算完成！总用时: {format_duration(time.time()-start_time)}")
    print(f"实际处理状态数: {scheduler.total_states:,}")
    print(f"最优完成时间: {makespan if makespan != float('inf') else '无可行解'}")
    
    if scheduler.best_schedule:
        print("\n各设备任务分配详情:")
        for dev in range(M):
            tasks = scheduler.best_schedule.get(dev, [])
            if not tasks:
                print(f"设备 {dev+1}: 未分配任务")
                continue
            
            total_time = 0
            details = []
            # 处理第一个任务
            prev = prev_tasks[dev]
            current = tasks[0]
            setup = S[prev][current]
            prod = T[day_tasks.index(current)]
            total_time += setup + prod
            details.append(f"{prev}→{current}(切{setup}+产{prod})")
            
            # 后续任务
            for i in range(1, len(tasks)):
                prev_task = tasks[i-1]
                current_task = tasks[i]
                setup = S[prev_task][current_task]
                prod = T[day_tasks.index(current_task)]
                total_time += setup + prod
                details.append(f"{prev_task}→{current_task}(切{setup}+产{prod})")
            
            print(f"设备 {dev+1}:")
            print(f"  任务顺序: {tasks}")
            print(f"  总耗时: {total_time}")
            print(f"  时间明细: {' + '.join(details)}")
    else:
        print("\n⚠️ 未找到可行调度方案")

if __name__ == "__main__":
    main()