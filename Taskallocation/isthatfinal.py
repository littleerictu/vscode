import sys
import time
from collections import defaultdict

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {seconds%60:.1f}s"
    else:
        return f"{int(seconds//3600)}h {int(seconds%3600//60)}m {seconds%60:.1f}s"

class Scheduler:
    def __init__(self, M, N, T, S, last_task, task_list):
        self.M = M          # 设备数量
        self.N = N          # 任务数量
        self.T = T          # 生产时间列表（索引对应内部任务编号）
        self.S = S          # 切换时间矩阵（实际任务编号）
        self.last_task = last_task  # 各设备前日最后任务（实际编号）
        self.task_list = task_list  # 当日任务实际编号列表
        self.task_map = {task: idx for idx, task in enumerate(task_list)}  # 任务到内部编号映射
        
        # 状态空间
        self.dp = defaultdict(dict)
        self.parent = {}
        self.min_makespan = float('inf')
        self.best_state = None
        
        # 进度跟踪
        self.start_time = time.time()
        self.last_print = 0
        self.print_interval = 5

    def initialize(self):
        """初始化状态"""
        initial_last = tuple(self.last_task)
        initial_times = tuple([-1] * self.M)  # -1表示尚未开始当天任务
        self.dp[0][initial_last] = (0, initial_times)

    def run(self):
        """执行动态规划主循环"""
        total_masks = 1 << self.N
        print(f"开始调度计算，总状态数: {total_masks:,}")
        
        for mask in range(total_masks):
            self.print_progress(mask, total_masks)
            
            if mask not in self.dp:
                continue
                
            for last_tasks in list(self.dp[mask].keys()):
                current_max, times = self.dp[mask][last_tasks]
                
                # 剪枝：如果当前最大值已超过最优解
                if current_max >= self.min_makespan:
                    continue
                
                # 遍历所有任务和设备
                for t in range(self.N):  # t为内部任务编号
                    if (mask & (1 << t)) != 0:
                        continue
                        
                    for dev in range(self.M):
                        self.process_transition(mask, last_tasks, times, t, dev)
                        
        return self.min_makespan

    def process_transition(self, mask, last_tasks, times, t, dev):
        """处理状态转移"""
        actual_task = self.task_list[t]  # 转换为实际任务编号
        prev_actual_task = last_tasks[dev]  # 前一个实际任务编号
        
        prev_time_state = times[dev]
        
        # 计算切换时间（关键修改点）
        if prev_time_state == -1:  # 当天首次任务
            switch = self.S[prev_actual_task][actual_task]
            new_time = switch + self.T[t]
            new_last = actual_task
        else:  # 已有当天任务
            switch = self.S[last_tasks[dev]][actual_task]
            new_time = prev_time_state + switch + self.T[t]
            new_last = actual_task
        
        new_last_list = list(last_tasks)
        new_last_list[dev] = new_last
        new_last_tuple = tuple(new_last_list)
        
        new_times = list(times)
        new_times[dev] = new_time
        new_times_tuple = tuple(new_times)
        
        new_mask = mask | (1 << t)
        new_max = max(new_time, self.dp[mask][last_tasks][0])
        
        # 更新最优解
        if new_mask == (1 << self.N) - 1 and new_max < self.min_makespan:
            self.min_makespan = new_max
            self.best_state = (new_mask, new_last_tuple, new_times_tuple)
        
        # 状态剪枝
        if new_last_tuple not in self.dp[new_mask] or new_max < self.dp[new_mask].get(new_last_tuple, (float('inf'),))[0]:
            self.dp[new_mask][new_last_tuple] = (new_max, new_times_tuple)
            self.parent[(new_mask, new_last_tuple)] = (mask, last_tasks, t, dev)

    def print_progress(self, mask, total):
        """打印进度信息"""
        current_time = time.time()
        if current_time - self.last_print < self.print_interval:
            return
        
        elapsed = current_time - self.start_time
        progress = mask / total * 100
        remaining = elapsed / (mask+1) * (total - mask) if mask > 0 else 0
        
        status = [
            f"进度: {progress:.1f}%",
            f"用时: {format_time(elapsed)}",
            f"剩余: {format_time(remaining)}",
            f"当前最优: {self.min_makespan if self.min_makespan < float('inf') else 'N/A'}"
        ]
        print("\r" + " | ".join(status), end="")
        self.last_print = current_time

    def get_schedule(self):
        """获取调度方案（返回内部任务编号）"""
        if not self.best_state:
            return None
            
        allocation = {i: [] for i in range(self.M)}
        current_state = self.best_state
        
        while current_state[0] > 0:
            mask, last_tasks, times = current_state
            if (mask, last_tasks) not in self.parent:
                break
                
            prev_mask, prev_last, t, dev = self.parent[(mask, last_tasks)]
            allocation[dev].append(t)  # 存储内部任务编号
            current_state = (prev_mask, prev_last, self.dp[prev_mask][prev_last][1])
            
        # 反转任务顺序
        for dev in allocation:
            allocation[dev].reverse()
            
        return allocation

def main():
    # ================= 测试数据配置 =================
    M = 5                   # 设备数量
    prev_tasks = [0, 1, 2, 3, 4]  # 各设备前日最后任务（实际编号）
    day_tasks = list(range(5, 21))  # 当日任务实际编号列表（5-20）
    N = len(day_tasks)      # 任务数量
    
    # 生产时间配置（索引对应内部任务编号）
    T = [3, 2, 4, 1, 5, 6, 2, 3, 4, 2, 
         5, 1, 3, 4, 2, 5]  # 任务5-20的生产时间
    
    # 生成切换时间矩阵（21x21，实际任务编号0-20）
    S = [[abs(i-j) for j in range(21)] for i in range(21)]  # 示例矩阵
    
    # ================= 初始化调度器 =================
    scheduler = Scheduler(
        M=M,
        N=N,
        T=T,
        S=S,
        last_task=prev_tasks,
        task_list=day_tasks
    )
    scheduler.initialize()
    
    # ================= 执行计算 =================
    print("正在计算最优调度方案...")
    makespan = scheduler.run()
    
    # ================= 结果输出 =================
    print("\n\n最终结果：")
    if makespan == float('inf'):
        print("无可行解")
        return
    
    # 获取分配方案（内部任务编号）
    allocation = scheduler.get_schedule()
    
    # 打印全局信息
    print(f"全局最大完成时间（Makespan）: {makespan}")
    
    # 打印各设备详细信息
    for dev in range(M):
        tasks = allocation.get(dev, [])
        actual_tasks = [day_tasks[t] for t in tasks]  # 转换为实际任务编号
        
        # 时间计算
        total_time = 0
        time_breakdown = []
        if tasks:
            # 处理第一个任务
            prev_actual = prev_tasks[dev]
            current_actual = day_tasks[tasks[0]]
            switch = S[prev_actual][current_actual]
            total_time += switch + T[tasks[0]]
            time_breakdown.append(
                f"前日{prev_actual}→{current_actual}"
                f"（切换{switch}+生产{T[tasks[0]]}）"
            )
            
            # 处理后续任务
            for i in range(1, len(tasks)):
                prev_actual = day_tasks[tasks[i-1]]
                current_actual = day_tasks[tasks[i]]
                switch = S[prev_actual][current_actual]
                total_time += switch + T[tasks[i]]
                time_breakdown.append(
                    f"{prev_actual}→{current_actual}"
                    f"（切换{switch}+生产{T[tasks[i]]}）"
                )
        
        print(f"\n设备 {dev+1}:")
        print(f"任务序列: {actual_tasks}")
        print(f"总耗时: {total_time}")
        if tasks:
            print("时间明细:")
            print(" + ".join(time_breakdown))
            print(f" = {total_time}")

if __name__ == "__main__":
    main()