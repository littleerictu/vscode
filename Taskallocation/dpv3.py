import sys
from collections import defaultdict

def main():
    M = 4  # 设备数量
    N = 8  # 任务数量
    T = [3, 2, 4, 1, 5, 6, 2, 3]  # 任务生产时间
    S = [  # 切换时间矩阵 (8x8)
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 2, 1, 3, 4, 5, 6],
        [2, 2, 0, 1, 2, 3, 4, 5],
        [3, 1, 1, 0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 1, 2],
        [6, 5, 4, 3, 2, 1, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0]
    ]
    last_task = [0, 3, 5, 7]  # 各设备前一天的最后一个任务

    # 初始化数据结构
    dp = defaultdict(dict)  # dp[mask][last_tasks_tuple] = (min_max_time, times_tuple)
    initial_last_tasks = tuple(last_task)
    initial_times = tuple([0] * M)  # 初始完成时间均为0
    initial_max = 0
    dp[0][initial_last_tasks] = (initial_max, initial_times)
    parent = {}

    full_mask = (1 << N) - 1
    min_makespan = float('inf')
    best_state = None

    # 遍历所有可能的掩码
    for mask in range(0, 1 << N):
        if mask not in dp:
            continue
        # 遍历该掩码下的所有最后任务组合
        for last_tasks in list(dp[mask].keys()):
            current_max, times = dp[mask][last_tasks]
            if current_max >= min_makespan:
                continue  # 剪枝
            
            # 遍历所有未分配的任务
            for t in range(N):
                if (mask & (1 << t)) != 0:
                    continue
                # 遍历所有设备
                for dev in range(M):
                    prev_task = last_tasks[dev]
                    prev_time = times[dev]
                    # 计算切换时间和新时间
                    if prev_time == 0:  # 当天第一个任务
                        switch = S[prev_task][t]
                        new_time = switch + T[t]
                    else:
                        switch = S[prev_task][t]
                        new_time = prev_time + switch + T[t]
                    
                    # 生成新状态
                    new_last_tasks = list(last_tasks)
                    new_last_tasks[dev] = t
                    new_last_tasks = tuple(new_last_tasks)
                    new_times = list(times)
                    new_times[dev] = new_time
                    new_times = tuple(new_times)
                    new_mask = mask | (1 << t)
                    new_max = max(current_max, new_time)
                    
                    # 更新最优解
                    if new_mask == full_mask and new_max < min_makespan:
                        min_makespan = new_max
                        best_state = (new_mask, new_last_tasks, new_times)
                    
                    # 剪枝：仅保留更优状态
                    key = new_last_tasks
                    if key not in dp[new_mask] or new_max < dp[new_mask].get(key, (float('inf'),))[0]:
                        dp[new_mask][key] = (new_max, new_times)
                        parent[(new_mask, key)] = (mask, last_tasks, t, dev)

    # 回溯路径
    if not best_state:
        print("No feasible solution")
        return

    # 重构任务分配
    allocation = {i: [] for i in range(M)}
    current_state = best_state
    while current_state in parent:
        new_mask, new_last_tasks, new_times = current_state
        mask, last_tasks, t, dev = parent[current_state]
        allocation[dev].append(t)
        # 找到前一个状态
        prev_max, prev_times = dp[mask][last_tasks]
        current_state = (mask, last_tasks, prev_times)

    # 反转任务顺序并输出
    print(f"Minimum Makespan: {min_makespan}")
    for dev in range(M):
        tasks = reversed(allocation[dev])  # 因回溯是逆序
        task_list = list(tasks)
        print(f"Device {dev+1} Tasks: {task_list}")

if __name__ == "__main__":
    main()