import sys
from collections import defaultdict

def main():
    '''
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
    last_task = [0, 1, 2, 3]  # 各设备前一天的最后一个任务
    '''
    
    import sys
from collections import defaultdict

def main():
    # 修改后的测试数据
    M = 5  # 设备数量
    N = 16  # 任务数量（编号5-20）
    T = [3, 2, 4, 1, 5, 6, 2, 3, 4, 2, 5, 1, 3, 4, 2, 5]  # 任务5-20的生产时间
    S = [  # 21x21切换时间矩阵（任务0-20）
        # 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20
        [0,  2,  3,  1,  4,  2,  3,  1,  5,  2,  4,  3,  1,  2,  5,  3,  2,  4,  1,  3,  2],  # 0
        [2,  0,  1,  3,  2,  4,  1,  2,  3,  5,  2,  1,  4,  3,  2,  1,  5,  3,  2,  4,  1],  # 1
        [3,  1,  0,  2,  1,  3,  2,  4,  1,  2,  3,  5,  2,  1,  4,  2,  3,  1,  5,  2,  3],  # 2
        [1,  3,  2,  0,  3,  1,  4,  2,  3,  1,  5,  2,  3,  4,  1,  2,  3,  5,  1,  2,  4],  # 3
        [4,  2,  1,  3,  0,  2,  1,  5,  2,  3,  1,  4,  2,  5,  3,  1,  2,  4,  3,  1,  5],  # 4
        [2,  4,  3,  1,  2,  0,  3,  1,  4,  2,  5,  3,  1,  2,  4,  5,  1,  2,  3,  4,  1],  # 5
        [3,  1,  2,  4,  1,  3,  0,  2,  1,  5,  2,  3,  4,  1,  2,  3,  5,  1,  2,  3,  4],  # 6
        [1,  2,  4,  2,  5,  1,  2,  0,  3,  1,  4,  2,  5,  3,  1,  2,  4,  5,  1,  2,  3],  # 7
        [5,  3,  1,  3,  2,  4,  1,  3,  0,  2,  1,  5,  3,  2,  4,  1,  2,  3,  5,  1,  2],  # 8
        [2,  5,  2,  1,  3,  2,  5,  1,  2,  0,  3,  1,  4,  2,  5,  3,  1,  2,  4,  5,  1],  # 9
        [4,  2,  3,  5,  1,  5,  2,  4,  1,  3,  0,  2,  1,  5,  2,  4,  3,  1,  2,  3,  5],  # 10
        [3,  1,  5,  2,  4,  3,  3,  2,  5,  1,  2,  0,  3,  1,  5,  2,  3,  4,  1,  2,  3],  # 11
        [1,  4,  2,  3,  2,  1,  4,  5,  3,  4,  1,  3,  0,  2,  1,  5,  2,  3,  4,  1,  2],  # 12
        [2,  3,  1,  4,  5,  2,  1,  3,  2,  2,  5,  1,  2,  0,  3,  1,  5,  2,  3,  4,  1],  # 13
        [5,  2,  4,  1,  3,  4,  2,  1,  4,  5,  2,  5,  1,  3,  0,  2,  1,  4,  2,  3,  5],  # 14
        [3,  1,  2,  2,  1,  5,  3,  2,  1,  3,  4,  2,  5,  1,  2,  0,  3,  1,  5,  2,  4],  # 15
        [2,  5,  3,  3,  2,  1,  5,  4,  2,  1,  3,  3,  2,  5,  1,  3,  0,  2,  1,  5,  3],  # 16
        [4,  3,  1,  5,  4,  2,  1,  5,  3,  2,  1,  4,  3,  2,  4,  1,  2,  0,  3,  1,  2],  # 17
        [1,  2,  5,  1,  3,  3,  2,  1,  5,  4,  2,  1,  4,  3,  2,  5,  1,  3,  0,  2,  1],  # 18
        [3,  4,  2,  2,  1,  4,  3,  2,  1,  5,  3,  2,  1,  4,  3,  2,  5,  1,  2,  0,  3],  # 19
        [2,  1,  3,  4,  5,  1,  4,  3,  2,  1,  5,  3,  2,  1,  5,  4,  3,  2,  1,  3,  0]   # 20
    ]
    last_task = [0, 1, 2, 3, 4]  # 各设备前一天的最后一个任务（编号0-4）

    # 任务编号映射（5-20对应索引0-15）
    task_mapping = {task: idx for idx, task in enumerate(range(5, 21))}
    inverse_mapping = {v: k for k, v in task_mapping.items()}

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
                        # 修正键：使用(new_mask, new_last_tasks)作为父键
                        parent_key = (new_mask, new_last_tasks)
                        parent[parent_key] = (mask, last_tasks, t, dev)

    # 回溯路径
    if not best_state:
        print("No feasible solution")
        return

    # 重构任务分配
    allocation = {i: [] for i in range(M)}
    current_mask, current_last_tasks, current_times = best_state
    while True:
        parent_key = (current_mask, current_last_tasks)
        if parent_key not in parent:
            break
        mask, last_tasks, t, dev = parent[parent_key]
        allocation[dev].append(t)
        # 获取前一个状态
        current_mask = mask
        current_last_tasks = last_tasks
        # 获取前一个时间（仅用于循环条件）
        _, prev_times = dp[current_mask][current_last_tasks]

    # 反转任务顺序并输出
    print(f"Minimum Makespan: {min_makespan}")
    for dev in range(M):
        tasks = reversed(allocation[dev])  # 因回溯是逆序
        task_list = list(tasks)
        # 计算总时间
        total_time = 0
        if task_list:
            prev = last_task[dev]
            switch = S[prev][task_list[0]]
            total_time = switch + T[task_list[0]]
            for i in range(1, len(task_list)):
                switch = S[task_list[i-1]][task_list[i]]
                total_time += switch + T[task_list[i]]
        print(f"Device {dev+1} Tasks: {task_list}, Total Time: {total_time}")
# 输出时转换回原始任务编号
    for dev in range(M):
        task_list = [inverse_mapping[t] for t in allocation[dev]]
        print(f"Device {dev+1} Tasks: {task_list}")

if __name__ == "__main__":
    main()
