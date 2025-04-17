import itertools

def dp_schedule_with_switching(M, N, T, S, last_task):
    """
    动态规划算法解决任务调度问题，最小化 Makespan，同时确保每个任务只被分配一次。
    
    参数：
    - M: 设备数量
    - N: 任务数量
    - T: 每个任务的生产时间
    - S: 切换时间矩阵
    - last_task: 每台设备前一天的最后任务
    
    返回：
    - 最小 Makespan
    - 每台设备的任务分配方案
    """
    inf = float('inf')
    all_tasks = (1 << N) - 1  # 所有任务的集合
    dp = [[inf] * M for _ in range(1 << N)]  # dp[mask][k] 表示任务集合 mask 分配到设备 k 的最小完成时间
    path = [[None] * M for _ in range(1 << N)]  # 路径追踪

    # 初始化
    for k in range(M):
        dp[0][k] = 0  # 初始状态，所有设备的完成时间为 0

    # 动态规划
    for mask in range(1 << N):  # 枚举任务集合
        for k in range(M):  # 枚举设备
            if dp[mask][k] == inf:  # 跳过无效状态
                continue
            for i in range(N):  # 枚举下一个任务
                if mask & (1 << i):  # 如果任务 i 已经在集合中，跳过
                    continue
                next_mask = mask | (1 << i)  # 将任务 i 加入集合
                switch_time = S[last_task[k]][i] if mask == 0 else S[last_task[k]][i]
                new_time = dp[mask][k] + switch_time + T[i]
                if new_time < dp[next_mask][k]:
                    dp[next_mask][k] = new_time
                    path[next_mask][k] = (mask, k, i)  # 保存路径

    # 找到最小 Makespan
    min_makespan = inf
    end_state = None
    for k in range(M):
        if dp[all_tasks][k] < min_makespan:
            min_makespan = dp[all_tasks][k]
            end_state = (all_tasks, k)

    # 回溯任务分配
    schedules = [[] for _ in range(M)]
    current_state = end_state
    while current_state and current_state[0] != 0:
        mask, device = current_state
        prev_mask, prev_device, task = path[mask][device]
        schedules[device].append(task)
        current_state = (prev_mask, prev_device)

    # 反转每台设备的任务顺序
    for device in range(M):
        schedules[device].reverse()

    # 确保任务尽可能分配到更多设备
    redistribute_tasks(schedules, M)

    return min_makespan, schedules

def redistribute_tasks(schedules, M):
    """
    确保任务尽可能分配到更多设备。
    如果某些设备没有任务，将任务从负载较多的设备重新分配到空闲设备。
    """
    empty_devices = [i for i in range(M) if not schedules[i]]
    while empty_devices:
        # 找到负载最多的设备
        max_load_device = max(range(M), key=lambda x: len(schedules[x]))
        if not schedules[max_load_device]:
            break
        # 将任务从负载最多的设备分配到空闲设备
        task_to_move = schedules[max_load_device].pop()
        empty_device = empty_devices.pop()
        schedules[empty_device].append(task_to_move)

    # 检查是否均衡分配
    for device in range(M):
        if not schedules[device]:
            print(f"Warning: Device {device + 1} has no tasks assigned!")

# 示例数据
M = 4  # 设备数量
N = 8  # 任务数量
T = [3, 2, 4, 1, 5, 6, 2, 3]  # 每个任务的生产时间
S = [  # 切换时间矩阵
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 2, 1, 3, 4, 5, 6],
    [2, 2, 0, 1, 2, 3, 4, 5],
    [3, 1, 1, 0, 1, 2, 3, 4],
    [4, 3, 2, 1, 0, 1, 2, 3],
    [5, 4, 3, 2, 1, 0, 1, 2],
    [6, 5, 4, 3, 2, 1, 0, 1],
    [7, 6, 5, 4, 3, 2, 1, 0]
]
last_task = [0, 3, 5, 7]  # 每个设备前一天的最后任务

# 调用动态规划算法
min_makespan, schedules = dp_schedule_with_switching(M, N, T, S, last_task)

# 输出结果
print(f"Minimum Makespan: {min_makespan}")
print("Device Schedules:")
for device, tasks in enumerate(schedules):
    print(f"  Device {device + 1}: {tasks}")