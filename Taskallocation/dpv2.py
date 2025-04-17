import sys
from itertools import product
from functools import lru_cache

def main():
    # 示例输入
    # 假设设备数为2，任务数为3，切换时间矩阵，生产时间列表，各设备前一天的最后一个任务
    m = 2
    prev_tasks = ['prev_1', 'prev_2']
    tasks = ['t1', 't2', 't3']
    p = {'t1': 2, 't2': 3, 't3': 1}
    s = {
        'prev_1': {'t1': 1, 't2': 2, 't3': 3},
        'prev_2': {'t1': 2, 't2': 1, 't3': 2},
        't1': {'t1': 0, 't2': 1, 't3': 2},
        't2': {'t1': 1, 't2': 0, 't3': 1},
        't3': {'t1': 2, 't2': 1, 't3': 0}
    }

    n = len(tasks)
    task_ids = {task: i for i, task in enumerate(tasks)}
    all_tasks = set(tasks)

    # 使用字典来保存状态，键为元组（已分配的任务掩码，各设备状态），值为最大时间
    dp = {}
    # 初始状态：掩码为0（无任务分配），各设备最后任务为prev，时间为0
    initial_state = tuple((prev, 0) for prev in prev_tasks)
    dp[(0, initial_state)] = 0

    # 记录父状态和分配信息用于回溯
    parent = {}

    # 遍历所有可能的掩码和状态
    for mask in range(0, 1 << n):
        for state in list(dp.keys()):
            current_mask, device_states = state
            if current_mask != mask:
                continue
            current_max = dp[state]
            # 遍历所有可能的未分配任务
            for t in tasks:
                tid = task_ids[t]
                if (current_mask >> tid) & 1:
                    continue  # 任务已分配
                # 遍历所有设备进行分配
                for dev_idx in range(m):
                    prev_dev_task, dev_time = device_states[dev_idx]
                    # 计算切换时间和新时间
                    if dev_time == 0:
                        switch = s[prev_dev_task][t]
                        new_time = switch + p[t]
                    else:
                        switch = s[prev_dev_task][t]
                        new_time = dev_time + switch + p[t]
                    # 创建新的设备状态
                    new_device_states = list(device_states)
                    new_device_states[dev_idx] = (t, new_time)
                    new_device_states = tuple(new_device_states)
                    # 计算新掩码
                    new_mask = current_mask | (1 << tid)
                    # 新的最大时间
                    new_max = max(current_max, new_time)
                    # 更新DP表
                    if (new_mask, new_device_states) not in dp or new_max < dp[(new_mask, new_device_states)]:
                        dp[(new_mask, new_device_states)] = new_max
                        parent[(new_mask, new_device_states)] = (state, t, dev_idx)

    # 寻找最优解
    min_makespan = float('inf')
    best_state = None
    full_mask = (1 << n) - 1
    for state in dp:
        mask, device_states = state
        if mask == full_mask and dp[state] < min_makespan:
            min_makespan = dp[state]
            best_state = state

    # 回溯路径
    if best_state is None:
        print("No solution found")
        return

    # 收集分配顺序
    allocation = {i: [] for i in range(m)}
    current_state = best_state
    while current_state in parent:
        state_info, task, dev_idx = parent[current_state]
        allocation[dev_idx].append(task)
        current_state = state_info

    # 反转顺序，因为回溯是逆序的
    for dev in allocation:
        allocation[dev].reverse()

    # 输出结果
    print("Minimum Makespan:", min_makespan)
    for dev in range(m):
        tasks_assigned = allocation[dev]
        # 初始任务可能被覆盖，需要检查设备是否有任务
        # 需要从初始状态开始追踪完整的任务顺序
        # 这里的回溯仅显示在动态规划过程中添加的任务，可能遗漏前置的prev任务
        # 更复杂的回溯可能需要记录完整历史
        print(f"Device {dev + 1} tasks:", tasks_assigned)

if __name__ == "__main__":
    main()