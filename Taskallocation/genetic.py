import random

# 适应度函数：计算 Makespan
def calculate_makespan(schedules, T, S, last_task):
    M = len(schedules)
    makespan = [0] * M  # 每个设备的完成时间
    for device, tasks in enumerate(schedules):
        if not tasks:
            continue
        time = S[last_task[device]][tasks[0]] + T[tasks[0]]  # 第一个任务
        for i in range(1, len(tasks)):
            time += S[tasks[i - 1]][tasks[i]] + T[tasks[i]]
        makespan[device] = time
    return max(makespan)

# 初始化种群
def initialize_population(pop_size, M, N):
    population = []
    for _ in range(pop_size):
        # 随机分配任务到设备，确保每个设备至少有一个任务
        schedules = [[] for _ in range(M)]
        tasks = list(range(N))
        random.shuffle(tasks)
        for i, task in enumerate(tasks):
            schedules[i % M].append(task)
        population.append(schedules)
    return population

# 选择操作：轮盘赌选择
def select(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected = random.choices(population, weights=probabilities, k=2)
    return selected

# 交叉操作：单点交叉，确保任务不重复且每个设备至少有一个任务
def crossover(parent1, parent2, M):
    N = sum(len(tasks) for tasks in parent1)  # 总任务数
    child1, child2 = [[] for _ in range(M)], [[] for _ in range(M)]
    all_tasks = set(range(N))

    # 随机选择一个交叉点
    crossover_point = random.randint(1, N - 1)

    # 子代继承父代的部分任务
    assigned_tasks1, assigned_tasks2 = set(), set()
    for i in range(M):
        for task in parent1[i]:
            if len(assigned_tasks1) < crossover_point:
                child1[i].append(task)
                assigned_tasks1.add(task)
        for task in parent2[i]:
            if len(assigned_tasks2) < crossover_point:
                child2[i].append(task)
                assigned_tasks2.add(task)

    # 补充剩余任务，确保每个设备至少有一个任务
    remaining_tasks1 = list(all_tasks - assigned_tasks1)
    remaining_tasks2 = list(all_tasks - assigned_tasks2)
    random.shuffle(remaining_tasks1)
    random.shuffle(remaining_tasks2)

    for task in remaining_tasks1:
        for i in range(M):
            if task not in child1[i]:
                child1[i].append(task)
                break

    for task in remaining_tasks2:
        for i in range(M):
            if task not in child2[i]:
                child2[i].append(task)
                break

    # 确保每个设备至少有一个任务
    for i in range(M):
        if not child1[i] and remaining_tasks1:  # 检查 remaining_tasks1 是否为空
            child1[i].append(remaining_tasks1.pop())
        if not child2[i] and remaining_tasks2:  # 检查 remaining_tasks2 是否为空
            child2[i].append(remaining_tasks2.pop())

    return child1, child2

# 变异操作：随机交换任务，确保任务不重复且每个设备至少有一个任务
def mutate(schedules, mutation_rate):
    if random.random() < mutation_rate:
        device1, device2 = random.sample(range(len(schedules)), 2)
        if len(schedules[device1]) > 1 and len(schedules[device2]) > 1:
            task1 = random.choice(schedules[device1])
            task2 = random.choice(schedules[device2])
            schedules[device1].remove(task1)
            schedules[device2].remove(task2)
            schedules[device1].append(task2)
            schedules[device2].append(task1)

# 遗传算法主函数
def genetic_algorithm(M, N, T, S, last_task, pop_size=50, generations=100, mutation_rate=0.1):
    # 初始化种群
    population = initialize_population(pop_size, M, N)

    for generation in range(generations):
        # 计算适应度
        fitness = [1 / calculate_makespan(ind, T, S, last_task) for ind in population]

        # 生成下一代
        new_population = []
        while len(new_population) < pop_size:
            # 选择父代
            parent1, parent2 = select(population, fitness)
            # 交叉
            child1, child2 = crossover(parent1, parent2, M)
            # 变异
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # 更新种群
        population = new_population[:pop_size]

    # 返回最优解
    best_schedule = min(population, key=lambda ind: calculate_makespan(ind, T, S, last_task))
    best_makespan = calculate_makespan(best_schedule, T, S, last_task)
    return best_makespan, best_schedule

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

# 调用遗传算法
min_makespan, schedules = genetic_algorithm(M, N, T, S, last_task)

# 输出结果
print(f"Minimum Makespan: {min_makespan}")
print("Device Schedules:")
for device, tasks in enumerate(schedules):
    print(f"  Device {device + 1}: {tasks}")
