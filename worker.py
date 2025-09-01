import argparse

# from sympy import false
import utils
import threading
import time
import subprocess
import os
import pynvml
import csv
import queue

from runtime.rpc import scheduler_server
from task import Task, JobInfo



# ┌─────────────────────────────────────────────────────────────┐
# │                        Host OS Layer                        │
# ├─────────────────────────────────────────────────────────────┤
# │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
# │  │   Worker    │  │    Task     │  │   Hijack    │         │
# │  │ Scheduler   │  │  Manager    │  │   Builder   │         │
# │  └─────────────┘  └─────────────┘  └─────────────┘         │
# │         │                │                │                │
# │         └────────────────┼────────────────┘                │
# │                          │                                 │
# │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
# │  │   gsharing  │  │   Docker    │  │   RPC       │         │
# │  │   Socket    │  │   Engine    │  │  Server     │         │
# │  └─────────────┘  └─────────────┘  └─────────────┘         │
# └─────────────────────────────────────────────────────────────┘
#                           │
#                           ▼
# ┌─────────────────────────────────────────────────────────────┐
# │                    Docker Container Layer                   │
# ├─────────────────────────────────────────────────────────────┤
# │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
# │  │   Trainer   │  │   DL App    │  │   Hijack    │         │
# │  │   Client    │  │ (PyTorch/   │  │   Library   │         │
# │  │             │  │ TensorFlow) │  │             │         │
# │  └─────────────┘  └─────────────┘  └─────────────┘         │
# │         │                │                │                │
# │         └────────────────┼────────────────┘                │
# │                          │                                 │
# │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
# │  │   RPC       │  │   gsharing  │  │   CUDA      │         │
# │  │  Client     │  │   Socket    │  │  Runtime    │         │
# │  └─────────────┘  └─────────────┘  └─────────────┘         │
# └─────────────────────────────────────────────────────────────┘


# 应用程序调用CUDA函数
#     ↓
# Hijack库拦截函数调用
#     ↓
# 根据优先级执行不同逻辑：
#     ├─ 高优先级：速率监控 + 优先执行
#     └─ 低优先级：速率限制 + 延迟执行
#     ↓
# 调用原始CUDA函数
#     ↓
# 返回结果给应用程序

# TGS (Transparent GPU Sharing) Worker 是一个用于容器云环境中深度学习工作负载的GPU共享调度系统。
# 该系统支持多种GPU共享策略，包括高优先级/低优先级调度、协同执行(Co-execution)、MPS (Multi-Process Service)、MIG (Multi-Instance GPU) 和 AntMan 等


# Worker 是 TGS 系统中的一个核心组件，负责从调度器接收任务，并执行这些任务。
# Worker 通过 RPC 与调度器通信，接收任务，并执行这些任务。
class Worker(object):
    # CSV配置文件 → Worker解析 → 任务队列 → GPU分配 → Docker容器执行 → 性能监控
    def __init__(self, trace_file_path: str, worker_ip, worker_port, gpus: str, mount: list, log_path: str, need_throughput) -> None:
        # trace_file_path: CSV格式的任务配置文件路径
        # worker_ip: Worker节点IP地址
        # worker_port: RPC服务端口
        # gpus: GPU设备列表，逗号分隔
        # mount: Docker挂载点列表
        # log_path: 日志文件路径
        # need_throughput: 是否需要吞吐量监控
        super().__init__()

        self._logger = utils.make_logger(__name__)
        self._writer = utils.Writer(log_path)

        self.parse_trace_config(trace_file_path)
        
        self._worker_ip = worker_ip
        self._worker_port = worker_port
        self._worker_id = None
        self.need_throughput = need_throughput
        
        self._gpus = gpus.split(',')
        self._num_gpus = len(self._gpus)

        self._mount = mount if mount != None else []

        self.tgs_init()
        
        self._tasks = dict()

        self._server_for_trainer = self.make_server_for_trainer(worker_port)

        self._start_time = time.time()
    

    def parse_trace_config(self, trace_file_path):
        # 解析CSV格式的任务配置文件，包含以下字段：
        # submit_time: 任务提交时间
        # model_name: 模型名称
        # batch_size: 批次大小
        # iterations: 迭代次数
        # gpu_requests: GPU请求数量
        # priority: 优先级 (high/low/Co-ex/mps/mig-high/mig-low/Ex)
        # thread_percentage: 线程百分比
        # image_name: Docker镜像名称
        # antman_config: AntMan配置文件
        # antman_status: AntMan状态文件

        assert trace_file_path[-4:] == '.csv'
        trace_file = open(trace_file_path, 'r')

        reader = csv.DictReader(trace_file, delimiter=',', skipinitialspace=True)

        self._submit_queue = list()
        self.next_job_id = 1
        for row in reader:
            self.parse_job(row)
        
        trace_file.close()
        self._submit_queue = sorted(self._submit_queue, key=lambda x: (x['submit_time'], 0 if x['priority'] == 'high' else 1))


    def parse_job(self, job_spec):
        assert 'submit_time' in job_spec
        assert 'model_name' in job_spec
        assert 'batch_size' in job_spec
        assert 'iterations' in job_spec
        assert 'gpu_requests' in job_spec
        assert 'priority' in job_spec

        # if job_spec['model_name'] == 'shufflenet':
        #     job_spec['model_name'] = 'shufflenet_v2_x1_0'

        spec = {
            'submit_time': float(job_spec['submit_time']),
            'job_id': self.next_job_id,
            'model_name': job_spec['model_name'],
            'batch_size': job_spec['batch_size'],
            'iterations': int(job_spec['iterations']),
            'num_gpus': int(job_spec['gpu_requests']),
            'priority': job_spec['priority'],
            'thread_percentage': job_spec['thread_percentage'] if 'thread_percentage' in job_spec else None,
            'image_name': job_spec['image_name'] if 'image_name' in job_spec else 'tf_torch',
            'antman_config': job_spec['antman_config'] if 'antman_config' in job_spec else None,
            'antman_status': job_spec['antman_status'] if 'antman_status' in job_spec else None,
        }
        
        self._submit_queue.append(spec)
        self.next_job_id += 1


    def tgs_init(self):
        # 初始化 TGS 系统，包括构建 hijack 库和设置挂载点
        # 挂载点包括集群路径、hijack 库路径和 gsharing 库路径

        # 为不同优先级配置Docker挂载点：
        # high/low: 使用不同的库文件劫持
        # Co-ex/mps: 支持协同执行
        # MIG: 多实例GPU配置
        # AntMan: 动态GPU配置
        assert subprocess.call(['./hijack/build.sh']) == 0

        # 应用程序调用CUDA函数
        #   ↓
        # 动态链接器加载劫持库
        #   ↓
        # 劫持库拦截函数调用
        #   ↓
        # 执行控制逻辑（速率限制、资源管理）
        #   ↓
        # 调用原始CUDA函数
        #   ↓
        # 返回结果给应用程序

        # 通过这种劫持机制，TGS能够：

        # 透明地控制GPU使用率：低优先级任务会被自动限流
        # 实现资源隔离：高优先级任务不受低优先级任务影响
        # 动态调整策略：根据系统负载实时调整控制参数
        # 保持兼容性：所有现有的CUDA应用程序都能正常运行
        # 这种劫持技术是TGS实现"透明GPU共享"的核心机制，让系统能够在应用层完全透明的情况下实现底层的GPU资源控制

        root_path = os.path.abspath('.')

        self.tgs_mounts = {
            'high': [
                root_path + ':/cluster',
                root_path + '/hijack/high-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/high-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
            'low': [
                root_path + ':/cluster',
                root_path + '/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/low-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
            'Ex': [
                root_path + ':/cluster',
            ],
            'Co-ex': [
                root_path + ':/cluster',
            ],
            'mig-high': [
                root_path + ':/cluster',
            ],
            'mig-low': [
                root_path + ':/cluster',
            ],
            'mps': [
                root_path + ':/cluster',
                '/tmp/nvidia-mps:/tmp/nvidia-mps',
            ],
        }


    def check_tasks(self):
        finished_tasks = []

        for job_id, task in self._tasks.items():
            if task.return_code == None:
                continue
            assert task._finished_iterations == task._iterations
            
            finished_tasks.append(task)
        
        if len(finished_tasks) > 0:
            self.record()
        for task in finished_tasks:
            self._tasks.pop(task._job_id)
        
        return finished_tasks
    

    def execute(self, job_info) -> bool:
        # 执行任务，包括创建任务对象、运行任务、记录日志
        success = True

        task = Task(job_info, self._worker_ip, self.tgs_mounts, self.need_throughput)
        self._tasks[task._job_id] = task
        cmd = task.run(self._mount)

        self._logger.info(f'{self._worker_id}, execute, {task._job_id}, {task._gpus}, {task._priority}, {" ".join(cmd)}')

        return success
    

    def kill(self, job_info) -> bool:
        # 杀死任务，包括终止容器、记录日志
        job_id = job_info.job_id

        if job_id not in self._tasks:
            return False

        task = self._tasks.pop(job_id)
        task.terminate()

        self._logger.info(f'{self._worker_id}, kill, {job_id}, {job_info.gpus}, {job_info.priority}')

        return True
    

    def query_node_stats(self):
        # 查询GPU利用率
        utilizations = []
        pynvml.nvmlInit()
        for gpu_id in range(self._num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            utilizations.append(str(utilization))
        pynvml.nvmlShutdown()

        self._logger.info(f'{self._worker_id}, query, {"-".join(utilizations)}')
        utilizations = ','.join(utilizations)
        return utilizations


    def _report_stats_impl(self, job_id, finished_iterations) -> bool:
        # 报告任务性能数据
        success = True
        assert job_id in self._tasks
        task = self._tasks[job_id]
        throughput = task.update(finished_iterations)

        self._logger.info(f'worker, report, {job_id}, {throughput}, {task._finished_iterations}')

        return success


    def make_server_for_trainer(self, port):
        # 创建RPC服务端，用于接收调度器报告任务性能数据
        # 通过RPC接收训练器报告
        callbacks = {
            'ReportStats' : self._report_stats_impl,
        }

        return scheduler_server.serve(port, self._logger, callbacks)


    def has_ready_jobs(self):
        # 检查是否有可运行的任务
        current_time = time.time()
        elapsed_time = current_time - self._start_time

        if len(self._submit_queue) > 0:
            job_spec = self._submit_queue[0]
            if job_spec['submit_time'] <= elapsed_time:
                return True
        
        return False


    def record(self):
        # 记录性能数据
        timestamp = time.time() - self._start_time
        for task in self._tasks.values():
            task.record(timestamp, self._writer)


    def close(self):
        self._writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_port', type=int, default=6889)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--mount', action='append')
    parser.add_argument('--trace', type=str,  required=True) # default='config/test_tgs.csv')
    parser.add_argument('--log_path', type=str,  required=True) # default='results/test_tgs_results.csv')
    parser.add_argument('--need_throughput', action='store_true', default=False)
    args = parser.parse_args()

    subprocess.call('docker stop $(docker ps -q)', shell=True)
    subprocess.call('docker rm $(docker ps -aq)', shell=True)

    worker_ip = utils.get_host_ip()
    worker = Worker(args.trace, worker_ip, args.worker_port, args.gpus, args.mount, args.log_path, args.need_throughput)

    runnable_tasks = list()
    gpu_list = args.gpus.split(',')
    machine = [{
        'Co-ex': list(),
        'mps': list()
    } for i in range(len(gpu_list))]

    # 主调度循环，持续处理任务
    while len(worker._submit_queue) + len(worker._tasks) + len(runnable_tasks) > 0:
        while worker.has_ready_jobs():
            job_spec = worker._submit_queue.pop(0)
            jobinfo = JobInfo(job_spec['job_id'], job_spec['model_name'], job_spec['batch_size'],
                 job_spec['iterations'], job_spec['num_gpus'], job_spec['priority'],
                 job_spec['thread_percentage'], job_spec['image_name'],
                 job_spec['antman_config'], job_spec['antman_status']
                )
            runnable_tasks.append(jobinfo)

        finished_tasks = worker.check_tasks()
        for task in finished_tasks:
            for gpu_id in task._gpus.split(','):
                if task._priority in ['Co-ex', 'mps']:
                    machine[int(gpu_id)][jobinfo.priority].remove(task._job_id)
                else:
                    machine[int(gpu_id)].pop(task._priority)
            # writer.save(task)
        
        new_runnable_tasks = []
        record_flag = (len(finished_tasks) != 0)

        # 遍历可运行任务，分配GPU
        for jobinfo in runnable_tasks:
            available_gpus = 0
            for gpu_instance in machine:
                if jobinfo.priority not in gpu_instance:
                    available_gpus += 1
                elif jobinfo.priority in ['Co-ex', 'mps'] and len(gpu_instance[jobinfo.priority]) < 2:
                    available_gpus += 1
            
            if available_gpus >= jobinfo.num_gpus:
                record_flag = True
                used_gpus = []
                for gpu_id, gpu_instance in enumerate(machine):
                    if jobinfo.priority not in gpu_instance:
                        used_gpus.append(str(gpu_id))
                        gpu_instance[jobinfo.priority] = jobinfo.job_id
                    elif jobinfo.priority in ['Co-ex', 'mps'] and len(gpu_instance[jobinfo.priority]) < 2:
                        used_gpus.append(str(gpu_id))
                        gpu_instance[jobinfo.priority].append(jobinfo.job_id)
                    
                    if len(used_gpus) == jobinfo.num_gpus:
                        break
                jobinfo.gpus = ','.join(used_gpus)
                worker.execute(jobinfo)
            else:
                new_runnable_tasks.append(jobinfo)

        # 记录性能数据
        if record_flag:
            worker.record()
        runnable_tasks = new_runnable_tasks

        sleep_time = 2
        if len(worker._submit_queue) > 0:
            sleep_time = min(sleep_time, (worker._start_time + worker._submit_queue[0]['submit_time'] - time.time()))
        time.sleep(sleep_time)
    
    worker.close()


    # TGS的调度是一个多层次的系统，既有应用层的任务调度，也有底层的CUDA库劫持来实现透明的GPU共享控制。
    # 核心的TGS调度逻辑在hijack/src/low_priority_hijack_call.c中的速率控制算法，而整体的任务调度框架在worker.py的主循环中实现