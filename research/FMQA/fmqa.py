##### IMPORT #####

import os
import sys
import time
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from zone_plate import ZonePlate

# Setting Environment Variable (Has to be set before importing mxnet, or it will perform multiprocessing automatically but without any imrpoved efficiency)
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Factorization Machine
import mxnet as mx
from mxnet import nd
from factorization_machine import FactorizationMachine as OriginalFactorizationMachine

# Fix the random seeds
random.seed(int(time.time()))
np.random.seed(int(time.time()))
mx.random.seed(int(time.time()))

# QUBO Sampler
import dimod
import dwave.inspector
from dimod import ExactSolver
from dwave.samplers import RandomSampler, SimulatedAnnealingSampler, SteepestDescentSampler, TabuSampler
from dwave.system import DWaveSampler, DWaveCliqueSampler, EmbeddingComposite, FixedEmbeddingComposite

##### FM #####

class FactorizationMachine(OriginalFactorizationMachine):

    def __call__(self, xs):
        """Override __call__ to handle numpy array input automatically."""
        if isinstance(xs, np.ndarray):
            xs = nd.array(xs)  # 自動轉換 numpy 陣列
        return super().__call__(xs)
    
    def loss(self, dataset):
        """
        計算loss
        - mean((ys - outputs)**2)
    
        參數
        - dataset: [xs, ys]
        """
        xs = nd.array(dataset[0])
        ys = nd.array(dataset[1])
        return nd.mean( ( ys - self(xs) ) ** 2 ).asscalar()

    def get_bhQ_scaled(self):
        """原始程式有做scaling"""
        # b: bias, h: linear, Q: quadratic (上三角部分)，皆為 numpy array
        b, h, Q = self.get_bhQ()  
        
        h_max = np.max(np.abs(h))
        Q_max = np.max(np.abs(Q))
        scaling_factor = max(h_max, Q_max)
        
        b /= scaling_factor
        h /= scaling_factor
        Q /= scaling_factor

        return b, h, Q

    def bqm(self):
        """從 model 得到 bqm"""
        b, h, Q = self.get_bhQ()
        # b, h, Q = self.get_bhQ_scaled()    # b: bias, h: linear, Q: quadratic (上三角部分)，皆為 numpy array

        return dimod.BinaryQuadraticModel(h, Q, b, dimod.BINARY)    # BQM(linear, quadratic, offset, vartype)

    def plot_Q_matrix(self, show_fig=False, path_fig="Q_matrix_heatmap.png", save_fig=True, path_data="Q_matrix.npy", save_data=True):
        """畫Q matrix的heatmap"""
        b, h, Q = self.get_bhQ()
        # b, h, Q = self.get_bhQ_scaled()    # bias, linear, qudratic(upper tri)
        Q = Q + Q.T
        np.fill_diagonal(Q, h)
        
        if save_data:
            np.save(path_data, Q)

        if show_fig or save_fig:
            max_abs_val = np.max(np.abs(Q))
            plt.figure()
            plt.imshow(Q, cmap="bwr", vmin=-max_abs_val, vmax=max_abs_val)
            plt.colorbar()
            plt.title("Q Matrix")

            if save_fig:
                plt.savefig(path_fig, bbox_inches='tight', transparent=True)
            if show_fig:
                plt.show()
            else:
                plt.close()

    def save_model(self, path="model.params"):
        """儲存model參數、optimizer狀態"""
        mx.nd.waitall()
        self.save_parameters(path)
        
        if self.trainer is not None:
            self.trainer.save_states(path + ".trainer")

    @staticmethod
    def load_model(var_num, K, path="model.params"):
        """
        讀取並返回model參數、optimizer狀態

        參數
        - var_num : qubits數
        - K       : Factorization Machine的K
        """
        model = FactorizationMachine(input_size=var_num, factorization_size=K, act="identity")
        model.load_parameters(path, ctx=mx.cpu())

        # 需要先初始化 Trainer，然後才能載入 state
        model.trainer = mx.gluon.Trainer(model.collect_params(), "adam")

        try:
            model.trainer.load_states(path + ".trainer")
        except FileNotFoundError:
            print("沒有讀到optimizer的參數")

        return model

##### VARS & OBJS #####

# vars & obj for ZonePlate
rings      = 64
multiple   = 8.0
resolution = 2_000
r_vec      = np.linspace(0, multiple, resolution)    # unit: wavelength
wavelength = 0.53                                    # unit: µm
k          = 2 * np.pi / wavelength
NA         = 0.95

zp = ZonePlate(multiple, resolution, wavelength, NA)

target_method = "intensity_step_tanh"
target_max_mul = 8.0
target_step_point_mul = float(sys.argv[1])
target_step_left_val = float(sys.argv[2])
target_tanh_thresh = 0.4
target_tanh_slope = 10
target_tanh_beta = 2
target = [target_method, target_max_mul, target_step_point_mul, target_step_left_val, target_tanh_thresh, target_tanh_slope, target_tanh_beta]
target_str = "_".join(map(str, target))

# vars for FM
input_size    = rings    # dimension of each config
K             = 8        # the latent factor in FM
num_epoch     = 1_000
learning_rate = 1.0e-2

# vars for QUBO
sampler_type = "SA"    # 有Exact, Random, SA, SD, Tabu, SA_SD, SA_Tabu, QA_Auto, QA_Clique
qpu_solver = None      # {"name": "Advantage2_system1.1"}

# vars for FMQA
init_dataset_size = 10_000
iteration         = 30
num_reads         = 2_000       # annealing times per iter
adding_method     = "FoM"    # "energy" | "FoM"
adding_num        = 1_000

##### PAHTS #####

def create_init_path(base_folder_name, program_id=None):
    """Creates a folder for this simulation

    Args:
        base_folder_name (str): Name of the parent directory (must be a valid path string).
        program_id (int, optional): Unique identifier for the simulation task. Defaults to None (auto-generated).
    """
    os.makedirs(base_folder_name, exist_ok=True)

    if program_id is None:
        # 找出所有數字資料夾的最大值
        existing_nums = []
        for folder in os.listdir(base_folder_name):
            folder_path = os.path.join(base_folder_name, folder)
            if os.path.isdir(folder_path) and folder.isdigit():
                existing_nums.append(int(folder))
        
        # 設定 program_id
        program_id = str(max(existing_nums, default=0) + 1)
    else:
        # 確保 program_id 是字串
        program_id = str(program_id)

    # Creates a folder corresponding to a specific program_id
    base_path = os.path.join(base_folder_name, program_id)
    os.makedirs(base_path)
    
    return program_id

# 設定基礎路徑
base_folder_name = "saved_files"
if len(sys.argv) > 3:
    program_id = sys.argv[3]
else:
    program_id = None
program_id       = create_init_path(base_folder_name, program_id=program_id)
base_path        = os.path.join(base_folder_name, program_id)
print(f"program id: {program_id}")

# 記錄設定參數檔案的路徑
params_file_name = "params.txt"
params_path      = os.path.join(base_path, params_file_name)

# configs & foms的base path（保存在與程式檔同層的資料夾底下的configs_foms資料夾，這樣其他程式也可以使用）
configs_foms_folder_name = "configs_foms"
os.makedirs(configs_foms_folder_name, exist_ok=True)

# 初始生成組態 & 對應的FoM的檔名
init_configs_file_name = f"init_configs_{init_dataset_size}_{rings}_{multiple}_{resolution}_{wavelength}_{NA}.npy"
init_foms_file_name    = f"init_foms_{init_dataset_size}_{target_str}_{rings}_{multiple}_{resolution}_{wavelength}_{NA}.npy"
init_configs_path      = os.path.join(configs_foms_folder_name, init_configs_file_name)
init_foms_path         = os.path.join(configs_foms_folder_name, init_foms_file_name)

# 跑完FMQA得到的configs, energies, foms
configs_file_name  = "configs.npy"
energies_file_name = "energies.npy"
foms_file_name     = "foms.npy"
configs_path       = os.path.join(base_path, configs_file_name)
energies_path      = os.path.join(base_path, energies_file_name)
foms_path          = os.path.join(base_path, foms_file_name)

# FMQA每次新增的筆數(actual_new_data)
actual_new_data_file_name = "actual_new_data.npy"
actual_new_data_path = os.path.join(base_path, actual_new_data_file_name)

# fmqa每次iter的model跟model的Q_matrix（每個iter都要調整path）
iter_model_Q_folder_name = "model&Q"
iter_model_Q_path        = os.path.join(base_path, iter_model_Q_folder_name)
os.makedirs(iter_model_Q_path)
iter_model_file_name = "model.params"
iter_Q_arr_file_name = "Q.npy"
iter_Q_fig_file_name = "Q.png"

# write the setting vars to a file

variables = {
    
    # vars for ZonePlate
    "rings"     : rings,
    "multiple"  : multiple,
    "resolution": resolution,
    "wavelength": wavelength,
    "NA"        : NA,
    "target"    : target,
    
    # vars for FM
    "K"        : K,
    "num_epoch": num_epoch,
    
    # vars for QUBO
    "sampler_type": sampler_type,
    "qpu_solver"  : qpu_solver,
    
    # vars for FMQA
    "init_dataset_size": init_dataset_size,
    "iteration"        : iteration,
    "num_reads"        : num_reads,
    "adding_method"    : adding_method,
    "adding_num"       : adding_num,
}

with open(params_path, "w", encoding="utf-8") as f:
    for key, value in variables.items():
        f.write(f"{key} = {value}\n")

##### FUNCS #####

def generate_unique_binary_array(init_dataset_size, rings):
    max_unique = 2 ** rings
    
    if init_dataset_size > max_unique:
        raise ValueError(f"Cannot generate {init_dataset_size} unique rows with only {rings} bits. Maximum is {max_unique}.")
    
    def unique_binary_generator():
        seen = set()
        while len(seen) < init_dataset_size:
            new_row = tuple(np.random.randint(0, 2, size=rings, dtype=np.int8))
            if new_row not in seen:
                seen.add(new_row)
                yield np.array(new_row, dtype=np.int8)
    
    # Use a generator to yield rows one at a time
    result = np.array(list(unique_binary_generator()))
    
    # Shuffle the rows
    np.random.shuffle(result)
    
    return result

def choose_sampler(sampler_type = "SA"):
    """
    定義要使用的sampler
    全域變數: qpu_solver
    """
    if sampler_type == "Exact":
        if rings <= 16:
            sampler = ExactSolver()
        else:
            print("環數太多，不能使用Exact Solver")
            sys.exit(1)
    elif sampler_type == "Random":
        sampler = RandomSampler()
    elif sampler_type == "SA":
        sampler = SimulatedAnnealingSampler()
    elif sampler_type == "SD":
        sampler = SteepestDescentSampler()
    elif sampler_type == "Tabu":
        sampler = TabuSampler()
    elif sampler_type == "QA_Auto":
        sampler = EmbeddingComposite(DWaveSampler(qpu_solver))
    elif sampler_type == "QA_Clique":
        sampler = DWaveCliqueSampler(qpu_solver)
    else:
        print("沒有這個QUBO sampler")
        sys.exit(1)
    
    return sampler

def sampling(sampler_type, bqm, num_reads=None):
    """
    開始annealing/sampling
    全域變數: 無
    """

    sampler_types = ["Exact", "Random", "SA", "SD", "Tabu", "QA_Auto", "QA_Clique"]
    if sampler_type in sampler_types:
        sampler = choose_sampler(sampler_type)
    
    # 單次模擬
    if sampler_type == "Exact":
        sampleset = sampler.sample(bqm)
    elif sampler_type == "Random":
        sampleset = sampler.sample(bqm, num_reads=num_reads)
    elif sampler_type == "SA":
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=1_000)
    elif sampler_type == "SD":
        sampleset = sampler.sample(bqm, num_reads=num_reads)
    elif sampler_type == "Tabu":
        sampleset = sampler.sample(bqm, num_reads=num_reads)
    elif sampler_type == "QA_Auto":
        sampleset = sampler.sample(bqm, num_reads=num_reads, annealing_time=20)
    elif sampler_type == "QA_Clique":
        sampleset = sampler.sample(bqm, num_reads=num_reads, annealing_time=20)

    # 多次模擬
    elif sampler_type == "SA_SD":
        sampler = choose_sampler("SA")
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=1_000)
        sampler = choose_sampler("SD")
        sampleset = sampler.sample(bqm, num_reads=num_reads, initial_states=sampleset)
    elif sampler_type == "SA_Tabu":
        sampler = choose_sampler("SA")
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=1_000)
        sampler = choose_sampler("Tabu")
        sampleset = sampler.sample(bqm, num_reads=num_reads, initial_states=sampleset)

    # 
    else:
        print("沒有這個sampler")
        sys.exit(1)
    
    return sampleset

def add_data(configs, energies, foms, sample_configs, sample_energies, adding_method, adding_num):
    
    # step1: choose
    if adding_method == "energy":
        sorted_indices  = np.argsort(sample_energies)[:adding_num]
        sample_configs  = sample_configs[sorted_indices]
        sample_energies = sample_energies[sorted_indices]
        sample_foms     = np.array([
                              zp.set_config(config) or zp.figure_of_merit_calc.calc_fom(target)
                              for config in sample_configs
                          ])
    elif adding_method == "FoM":
        sample_foms     = np.array([
                              zp.set_config(config) or zp.figure_of_merit_calc.calc_fom(target)
                              for config in sample_configs
                          ])
        sorted_indices  = np.argsort(sample_foms)[:adding_num]
        sample_configs  = sample_configs[sorted_indices]
        sample_energies = sample_energies[sorted_indices]
        sample_foms     = sample_foms[sorted_indices]
    else:
        return
    
    # step2: filter out the repeated data
    unique_indices  = np.unique(sample_configs, axis=0, return_index=True)[1]    # 移除 sample_configs 中重複的 row，並同步刪除 sample_foms 對應元素
    unique_indices  = np.sort(unique_indices)
    sample_configs  = sample_configs[unique_indices]
    sample_energies = sample_energies[unique_indices]
    sample_foms     = sample_foms[unique_indices]
    mask = np.array([not np.any(np.all(configs == row, axis=1)) for row in sample_configs])    # 移除 sample_configs 中出現在 configs 的 row，並同步刪除 sample_foms 對應元素
    sample_configs  = sample_configs[mask]
    sample_energies = sample_energies[mask]
    sample_foms     = sample_foms[mask]

    # step3: sort
    sort_indices    = np.argsort(sample_foms)[::-1]    # 依照FoM由大到小排序
    sample_configs  = sample_configs[sort_indices]
    sample_energies = sample_energies[sort_indices]
    sample_foms     = sample_foms[sort_indices]

    # step4: add
    configs  = np.r_[configs, sample_configs]
    energies = np.r_[energies, sample_energies]
    foms     = np.r_[foms, sample_foms]

    return configs, energies, foms

##### FMQA #####

# generate initial data
if os.path.exists(init_configs_path):
    configs = np.load(init_configs_path)
else:
    configs = generate_unique_binary_array(init_dataset_size=init_dataset_size, rings=rings)
    np.save(init_configs_path, configs)

if os.path.exists(init_foms_path):
    foms = np.load(init_foms_path)
else:
    foms = np.array([
               zp.set_config(config) or zp.figure_of_merit_calc.calc_fom(target)
               for config in configs
           ])
    np.save(init_foms_path, foms)
print(np.min(foms))

# initialization for running FMQA

actual_new_data = []
energies = np.array([])
min_fom = np.min(foms)
min_fom_iter = 0

# initialize FM
model = FactorizationMachine(input_size=input_size, factorization_size=K, act="identity")
model.init_params(initializer=mx.init.Normal())

for i in range(iteration):

    print(f"*** {i+1}th iteration ***")

    # define file names
    start_time = time.time()
    iter_model_path = os.path.join(iter_model_Q_path, f"{i+1}th_{iter_model_file_name}")
    iter_Q_arr_path = os.path.join(iter_model_Q_path, f"{i+1}th_{iter_Q_arr_file_name}")
    iter_Q_fig_path = os.path.join(iter_model_Q_path, f"{i+1}th_{iter_Q_fig_file_name}")
    end_time = time.time()
    print(f"設定路徑: {end_time - start_time} s")

    # train FM
    start_time = time.time()
    model.train(configs, foms, num_epoch, learning_rate)
    model.save_model(path=iter_model_path)
    end_time = time.time()
    print(f"訓練FM: {end_time - start_time} s")

    # plot Q matrix
    start_time = time.time()
    model.plot_Q_matrix(show_fig=False, path_fig=iter_Q_fig_path, save_fig=True, path_data=iter_Q_arr_path, save_data=True)
    end_time = time.time()
    print(f"畫Q matrix: {end_time - start_time} s")

    # sampling from a QUBO sampler
    start_time = time.time()
    sampleset = sampling(sampler_type=sampler_type, bqm=model.bqm(), num_reads=num_reads)
    end_time = time.time()
    print(f"sampling: {end_time - start_time} s")

    # adding back new data to dataset
    start_time = time.time()
    sample_configs  = sampleset.record["sample"]
    sample_energies = sampleset.record["energy"]
    configs, energies, foms = add_data(configs, energies, foms, sample_configs, sample_energies, adding_method, adding_num)
    end_time = time.time()
    print(f"篩選數據: {end_time - start_time} s")

    # record min FoM
    if min_fom > np.min(foms):
        min_fom      = np.min(foms)
        min_fom_iter = i + 1
    print(f"目前最小的FoM: {min_fom}")

    print("=" * 50)

# save: configs, energies, foms
np.save(configs_path, configs)
np.save(energies_path, energies)
np.save(foms_path, foms)

# save: actual_new_data
actual_new_data = np.array(actual_new_data)
np.save(actual_new_data_path, actual_new_data)

# create a folder with name including "in which iter was the min FoM found" and "min FoM"
min_fom_folder = f"第{min_fom_iter}次iter找到最小FoM:{min_fom}"
min_fom_path = os.path.join(base_path, min_fom_folder)
os.makedirs(min_fom_path)