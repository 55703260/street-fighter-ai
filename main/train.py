import os
import sys
import retro
from stable_baselines3 import PPO #用于游戏模拟
from stable_baselines3.common.monitor import Monitor  #监控学习环境
from stable_baselines3.common.callbacks import CheckpointCallback  #定期保存模型的回调函数
from stable_baselines3.common.vec_env import SubprocVecEnv  # 用于并行环境

#导入自定义环境包装器
from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_ENV = 16  #并行环境数目，原值16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# 定义线性调度器，用于动态调整学习率或剪裁范围
def linear_schedule(initial_value, final_value=0.0):
    # 确保初始值和最终值都是浮点数，并且初始值大于0
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    # 调度器函数，根据进度调整当前值
    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, #定义动作空间
            obs_type=retro.Observations.IMAGE    #状态空间被定义为由retro.Observations.IMAGE返回的图像
        )
        # 应用自定义环境包装器
        env = StreetFighterCustomWrapper(env,reset_round=True, rendering=True)
        # 监控环境
        env = Monitor(env)
        # 设置随机种子
        env.seed(seed)
        return env
    return _init

def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = SubprocVecEnv([make_env(game, state="Champion.Level1.RyuVsGuile", seed=i) for i in range(NUM_ENV)])

    # 定义学习率线性调度器
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # 定义剪裁范围线性调度器
    clip_range_schedule = linear_schedule(0.15, 0.025)

    model = PPO(
        "CnnPolicy", #这是选择的策略网络类型。"CnnPolicy"表示使用的是卷积神经网络，这适合处理像素级的图像输入，比如游戏的屏幕画面
        env,
        device="cuda", 
        verbose=1, #这个参数用于控制训练过程的信息输出。1 表示输出详细信息。
        n_steps=512, #这是采集多少个时间步的经验后进行一次策略更新。
        batch_size=512, #这是每次更新策略时，从采集的经验中随机选取的小批量样本的大小
        n_epochs=4, #这是在每次策略更新时，网络会看几遍所有的训练数据
        gamma=0.94, #这是强化学习中的折扣因子，用于计算未来奖励的现值。gamma越接近1，模型越重视远期的奖励；gamma越小，模型越重视近期的奖励。
        learning_rate=lr_schedule, #这是学习率调度器，它将根据训练的进度动态调整学习率。
        clip_range=clip_range_schedule, #这是剪裁范围调度器，它将根据训练的进度动态调整PPO算法中的剪裁范围
        tensorboard_log="logs"
    )

    # 定义模型保存路径，并创建该路径
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # 创建训练检查点回调函数,每31250步保存一次训练模型
    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_ryu")

    # 定义日志文件，并将训练过程的输出重定向到该文件
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(100000000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[checkpoint_callback]#, stage_increase_callback]
        )
        env.close()

    # 还原标准输出
    sys.stdout = original_stdout

    # 保存最终的模型
    model.save(os.path.join(save_dir, "ppo_sf2_ryu_final.zip"))

if __name__ == "__main__":
    main()
