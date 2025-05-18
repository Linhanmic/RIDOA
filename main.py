import ridoa

def configure_doa_system():
    """配置DOA系统参数
    
    返回:
        DOAConfig: 配置好的DOA配置对象
    """
    # 获取单例配置对象
    config = ridoa.DOAConfig.get_instance()
    
    # 硬件加速设置
    config.useGPU = True          # 启用GPU加速
    
    # 阵列参数设置
    config.nElements = 10         # 阵元数量
    config.samplingRate = 1000000 # 采样率(Hz)
    
    # 算法参数设置
    config.forwardSmoothingSize = 0  # 前向平滑窗口大小(0表示禁用)
    config.maxSources = 2            # 最大信源数
    config.estimateRate = 10000      # 估计速率(Hz)
    
    return config


def generate_test_signal(doa_system):
    """生成模拟测试信号
    
    参数:
        doa_system: DOA系统对象
    
    返回:
        ndarray: 生成的信号数据
    
    异常:
        ValueError: 当信号生成失败时抛出
    """
    # 设置目标角度参数
    elevations = [30.0, 45.0]  # 俯仰角(度)
    azimuths = [60.0, 90.0]    # 方位角(度)
    duration = 1.0             # 信号时长(秒)
    snr_db = 20.0              # 信噪比(dB)

    print("[信号生成] 正在生成测试信号...")
    signal_data = doa_system.generate_simulation_data(
        elevations, azimuths, duration, snr_db
    )

    # 验证信号数据有效性
    if signal_data.shape[0] == 0 or signal_data.shape[1] == 0:
        raise ValueError("错误：信号数据生成失败，返回空数组")
    
    print(f"[信号生成] 成功生成信号数据，维度: {signal_data.shape}")
    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        print(f"信源{i+1}: 俯仰角={elev}°，方位角={azim}°")
    return signal_data


def estimate_and_display_results(doa_system, signal_data):
    """执行DOA估计并显示结果
    
    参数:
        doa_system: DOA系统对象
        signal_data: 输入信号数据
    
    返回:
        DOAResult: 估计结果对象
    """
    print("[DOA估计] 正在进行波达方向估计...")
    
    # 执行DOA估计，第二个参数指定信源数量
    result = doa_system.estimate_doa(signal_data, 2)
    
    print("[DOA估计] 估计完成!")

    # 格式化输出估计结果
    print("估计结果:")
    print(f"俯仰角: {', '.join(f'{elev:.1f}°' for elev in result.estElevations)}")
    print(f"方位角: {', '.join(f'{azim:.1f}°' for azim in result.estAzimuths)}")
    
    return result


def main():
    """主程序入口"""
    try:
        # ========== 系统配置阶段 ==========
        print("\n=== 系统配置阶段 ===")
        config = configure_doa_system()
        print("[系统配置] DOA系统配置完成")
        config.print()
        
        # ========== 系统初始化阶段 ==========
        print("\n=== 系统初始化阶段 ===")
        print("[系统初始化] 正在创建DOA系统...")
        doa_system = ridoa.DOASystem()
        print("[系统初始化] DOA系统创建成功")

        # ========== 信号处理阶段 ==========
        print("\n=== 信号处理阶段 ===")
        signal_data = generate_test_signal(doa_system)

        # ========== DOA估计阶段 ==========
        print("\n=== DOA估计阶段 ===")
        estimate_and_display_results(doa_system, signal_data)

    except Exception as e:
        # 异常处理
        print(f"\n[错误] 程序执行异常: {type(e).__name__}: {e}")
        print("异常堆栈信息:")
        import traceback
        traceback.print_exc()
        return 1  # 返回非零表示错误
    
    return 0  # 返回0表示成功执行


if __name__ == "__main__":
    # 执行主程序并返回系统状态码
    exit_code = main()
    print(f"\n程序执行结束，返回码: {exit_code}")
    exit(exit_code)