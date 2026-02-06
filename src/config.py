import argparse
import sys # 
from isaaclab.app import AppLauncher

def initialize_app(description: str = "Custom Scene", arg_parser_callback=None):
    """
    [功能]: 初始化 Isaac Lab 应用
    """
  
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

    parser = argparse.ArgumentParser(description=description)
    
    if arg_parser_callback:
        arg_parser_callback(parser)
        
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    
    # [调试] 打印 headless 状态
    headless_mode = getattr(args_cli, 'headless', False)
    print(f"[Config] Headless 模式: {headless_mode}")
    print(f"[Config] sys.argv: {sys.argv}")
    
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    return simulation_app, app_launcher, args_cli

def configure_simulation(app_launcher):
    """
    [功能]: 配置仿真上下文
    [输入]: app_launcher
    [输出]: sim (SimulationContext)
    """
    from isaaclab.sim import SimulationContext, SimulationCfg
    import carb
 
    sim_cfg = SimulationCfg(device="cuda:0", use_fabric=True)
 
    sim = SimulationContext(cfg=sim_cfg)
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])

    carb.settings.get_settings().set_bool("/persistent/physics/resetOnStop", True)
    carb.settings.get_settings().set_bool("/physics/autoPopupSimulationOutputWindow", False)
    carb.settings.get_settings().set_bool("/physics/fabric/useCuda", True)
    carb.settings.get_settings().set_bool("/persistent/physics/visualizationSimulationOutput", False)
    carb.settings.get_settings().set_bool("/rtx/post/dlss/enabled", False)
    
    # ========== 物理稳定性参数（防止崩飞） ==========
    # 1. 限制穿透修正速度（最重要）- 降低可以减少弹飞
    carb.settings.get_settings().set_float("/physics/maxDepenetrationVelocity", 0.5)
    
    # 2. 增加求解器迭代次数（更准确的碰撞检测）
    carb.settings.get_settings().set_int("/physics/numPositionIterations", 8)  # 默认4
    carb.settings.get_settings().set_int("/physics/numVelocityIterations", 4)  # 默认1
    
    # 3. 减小碰撞偏移量（更精确的碰撞）
    carb.settings.get_settings().set_float("/physics/contactOffset", 0.002)  # 2mm
    carb.settings.get_settings().set_float("/physics/restOffset", 0.001)     # 1mm
    
    # 4. 设置最大线性速度限制（防止物体飞太快）
    carb.settings.get_settings().set_float("/physics/maxLinearVelocity", 5.0)  # 5 m/s

    return sim
