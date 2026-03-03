"""
启动 GauzRag FastAPI 服务
"""
import sys
import os
from pathlib import Path


def main():
    """启动服务"""
    project_root = Path(__file__).resolve().parent
    
    # 确保项目根目录在 Python 路径中
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 设置环境变量，供 create_app 使用
    os.environ["GAUZRAG_PROJECT_ROOT"] = str(project_root)
    
    from GauzRag.config import GauzRagConfig
    
    # 加载配置验证
    config = GauzRagConfig.from_env(project_root / ".env")
    config.project_root = project_root
    
    if not config.validate():
        print("\n配置验证失败，请检查 .env 文件")
        return
    
    # 启动信息
    print("\n" + "="*80)
    print("GauzRag API 服务启动中...")
    print("="*80)
    print(f"\n服务地址: http://0.0.0.0:1234")
    print(f"API 文档: http://0.0.0.0:1234/docs")
    print(f"项目根目录: {project_root}")
    print("\n按 Ctrl+C 停止服务\n")
    print("="*80 + "\n")
    
    # 使用 uvicorn 直接运行
    import uvicorn
    uvicorn.run(
        "GauzRag.api:create_app",
        host="0.0.0.0",
        port=1235,
        reload=True,  # 关闭热重载，避免性能测试时服务重启
        factory=True
    )


if __name__ == "__main__":
    main()

