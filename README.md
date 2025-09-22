# R-INN 论文复现项目 README​
## 项目概述​
本仓库用于复现论文《R-INN: An Efficient Reversible Design Model for Microwave Circuit Design》中的可逆神经网络（Real NVP-based Invertible Neural Network）模型。该模型将可逆神经网络应用于微波电路设计，通过学习电路参数与电磁响应之间的映射关系，实现高效的电路设计与优化。​
复现目标包括：​
实现论文提出的 R-INN 网络结构​
复现关键实验结果（如 NMSE 误差、S 参数拟合曲线）​
建立可复用的微波电路设计深度学习工作流


# 环境配置

Created by: 卓杭 田
Created time: September 22, 2025 5:19 PM
类别: 代码
Last edited by: 卓杭 田
Last updated time: September 22, 2025 8:28 PM

- python配置
    
    # 打开anaconda输入以下内容
    
    ```python
    conda create -n r-inn-env python=3.12
    conda activate r-inn-env
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    conda install numpy pandas scipy scikit-learn h5py matplotlib seaborn tqdm
    python -c "import sys; print(sys.executable)"
    ```
    
- Git网络配置：
    
    ### **解决方案二：手动配置Git代理（特别是用VPN/代理的朋友，这里是重点！）**
    
    如果你是命令行死忠粉，或者必须依赖 VPN/代理上网，那么手动配置 Git 代理就是你的必修课了。
    
    Git 可以通过全局配置来设置 HTTP/HTTPS 代理：
    
    ```bash
    # 设置 HTTP 代理
    git config --global http.proxy http://127.0.0.1:你的代理端口号
    
    # 设置 HTTPS 代理
    git config --global https://127.0.0.1:你的代理端口号
    
    ```
    
    **敲黑板！为什么网上很多教程你跟着做却没用？**
    
    问题就出在那个 `你的代理端口号` 上！网上大部分教程会直接给个 `7890` 或者 `1080`，但**你的 VPN 或代理软件用的端口，很可能跟它们不一样！** 你不能直接抄作业。
    
    **怎么找到你自己的“专属”代理端口号？**
    
    这才是解决问题的核心！通常，你的 VPN 或代理软件都会在设置界面里明明白白地告诉你它正在监听哪个端口。
    
    - **去代理工具里找：** 打开你正在用的代理软件（比如 Clash、V2RayN等），进到设置页面，找“代理端口”、“HTTP代理端口”或者类似字眼，那个数字就是你的真命天子！（田卓杭的例子是7890）
        
        ![image.png](image.png)
        
        - **举个栗子：** 我自己的代理，它的端口就是 `17890`。所以，我的命令是这样：
            
            ```bash
            git config --global http.proxy http://127.0.0.1:17890
            git config --global https.proxy https://127.0.0.1:17890
            
            ```
            
    - **从电脑系统设置里看：**
        - **Windows：** “设置” -> “网络和 Internet” -> “代理”，看看“手动设置代理”那里写的是啥端口。
        - **macOS：** “系统设置” -> “网络” -> 选中当前连接的网络服务 -> “详细信息” -> “代理”选项卡。
    
    找到你的私人端口号，替换掉上面的 `你的代理端口号`，再跑一遍命令，基本上问题就迎刃而解了！
    
    - -
 

