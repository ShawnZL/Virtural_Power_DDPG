# main.py

该代码实现了使用NEAT算法（神经进化增强拓扑）来训练递归神经网络（Recurrent Network）来控制价格响应性负载以优化电价对负载的影响。

主要步骤如下：

1. `PriceResponsiveLoad`类定义了一个价格响应性负载的模型，并且可以根据价格水平和时间来动态调整负载。
2. `evaluate_network`函数用于评估递归神经网络（Recurrent Network）在环境中的性能。它使用`gym`库创建环境，然后模拟在环境中运行负载控制，并计算负载控制在一年内的平均奖励作为性能指标。
3. `neat_fitness_function`函数用于计算每个基因组的适应度（fitness）。它使用多进程（`Pool`）来并行地计算每个基因组的适应度，并将适应度值分配给每个基因组。
4. `evaluate_genome`函数用于对每个基因组进行评估，并返回其适应度值。
5. `main`函数是整个训练过程的主函数。它首先定义了NEAT算法的配置参数（`neat_config`），然后使用这些配置参数创建了一个`Evolution`实例（`evolution`）来进行遗传进化训练。最后，它调用`evolution.run`方法来执行训练，直到达到指定的适应度目标或指定的迭代次数。
6. `draw_species_graph`函数用于绘制训练过程中物种数量的变化图。

请注意，由于代码中使用了一些自定义的模块和库，其中的一些细节可能无法完全理解。如果需要更深入地了解代码的功能和执行过程，需要查看其他自定义模块和库的代码实现。

# custom_envs

## custom_cartpole

### envs

#### custom_cartpole_env.py

这个代码实现了一个 CartPole-v1 环境，并定义了一个名为 `CustomCartPoleEnv` 的类。该类继承了 `gym.Env` 类，用于定义自定义的 CartPole-v1 环境。

CartPole-v1 是 OpenAI Gym 中的一个经典控制问题，目标是使得一个连接在推车上的摆杆保持平衡。在这个环境中，摆杆上方的力点不受控制，推车可以向左或向右施加恒定的力。目标是通过施加合适的力来保持摆杆垂直，尽可能长时间地保持平衡。

这个自定义环境中的 CartPole-v1 问题的动作空间（Action Space）是一个长度为 1 的 ndarray，可以采取值 `{0, 1}`，其中 `0` 表示向左推车，`1` 表示向右推车。

状态空间（Observation Space）是一个长度为 4 的 ndarray，包含了以下四个元素：

1. 推车位置（Cart Position）
2. 推车速度（Cart Velocity）
3. 摆杆角度（Pole Angle）
4. 摆杆角速度（Pole Angular Velocity）

每个元素的取值范围不同，具体取值范围如下：

| Num  | Observation           | Min                   | Max                 |
| ---- | --------------------- | --------------------- | ------------------- |
| 0    | Cart Position         | -4.8                  | 4.8                 |
| 1    | Cart Velocity         | -∞                    | ∞                   |
| 2    | Pole Angle            | ~ -0.418 rad (~ -24°) | ~ 0.418 rad (~ 24°) |
| 3    | Pole Angular Velocity | -∞                    | ∞                   |

自定义 CartPole-v1 环境的 reset 方法会将所有观测值初始化为均匀分布在 (-0.05, 0.05) 范围内的随机值。

step 方法用于执行一个动作并返回新的状态、奖励、是否终止、以及附加信息。

render 方法用于可视化环境状态，可以选择显示为图像（render_mode="rgb_array"）或显示在屏幕上（render_mode="human"）。这里使用了 pygame 库来实现图像渲染。如果 render 方法未指定 render_mode，则会显示警告信息。

close 方法用于关闭环境的显示窗口，释放相关资源。

这个自定义环境提供了一个实现 CartPole-v1 问题的接口，可以用于创建自定义的 CartPole-v1 环境，并在 Gym 中进行训练和测试。

## custom_envs.egg-info

### SOURCES.txt

当然，以下是每个文件和文件夹的简要描述：

1. `setup.py`：这是Python包的安装配置脚本。它包含构建、分发和安装包所需的信息。

2. `custom_cartpole/`：这是Python包的根目录。它包含`custom_cartpole`包的实现代码。

3. `custom_cartpole/__init__.py`：这是`custom_cartpole`包的初始化文件。在导入包时执行此文件。

4. `custom_cartpole/envs/`：这是`custom_cartpole`包的子目录，包含与环境相关的代码。

5. `custom_cartpole/envs/__init__.py`：这是`custom_cartpole.envs`模块的初始化文件。在导入模块时执行此文件。

6. `custom_cartpole/envs/custom_cartpole_env.py`：这是自定义CartPole环境的主要实现文件。它定义了`CustomCartPoleEnv`类，代表自定义的CartPole-v1环境。

7. `custom_envs.egg-info/`：这个文件夹包含有关Python包的元数据信息，包括包的版本、依赖项和其他相关信息。

8. `custom_envs.egg-info/PKG-INFO`：这个文件包含有关包的元数据信息。

9. `custom_envs.egg-info/SOURCES.txt`：这个文件列出了包含在包中的源文件。

10. `custom_envs.egg-info/dependency_links.txt`：这个文件可能包含指向包所需依赖项的链接。

11. `custom_envs.egg-info/requires.txt`：这个文件可能列出了所需的Python包及其版本。

12. `custom_envs.egg-info/top_level.txt`：这个文件列出了包中的顶层模块。

13. `grid_v0/__init__.py`：这是`grid_v0`模块的初始化文件，似乎是Python包的另一部分。

所提供的文件和文件夹似乎构成了一个自定义的Python包，其中包含了自定义的CartPole-v1环境的实现。可以使用`setup.py`脚本安装此包，并在Gym或其他强化学习框架中使用自定义的CartPole环境进行训练和测试。

## grid_v0

### __init__.py

这段代码使用`gym.envs.registration.register`函数注册一个名为`Grid-v0`的新环境。以下是每个参数的解释：

1. `id`: 新环境的唯一标识符。在调用`gym.make()`函数创建环境时，可以使用该标识符来引用这个新环境。

2. `entry_point`: 新环境的入口点，即包含自定义环境类的模块路径。在这种情况下，环境类`GridV0Env`位于`custom_envs.grid_v0.envs`模块中。

3. `max_episode_steps`: 最大的回合步数。这定义了每个回合的最大允许步数，以防止回合过长。

通过使用这个`register()`函数，新环境`Grid-v0`就可以在Gym中使用，并通过`gym.make('Grid-v0')`来创建一个实例。

### envs

#### __init__.py

这行代码导入了定义在`custom_envs.grid_v0.envs.grid_v0_env`模块中的`GridV0Env`类。根据你之前提供的代码，`GridV0Env`是一个自定义环境类，它继承自`gym.Env`类，并实现了必要的`reset`、`step`、`render`和`close`函数。现在可以使用`GridV0Env`类来创建自定义的Grid-v0环境的实例。

#### grid_v0_env.py

这段代码定义了一个名为`GridV0Env`的OpenAI Gym环境。该环境是用于微电网仿真的自定义环境，继承自`gym.Env`类，并实现了必要的`reset`、`step`、`render`和`close`函数。

环境使用`spaces`定义了观察空间（`observation_space`）和动作空间（`action_space`）。观察空间是一个元组，包含了六个连续的浮点数，一个离散的整数和另一个离散的整数。动作空间是一个`MultiDiscrete`空间，其中包含四个离散子空间，分别对应TCL动作、价格动作、能量缺乏动作和能量过剩动作。

`step`函数接收一个动作作为输入，并返回新的状态和奖励。状态包含了TCL SoC、ESS SoC、室外温度、产生的能量、上行价格和基本住宅负荷等六个连续的浮点数，以及定价计数器和一天中的小时数等两个离散整数。

`reset`函数用于将环境重置为初始状态。



这是一个用于模拟微电网的OpenAI Gym环境。该环境的名称为`Grid-v0`，最大步数为24。

该环境的初始化函数`__init__`接受一个参数`max_total_steps`，用于设置最大步数。

该环境的状态空间包含三个子空间：
1. `spaces.Box`：包含6个浮点数，对应以下内容：
   - TCL SoC (State of Charge)：充电控制器的电池电量状态（范围：0.0至1.0）
   - ESS SoC (State of Charge)：储能系统的电池电量状态（范围：0.0至1.0）
   - 外部温度：温度（范围：-22.0至32.0）
   - 产生的能量：能量生成量（范围：0.0至1800.0）
   - 上涨价格：能源价格上涨（范围：0.0至2999.0）
   - 基本住宅负荷：基本住宅电负荷（范围：0.0至1.4，默认最大值为1.4）

2. `spaces.Discrete`：对应定价计数器（范围：-48至28）

3. `spaces.Discrete`：对应一天中的小时数（范围：0至23）

该环境的动作空间为一个`spaces.MultiDiscrete`，包含4个离散动作子空间：
- TCL动作：包含4个离散动作（范围：0至3）
- 价格动作：包含5个离散动作（范围：0至4）
- 能量不足动作：包含2个离散动作（范围：0至1）
- 能量过剩动作：包含2个离散动作（范围：0至1）

在每个步骤中，`step`函数接收一个动作（`spaces.MultiDiscrete`类型），然后返回新的状态、奖励、是否终止、是否需要渲染和其他信息。

`reset`函数用于重置环境的初始状态，接受一个可选的`seed`参数用于设置随机种子。在重置后，环境将重新获取状态。

`render`和`close`函数用于可视化和关闭可视化，但在该环境中未使用。



在`__init__`函数中，构造函数接收一个参数`max_total_steps`，该参数表示环境中允许的最大步数。在初始化过程中，首先保存`max_total_steps`到实例变量`self._max_total_steps`中。

然后，通过获取项目目录，拼接上"data"字符串，得到数据路径`self._data_path`。接下来，通过随机生成一个起始索引`start_idx`，以确保每次训练的起始状态是随机的。然后使用`get_default_microgrid_env`函数创建一个名为`self._env`的微网仿真环境。

初始化中还设置了状态变量`self.state`为`None`，以及步数变量`self._step`为0。

接下来，定义了观察空间`observation_space`和动作空间`action_space`。观察空间是一个元组，包含了六个连续的浮点数范围和两个离散的整数范围。连续浮点数范围对应了TCL SoC、ESS SoC、室外温度、产生的能量、上行价格和基本住宅负荷。离散整数范围对应了定价计数器和一天中的小时数。

动作空间是一个`MultiDiscrete`空间，包含了四个离散子空间。第一个子空间有4个动作，对应TCL动作；第二个子空间有5个动作，对应价格动作；第三个和第四个子空间各有2个动作，对应能量缺乏动作和能量过剩动作。



在`step`函数中，接收`action`作为输入，返回新的状态、奖励和是否终止的信息。

首先，该函数检查输入的动作是否在动作空间中。如果不在动作空间中，会抛出一个错误。然后，检查当前状态是否为空，如果为空说明还没有调用`reset`函数，此时会抛出一个错误，提示在使用`step`函数前需要先调用`reset`函数。

接下来，将`action`转换为一个四元组`action_tuple`，其中第一个元素是TCL动作，第二个元素是价格动作，第三个元素是能量缺乏动作，第四个元素是能量过剩动作。然后，调用`self._env.step(action_tuple)`执行环境的一步仿真，并更新状态和奖励。

然后，步数加1，并判断是否达到了最大允许步数。如果超过了最大允许步数，会发出一个警告。最后，返回一个包含新的状态、奖励、终止状态、是否需要渲染和额外信息的元组。



这是环境中的`reset`函数，用于将环境重置到初始状态。在这个函数中，首先调用了父类的`reset`方法，以确保环境状态的正确重置。然后，根据给定的`seed`参数随机选择一个起始索引`start_idx`。接着，通过`get_default_microgrid_env`函数从数据路径`self._data_path`中获取一个新的微电网环境。重置环境的状态和步数，并获取新的环境状态。最后，返回一个元组，其中包含了环境状态的一部分（前6个元素）、定价计数器和一天中的小时数。

`options`参数并未在该函数中使用，它只是一个可选的参数字典，允许你向环境传递一些额外的选项。

这个`reset`函数在每个新的回合开始时被调用，以确保环境在每个回合开始时处于相同的初始状态。



## setup.py

这是一个用于设置Python包的`setup.py`文件。在此文件中，你需要提供一些关键信息，以便构建和安装包。

- `name`: 指定包的名称，即在安装时使用的名称。
- `version`: 包的版本号。每次更新代码时，你可以递增版本号，以确保在安装新版本时不会与旧版本冲突。
- `install_requires`: 指定运行该包所需的最小依赖项列表。在这里，指定了需要安装的gym和numpy的版本。
- `packages`: 指定要包含在包中的子包。在这里，包含了名为`custom_cartpole`和`grid_v0`的子包。

此外，还可以在`setup()`函数中提供其他信息，例如作者、描述、许可证等。

在运行这个`setup.py`文件后，你可以使用`pip install .`来安装你的自定义环境包。安装成功后，你就可以在代码中通过`import custom_cartpole`和`import grid_v0`来使用自定义环境。

## utils.py

这是一组用于经典控制环境的实用函数。它们用于验证和转换输入参数，并在重置环境时设置初始状态的采样范围。

1. `verify_number_and_cast(x: SupportsFloat) -> float`: 这个函数用于验证参数`x`是否是一个可转换为浮点数的单一数字，并将其转换为浮点数。如果参数不是一个数字或无法转换为浮点数，则会抛出一个`ValueError`异常。

2. `maybe_parse_reset_bounds(options: Optional[dict], default_low: float, default_high: float) -> Tuple[float, float]`: 这个函数用于解析重置环境时传入的选项`options`，并返回初始状态采样范围的下界和上界。如果`options`为`None`，则会使用默认的下界`default_low`和上界`default_high`。如果`options`中指定了`low`和`high`选项，则会使用这些值。函数还会调用`verify_number_and_cast`函数来确保下界和上界是合法的数字，并确保下界小于上界。最终，函数会返回一个包含下界和上界的元组。 



# neat

## config.py

`NeatParams`类是一个用于存储NEAT算法（神经进化增强拓扑）的特定参数的数据类（dataclass）。

在Python中，`dataclass`是一个装饰器，它自动为类生成一些基本的特殊方法，例如`__init__`、`__repr__`、`__eq__`等。通过使用`dataclass`装饰器，我们可以简化类的定义，并且不需要显式地编写这些特殊方法。

在`NeatParams`类中，定义了许多与NEAT算法相关的参数，包括种群大小、物种繁殖率、物种最小规模、最大停滞代数、存活的精英物种数量等等。这些参数用于配置NEAT算法的执行过程，以控制种群进化的方式。

例如，`population_size`指定了种群的大小，`repro_survival_rate`指定了繁殖时保留的物种中前多少比例的个体，`min_species_size`指定了一个物种最少有多少个体，`max_stagnation`指定了在停滞多少代后允许进行种群清理，等等。

这样，通过定义`NeatParams`类，我们可以方便地组织和存储所有与NEAT算法相关的参数，使得代码更加清晰和易于维护。

## evolution.py

`Evolution`类负责协调和执行NEAT算法，即神经进化增强拓扑算法。

主要属性：
- `_neat_params`：存储NEAT算法的参数的实例，即`NeatParams`类的对象。
- `generation`：代表当前的进化代数。
- `population`：存储当前种群中的所有基因组。
- `reproduction`：负责进行繁殖和进化的实例，即`Reproduction`类的对象。
- `species_set`：存储当前物种集合的实例，即`SpeciesSet`类的对象。
- `species_history`：记录每一代中每个物种的信息，以便后续可视化。
- `best_genome`：存储当前进化过程中表现最好的基因组。

主要方法：
- `__init__(...)`：初始化`Evolution`类的实例，设置初始种群和物种。
- `run(...)`：执行NEAT算法的主要循环，进行种群的进化。通过调用`fitness_function`来计算每个基因组的适应度，并更新物种的信息。根据适应度来选择新的基因组，执行繁殖过程，并更新物种集合。
- `_get_best_genome()`：获取当前种群中适应度最高的基因组。

在运行`run(...)`方法后，NEAT算法会根据设定的终止条件（例如达到目标适应度或达到最大代数）停止，并返回表现最好的基因组。

总体上，`Evolution`类是一个用于执行NEAT算法的主要控制器，并通过协调`Reproduction`和`SpeciesSet`类的操作来完成种群进化和物种划分。

## reproduction.py

`Reproduction`类负责处理NEAT的繁殖（创建新的基因组），包括基因组的突变。

主要属性：
- `num_inputs`：输入节点数。
- `num_outputs`：输出节点数。
- `neat_params`：存储NEAT算法的参数的实例，即`NeatParams`类的对象。
- `_weight_options`：权重选项，包含初始化权重和调整权重的相关参数。
- `_bias_options`：偏置选项，包含初始化偏置和调整偏置的相关参数。
- `_mutate_params`：突变参数，包含各类突变的相关概率和选项。
- `species_fitness_function`：物种适应度函数，用于计算物种的适应度。

主要方法：
- `__init__(...)`：初始化`Reproduction`类的实例，并设置相关参数。
- `create_new_population(...)`：创建完全随机化的新种群，其中的基因组是基于随机化的最小基因组生成的。
- `reproduce(...)`：根据NEAT算法创建新一代基因组，包括基因组的交叉和突变过程。首先通过适应度进行选择和调整，然后计算每个物种的产生后代数量，接着从物种中选择合适的基因组作为父代，最后执行交叉和突变操作生成新的基因组。
- `_adjust_genome_fitnesses_for_species(...)`：对物种中的每个基因组的适应度进行调整，以实现在物种内部使用共享适应度的效果。
- `_get_stagnant_species(...)`：判断物种是否处于停滞状态，如果物种在一定代数内没有进化，则判定为停滞状态。
- `_get_adjusted_fitnesses(...)`：计算每个物种的调整适应度，用于后续的选择和产生后代。
- `_compute_spawn_amounts(...)`：计算每个物种应该产生的后代数量，根据适应度进行比例分配。
- `_select_genomes_for_reproduction(...)`：选择作为父代的基因组，并将其中表现最好的基因组直接保留，而其余的作为父代进行后续交叉和突变。
- `_spawn_offspring(...)`：产生新的后代基因组，通过随机选择两个父代，进行交叉和突变操作来生成新的基因组。

总体上，`Reproduction`类负责处理基因组的繁殖和突变过程，是NEAT算法中最关键的部分之一，通过基因组的交叉和突变操作产生新的后代基因组，并将其加入到新的种群中。

## nn

### graphs.py

这些代码涉及三个与有向图相关的函数：

1. `creates_cycle(connections, test)`: 此函数确定在将由 `test` 表示的新连接添加到图中（由 `connections` 列表表示）是否会创建一个环。如果添加新连接将创建一个环，则返回 `True`，否则返回 `False`。

2. `required_for_output(inputs, outputs, connections)`: 此函数收集计算最终网络输出所需的节点。它接受 `inputs` 列表，表示输入节点的标识符，`outputs` 列表，表示输出节点的标识符，以及 `connections` 列表，表示网络中的连接。该函数返回一个节点标识符的集合，这些节点用于计算最终的输出。

3. `feed_forward_layers(inputs, outputs, connections)`: 此函数在有向图中收集可以并行计算的层。它接受 `inputs` 列表，表示网络输入节点，`outputs` 列表，表示输出节点的标识符，以及 `connections` 列表，表示网络中的连接。该函数返回一个层的列表，每个层由一组节点标识符组成。这些层表示可以以前馈方式并行计算的节点，而不会创建任何环路。

### recurrent.py

上述代码实现了一个递归神经网络（Recurrent Network），其中包括以下主要部分：

1. `required_for_output(inputs, outputs, connections)`: 此函数是先前提到的函数的一个改进版本，用于收集计算最终网络输出所需的节点。它接受输入节点的标识符列表 `inputs`、输出节点的标识符列表 `outputs` 以及连接的字典 `connections`，表示网络中的连接。该函数返回一个节点标识符的集合，这些节点用于计算最终的输出。

2. `class RecurrentNetwork`: 这是递归神经网络的实现类。它包括初始化方法 `__init__`，用于初始化网络；`reset` 方法，用于重置网络状态；`activate` 方法，用于激活网络并计算输出。此外，还有一个静态方法 `create`，用于从给定的 `Genome` 创建递归神经网络的实例。

3. `activate` 方法：此方法用于激活递归神经网络，并将给定的输入列表传递到网络中以计算输出。它使用 `sigmoid_activation` 激活函数来处理节点的输出。

4. `create` 静态方法：此方法从给定的 `Genome` 创建一个递归神经网络实例。在这里，通过遍历连接来构建递归神经网络的拓扑结构，并且仅使用与输出节点相关的连接。这样可以减少网络的计算量，并且仅考虑输出节点所需的节点。

### test_recurrent.py

这里是一些测试用例来测试 `RecurrentNetwork` 类的功能。测试用例使用 `assert` 语句检查递归神经网络是否按预期工作。

1. `test_unconnected`: 测试未连接的网络，该网络没有输入节点，并且只有一个输出神经元。在这个测试中，创建了一个不连接的神经网络，然后激活它两次，检查激活的输出是否正确。

2. `test_basic`: 测试非常简单的网络，其中只有一个连接权重为 1 的连接到一个 sigmoid 输出节点。在这个测试中，创建了一个简单的递归神经网络，然后激活它两次，检查激活的输出是否正确。

如果所有测试用例通过（即没有引发`AssertionError`异常），则表示递归神经网络的实现是正确的。

## genetics

### genes.py

这些代码定义了两个数据类：`NodeGene` 和 `ConnectionGene`，用于表示神经网络的节点和连接。此外，还定义了一个枚举类型 `NodeType`，用于跟踪不同类型的节点。

1. `NodeType` 是一个枚举类型，用于表示不同类型的节点。其中 `SENSOR` 节点表示传感器节点，`OUTPUT` 节点表示输出节点，`HIDDEN` 节点表示隐藏节点。

2. `NodeGene` 是一个数据类，表示神经网络的节点基因。它具有以下属性：
   - `idx`: 节点的唯一标识符。
   - `node_type`: 节点的类型，使用 `NodeType` 枚举表示。
   - `bias`: 节点的偏置。

3. `ConnectionGene` 是一个数据类，表示神经网络的连接基因。它具有以下属性：
   - `node_in_idx`: 输入节点的标识符。
   - `node_out_idx`: 输出节点的标识符。
   - `weight`: 连接的权重。
   - `enabled`: 连接是否启用。
   - `innovation_num`: 连接的创新编号。

此外，`ConnectionGene` 类还有几个方法：
   - `crossover(self, other_conn: "ConnectionGene",  keep_disable_prob: float) -> "ConnectionGene"`：与另一个连接基因进行交叉，生成一个新的连接基因。
   - `distance(conn_1: "ConnectionGene", conn_2: "ConnectionGene") -> float`：计算两个连接基因之间的距离，使用连接权重的差异来表示距离。

### genome.py

这些代码实现了遗传算法的关键部分，用于进行神经网络的进化和变异。以下是一些主要组件的介绍：

1. `WeightOptions`: 用于配置连接权重的初始化和变异选项的数据类。它包含初始化的均值和标准差，最大调整量，以及权重的最小值和最大值。可以使用 `get_new_val()` 方法来获取新的随机权重，使用 `adjust(old_val)` 方法来调整现有的权重。

2. `MutationParams`: 用于配置遗传算法中突变的概率和参数的数据类。它包含添加节点、添加连接、调整权重和偏置等概率，以及 `WeightOptions` 和 `bias_options` 对象用于连接权重和偏置的配置。

3. `Innovations`: 用于跟踪当前一代中创新的数据类。在繁殖过程中，会产生新的连接和节点，这些新的连接和节点的创新编号会记录在 `Innovations` 对象中。

4. `Genome`: 代表神经网络的基因组，是遗传算法进化过程中的一个个体。它包含输入节点、输出节点、节点基因和连接基因等信息。

   - `create_new()`: 用于创建一个全新的基因组，随机初始化连接权重，没有隐藏节点。
   - `from_crossover()`: 通过交叉（融合）两个父代基因组创建一个新的子代基因组。
   - `mutate()`: 对基因组进行突变，可能增加节点、添加连接，调整权重和偏置等。
   - `genome_distance()`: 计算两个基因组之间的差异度，用于在遗传算法中选择合适的个体。

在遗传算法中，通过随机的交叉和变异操作来不断改进和进化神经网络，从而逐步优化神经网络的性能。这些组件共同构成了一个遗传算法的框架，用于生成新的神经网络，并选择适应度较高的网络进行进化和繁殖。

### species.py

这些代码实现了基于种群中个体之间相似度的遗传算法中的物种（Species）管理。在遗传算法中，物种用于将种群划分为不同的群体，每个群体称为一个物种，其中个体（基因组）之间的相似度较高。

以下是物种管理的主要组件：

1. `Species`: 代表一个物种，用于跟踪该物种的代表（代表基因组）、成员（属于该物种的所有基因组）、适应度（平均适应度和适应度历史记录等）等信息。在每代进化过程中，物种会更新其成员和代表。

2. `DistanceCache`: 用于缓存基因组之间的相似度（距离），避免重复计算。这里的相似度由 `Genome.genome_distance()` 方法计算得出，其中包含节点差异度和连接权重差异度。

3. `SpeciesSet`: 用于物种的创建和更新。它会根据一定的相似度阈值（compatibility_threshold）将种群中的基因组划分为不同的物种。

主要过程如下：

1. `speciate()`: 将整个种群根据相似度阈值划分为不同的物种。这个过程会找出每个物种的新代表和成员。

2. `_get_new_representatives()`: 为每个物种找到新的代表基因组。新的代表是上一代代表周围距离最近的基因组。

3. `_partition_to_species()`: 将未分配物种的基因组根据距离分配到相应的物种。

4. `_get_candidate_species()`: 获取候选物种，检查未分配物种的基因组是否与现有物种的代表相似。

5. `_update_collections()`: 更新物种的成员和代表，同时更新基因组到物种的映射关系 `_genome_to_species`。

通过物种管理，遗传算法可以维护多样性，并鼓励种群中更多的探索，提高算法的全局搜索能力。

# microgrid_sim

## components

### components.py

这里是与`Environment`类相关的`Components`类，以及用于创建`Components`对象的函数`get_components_by_param_dicts`。

1. `Components`类：这是一个帮助类，用于管理微电网环境中的各个组件。它包含了主电网（`MainGrid`）、分布式能源资源（DER，`DER`）、储能系统（ESS，`ESS`）、居民家庭管理器（`HouseholdsManager`）和热控装置聚合器（TCL聚合器，`TCLAggregator`）。这个类的作用是将所有组件打包在一起，以便在环境中进行更方便的交互和管理。

   `Components`类是一个辅助类，用于管理环境的各个组件。该类包含了以下属性和方法：

   - 属性：
     - `main_grid`：主电网对象（MainGrid），表示环境中的主电网。
     - `der`：分布式能源资源对象（DER），表示环境中的分布式能源资源。
     - `ess`：储能系统对象（ESS），表示环境中的储能系统。
     - `households_manager`：家庭管理器对象（HouseholdsManager），表示环境中的家庭管理器。
     - `tcl_aggregator`：TCL聚合器对象（TCLAggregator），表示环境中的TCL聚合器。

   - 方法：
     - `get_hour_of_day(self, idx: int) -> int`：获取指定索引（`idx`）对应的小时数。该方法是一个简单的包装器，用于通过调用DER组件的`get_hour_of_day`方法来获取指定索引对应的小时数。
     - `get_outdoor_temperature(self, idx: int) -> float`：获取指定索引（`idx`）对应的室外温度。该方法是一个简单的包装器，用于通过调用TCL聚合器组件的`get_outdoor_temperature`方法来获取指定索引对应的室外温度。

   该类的主要目的是将环境中各个组件进行封装，提供更方便的访问方式，同时避免直接访问各个组件的内部属性和方法，从而简化代码的编写和维护。

2. `get_components_by_param_dicts()`函数：这是一个工厂函数，用于根据参数字典创建`Components`对象。它接收五个参数，分别是TCL参数字典、ESS参数字典、主电网参数字典、DER参数字典和居民家庭负载参数字典。这个函数内部会调用其他工厂函数（如`get_tcl_aggregator_from_params_dict`、`get_ess_from_params_dict`等）来根据参数字典创建各个组件对象，并最终返回一个`Components`对象，其中包含了这些组件。

这个`Components`类的设计和`get_components_by_param_dicts()`函数的作用是为了将微电网环境的各个组件封装在一起，并提供一种简单的方式来根据参数字典创建这些组件对象。在`Environment`类的构造函数中，就是通过调用`get_components_by_param_dicts()`来创建微电网环境的各个组件。这种设计的好处是，让`Environment`类的构造函数保持简洁，而具体的组件创建过程则在`get_components_by_param_dicts()`函数中实现，使得代码更易于维护和扩展。

### der.py

这里有关于分布式能源资源（DER）的相关代码。

1. `DERParams`类：这是一个数据类，用于存储与DER相关的参数。它有两个属性：`hourly_generated_energies_file_path`和`generation_cost`。`hourly_generated_energies_file_path`是一个字符串，表示包含每小时发电量数据的CSV文件的路径。`generation_cost`是一个浮点数，表示发电成本，默认值为0.032。此类还有一个`from_dict`方法，它接收一个参数`der_params_dict`，是一个包含DER参数的字典。**`from_dict`方法将从字典中提取数据来创建`DERParams`对象。**

2. `DER`类：这是一个模拟分布式能源资源（DER）的类，例如一组风力涡轮机。该实现从提供的CSV文件中读取数据，并在请求时逐个返回值。它有以下属性：

   - `_data`：存储从CSV文件中读取的能量发电数据的Pandas DataFrame。
   - `generation_cost`：表示发电成本的浮点数。

   类有三个方法：

   - `from_params`：该方法根据传入的`DERParams`参数创建`DER`对象。它会从`DERParams`参数中读取CSV文件的路径，然后使用Pandas读取数据并创建`DER`对象。
   - `get_generated_energy`：该方法接收一个索引`idx`，返回在该索引处得到的发电量数据（float）。
   - `get_data_size`：该方法返回能量发电数据的长度（DataFrame的行数）。
   - `get_hour_of_day`：该方法接收一个索引`idx`，从能量发电数据中获取该索引处对应的小时数，并返回该小时数（int）。

这些类和方法共同构成了对分布式能源资源（DER）进行建模和模拟的组件。`DER`类提供了对从CSV文件读取发电量数据的功能，而`DERParams`类则是用于参数传递和配置的数据类。在实际使用中，可以使用`DERParams`类来传递DER的参数，然后通过`DER.from_params()`方法来创建DER对象，并使用`get_generated_energy()`方法来获取发电量数据。

### ess.py

这里有关于能量储存系统（ESS）的相关代码。

1. `ESSParams`类：这是一个数据类，用于存储与ESS相关的参数。它有五个属性：`charge_efficiency`、`discharge_efficiency`、`max_charge`、`max_discharge`和`max_energy`。`charge_efficiency`和`discharge_efficiency`分别表示充电效率和放电效率，默认值都为0.9。`max_charge`和`max_discharge`分别表示最大充电功率和最大放电功率，默认值都为250.0。`max_energy`表示ESS的最大能量容量，默认值为500.0。**此类还有一个`from_dict`方法，它接收一个参数`ess_params_dict`，是一个包含ESS参数的字典。`from_dict`方法将从字典中提取数据来创建`ESSParams`对象。**

2. `ESS`类：这是一个模拟能量储存系统（ESS）的类，例如一个电池。该实现通过模拟充电和放电的过程来更新能量储存系统的状态。它有以下属性：

   - `energy`：表示当前能量（float）。
   - `_max_energy`：表示最大能量容量（float）。
   - `soc`：表示能量储存系统的状态 of charge，即当前能量与最大能量容量之间的比例（float）。此属性是在初始化后自动计算的。
   - `_max_charge_power`：表示最大充电功率（float）。
   - `_max_discharge_power`：表示最大放电功率（float）。
   - `_charge_efficiency`：表示充电效率（float）。
   - `_discharge_efficiency`：表示放电效率（float）。

   类有三个方法：

   - `from_params`：该方法根据传入的`ESSParams`参数创建`ESS`对象。它会使用高斯分布来初始化能量，并返回一个`ESS`对象。
   - `charge`：该方法接收一个充电功率`charge_power`，将能量储存系统充电，并返回多余的能量。
   - `discharge`：该方法接收一个放电功率`discharge_power`，将能量从储存系统中提取，并返回提供的能量。
   - `_update`：这是一个内部方法，用于更新能量储存系统的状态。它会根据充电功率和放电功率来更新能量，并计算多余或提供的能量。
   - `_get_limited_charge_power`：这是一个内部方法，用于获取受限制的充电功率。它将根据最大充电功率、储存系统剩余能量和充电效率来计算限制后的充电功率。
   - `_get_limited_discharge_power`：这是一个内部方法，用于获取受限制的放电功率。它将根据最大放电功率、储存系统当前能量和放电效率来计算限制后的放电功率。

这些类和方法共同构成了对能量储存系统（ESS）进行建模和模拟的组件。`ESS`类提供了对能量储存系统的模拟，并且可以通过`charge`和`discharge`方法来进行充电和放电。而`ESSParams`类则是用于参数传递和配置的数据类。在实际使用中，可以使用`ESSParams`类来传递ESS的参数，然后通过`ESS.from_params()`方法来创建ESS对象，并使用`charge()`和`discharge()`方法来模拟能量储存系统的充放电过程。

### from_dict_factories.py

这里提供了一些工厂函数，用于根据传入的参数字典创建不同组件的实例。

1. `get_tcl_aggregator_from_params_dict`：该函数接收一个参数字典`params_dict`，然后使用`TCLParams.from_dict`方法将字典转换为`TCLParams`对象，并进一步使用`TCLAggregator.from_params`方法创建一个`TCLAggregator`实例。

2. `get_ess_from_params_dict`：该函数接收一个参数字典`params_dict`，然后使用`ESSParams.from_dict`方法将字典转换为`ESSParams`对象，并进一步使用`ESS.from_params`方法创建一个`ESS`实例。

3. `get_main_grid_from_params_dict`：该函数接收一个参数字典`params_dict`，然后使用`MainGridParams.from_dict`方法将字典转换为`MainGridParams`对象，并进一步使用`MainGrid.from_params`方法创建一个`MainGrid`实例。

4. `get_der_from_params_dict`：该函数接收一个参数字典`params_dict`，然后使用`DERParams.from_dict`方法将字典转换为`DERParams`对象，并进一步使用`DER.from_params`方法创建一个`DER`实例。

5. `get_household_manager_from_params_dict`：该函数接收一个参数字典`params_dict`，然后使用`ResidentialLoadParams.from_dict`方法将字典转换为`ResidentialLoadParams`对象，并进一步使用`HouseholdsManager.from_params`方法创建一个`HouseholdsManager`实例。

这些工厂函数提供了一种简便的方式来创建不同组件的实例，使得在创建模拟环境时更加方便和灵活。通过传递适当的参数字典，即可创建所需的组件实例，并将其组合成一个完整的微网模拟环境。

### households.py

这里定义了几个与房屋（Households）相关的类和函数：

1. `PricingManager`：价格管理器，用于跟踪能源价格并验证代理（agent）的价格级别决策。通过`validate_price_level`方法，根据代理给出的价格级别，验证并返回有效的价格级别。实际验证中，价格管理器可根据设定的规则来判断是否采用代理建议的价格级别，或者是否使用预定的特定价格级别。

2. `HouseholdsManager`：房屋管理器，用于处理微网中的房屋（PriceResponsiveLoad）以及相关的价格。根据代理给出的价格级别，验证并返回房屋的能源消耗和获得的利润。该类包含了价格敏感负荷（`PriceResponsiveLoad`）的列表，以及代表价格变化的价格数组。使用`get_pricing_counter`方法返回价格计数器，该计数器跟踪累积的价格级别。使用`get_base_residential_load`方法根据指定的时间来获取房屋的基础负荷。使用`get_consumption_and_profit`方法返回房屋的能源消耗和获得的利润。

3. `PriceResponsiveLoad`：代表价格敏感负荷的类。每个价格敏感负荷具有敏感性和耐心这两个属性。根据基础负荷和价格级别，通过`get_load`方法计算并返回负荷。

这些类和函数增强了房屋模拟的功能，使得模拟环境更贴近实际情况，并且能够模拟代理在不同价格级别下的能源消耗和利润。

### main_grid.py

在这里，定义了与主电网（MainGrid）相关的类和函数：

1. `MainGridParams`：主电网参数类，用于存储主电网的配置信息，包括上行和下行价格文件路径，以及导入和导出的传输价格。

2. `MainGrid`：主电网模型类，用于模拟主电网的行为。在`__init__`方法中，接受上行和下行价格的DataFrame，以及导入和导出的传输价格。通过`from_params`方法，从`MainGridParams`对象创建主电网实例。`get_prices`方法返回指定索引处的上行和下行价格。`get_up_price`和`get_down_price`方法分别返回指定索引处的上行和下行价格。`get_bought_cost`方法返回指定小时从主电网购买能源的成本，包括传输成本。`get_sold_profit`方法返回指定小时向主电网出售能源所获得的利润，考虑传输成本。

这些类和函数使得主电网的模拟更加完整，可以模拟代理从主电网购买或出售能源，并计算相应的成本和利润。这样，整个环境可以更准确地模拟微网中各个组件的相互作用和代理的行为。

### price_responsive.py

在这里，定义了一个价格响应负载（PriceResponsiveLoad）模型类：

`PriceResponsiveLoad`是一个简单的模型，用于表示一个响应价格变化的负载。它模拟了负载在不同的价格级别下的行为。负载的响应受到`patience`参数的影响，表示负载在收到价格信号后需要多长时间来调整自身的响应。模型在每个时间步中根据当前的价格级别和`patience`参数，决定是否在该时间步中执行相应的负载。

属性：

- `sensitivity`：负荷对价格的敏感度，表示负荷对价格变化的响应程度。
- `patience`：负荷响应的耐心度，表示负荷在响应价格变化后会持续多少个时间步骤。
- `_shifted_loads`：存储待执行的负荷的字典，键为时间步骤，值为负荷值。
- `_timestep_counter`：时间步骤计数器，用于生成时间步骤的序列。

该模型包含以下方法：

- `get_load(base_load, price_level)`：在给定基础负载和当前价格级别的情况下，返回实际执行的负载。它会计算当前时间步的负载偏移，并将其添加到要在未来时间步执行的列表中。然后，它返回根据当前价格级别调整后的负载。

  在给定的代码片段中，`get_load`方法用于更新模型并获取在当前时间步骤上执行的负荷。该方法的输入参数包括基本负荷 `base_load` 和当前的价格水平 `price_level`，输出为最终的负荷值。

  方法的逻辑如下：

  1. 首先，从计数器 `_timestep_counter` 中获取当前时间步骤 `timestep`。

  2. 调用 `_get_shifted_load_to_execute` 方法，该方法用于获取待执行的负荷，根据当前的价格水平和时间步骤。

  3. 计算需要转移的负荷 `load_to_shift`，它是基本负荷 `base_load` 乘以敏感度 `sensitivity` 和价格水平 `price_level` 的结果。

  4. 将待执行的负荷添加到待执行负荷字典 `_shifted_loads` 中，其中的键是时间步骤 `timestep`，值是待执行的负荷。

  5. 返回最终的负荷值，通过将基本负荷 `base_load` 减去 `load_to_shift` 的结果，并加上 `shifted_load_to_execute`。

  总的来说，`get_load`方法根据当前的价格水平和时间步骤来调整基本负荷，以响应不同的价格信号，从而实现对负荷的优化。

- `_get_shifted_load_to_execute(current_price_level, current_timestep)`：返回在当前时间步需要执行的负载。它会检查先前添加到列表中的偏移负载，如果到达执行时间（考虑`patience`参数），则从列表中移除并返回执行的负载。

- `_execute_load(load, load_timestep, current_timestep, current_price_level)`：根据当前时间步、当前价格级别、负载执行时间和`patience`参数，计算是否在当前时间步执行负载的概率。

- `_add_new_shifted_load(load, timestep)`：将新的负载偏移添加到要在未来时间步执行的列表中。

这个价格响应负载模型可以被用作环境中的一个组件，例如家庭负载的模拟，以更加真实地反映代理的行为和负载在不同价格水平下的变化。

### tcl.py

在这里，定义了一个备用控制器（BackupController）模型类和一个 TCL 温度模型（TCLTemperatureModel）类：

- `BackupController` 是一个用于控制室内温度保持在可接受范围内的备用控制器。它根据设置的温度限制限制给定的 TCL 动作。当室内温度高于最大温度限制时，将 TCL 动作设为 OFF（0）。当室内温度低于最小温度限制时，将 TCL 动作设为 ON（1）。否则，将保留原 TCL 动作。

- `TCLTemperatureModel` 是一个用于存储和更新温度信息的类。它可以根据当前室外温度和 TCL 提供的供暖/冷却来更新室内温度。温度模型基于热量平衡原理来计算室内温度的变化。模型考虑了室内空气和建筑物的热质量，并计算室内温度的变化。通过提供 TCL 的供暖功率和建筑物的热质量，可以计算出新的室内温度和建筑物温度。

  `TCLTemperatureModel`是用于存储和更新温度信息的类，表示一个可热控负载（TCL）的温度模型。它基于空气和建筑物的热质量来存储和更新温度信息。该类有以下属性：

  1. `in_temp`：浮点数，表示TCL的室内温度，即TCL试图维持的目标温度。
  2. `_out_temp`：浮点数，表示室外温度，影响TCL的温度动态。
  3. `_building_temp`：浮点数，表示TCL所在建筑物的当前温度。
  4. `_therm_mass_air`：浮点数，表示TCL内空气的热质量。它影响室内温度对外部因素的响应速度。
  5. `_therm_mass_building`：浮点数，表示建筑物结构的热质量。它也影响室内温度随时间的变化。
  6. `_building_heating`：浮点数，表示施加在建筑物上的加热功率。它影响由于加热而引起的室内温度变化。

  该类可用于模拟TCL随时间的室内温度动态，考虑了室外温度和施加的加热功率。具体的计算和更新室内温度的方法没有在代码片段中提供。

  要使用该类，通常需要根据外部因素（例如天气数据、加热控制信号）更新`_out_temp`和`_building_heating`属性，并调用用于更新室内温度的方法，例如`update()`，该方法会根据热质量和加热功率应用温度动态。

  请注意，代码片段仅定义了类属性，并且在完整的类定义中可能还有其他处理温度更新和其他功能的方法。

- `TCL` 是一个用于建模可调节温控负载（TCL）的模型类，例如空调或热水器等。TCL 模型包含一个备用控制器和一个 TCL 温度模型。`__post_init__` 方法初始化 TCL 的状态，并计算初始状态的充电状态（SoC）。`update` 方法根据室外温度和所需的 TCL 动作（ON/OFF）来更新 TCL 的状态。它调用备用控制器来限制 TCL 的动作，然后调用 TCL 温度模型来更新室内温度和建筑物温度，并计算 TCL 的能量消耗和充电状态。

  `TCL`类是一个模拟具有温控功能的负载（Thermostatically Controlled Load，TCL）的模型，例如空调或热水器等。该类具有以下属性和方法：

  - 属性：
    - `soc`：当前TCL的状态-of-charge（SOC），表示其充电状态，是一个浮点数。
    - `nominal_power`：TCL的额定功率，表示TCL的最大能量消耗，是一个浮点数。
    - `_backup_controller`：TCL的备用控制器（BackupController）对象，用于控制TCL的启停和调整能量消耗。
    - `_temp_model`：TCL的温度模型（TCLTemperatureModel）对象，用于存储和更新温度信息。

  - 方法：
    - `__post_init__(self)`：类的初始化方法，在对象创建后自动调用。该方法通过调用备用控制器（`_backup_controller`）的`get_state_of_charge`方法来计算初始SOC，并将其存储在`soc`属性中。
    - `update(self, out_temp: float, tcl_action: int) -> float`：更新TCL的状态。该方法接受外部温度`out_temp`和TCL的动作`tcl_action`作为输入，并根据`tcl_action`计算TCL的能量消耗。具体步骤如下：
      1. 调用备用控制器（`_backup_controller`）的`get_action`方法，根据`tcl_action`和当前室内温度（`_temp_model.in_temp`）计算TCL的实际动作（`action`）。
      2. 根据TCL的额定功率（`nominal_power`）和实际动作（`action`）计算TCL的加热功率（`tcl_heating`）。
      3. 调用温度模型（`_temp_model`）的`update`方法，根据外部温度（`out_temp`）和TCL的加热功率（`tcl_heating`）更新TCL的室内温度，并将更新后的室内温度存储在`in_temp`变量中。
      4. 根据更新后的室内温度调用备用控制器（`_backup_controller`）的`get_state_of_charge`方法，计算TCL的新SOC，并将其存储在`soc`属性中。
      5. 返回TCL的加热功率（`tcl_heating`）。

  该类模拟了TCL的动态行为，根据外部温度和用户设置的动作来控制TCL的启停和能量消耗，并实时更新TCL的室内温度和SOC。

这个 TCL 模型可以在环境中作为一个组件，以更好地模拟代理和 TCL 设备的行为，并影响环境状态（室内温度等）。

### tcl_aggregator.py

这里定义了一个 TCL 聚合器（TCLAggregator）类，用于控制 TCL 设备集群的行为：

- `TCLAggregator` 是一个代理，用于控制 TCL 设备集群的行为。它包含多个 TCL 实例（TCL 对象列表），每个 TCL 实例表示一个 TCL 设备。TCLAggregator 可以根据外部环境（室外温度）的情况来分配电能给 TCL 设备，控制 TCL 设备的行为，以实现对室内温度的调节。

- `TCLParams` 是一个用于初始化 TCLAggregator 的参数类。它定义了 TCL 设备集群的数量、室外温度序列以及其他 TCL 参数（例如热质量、最大功率等）的平均值和标准差。`from_dict` 方法用于从字典中创建 TCLParams 实例。

- `TCLAggregator.from_params` 方法根据传入的 TCLParams 参数来创建 TCLAggregator 实例。它会初始化 TCL 设备集群（TCL 实例列表），并为每个 TCL 设备设置备用控制器和 TCL 温度模型。同时，它还会为 TCL 设备分配随机的初始功率，并将室内温度初始化为介于最小温度和最大温度之间的随机值。

- `TCLAggregator.get_outdoor_temperature` 方法用于获取给定索引处的室外温度。

- `TCLAggregator.get_state_of_charge` 方法用于计算 TCL 设备集群的平均充电状态（SoC）。

- `TCLAggregator.allocate_energy` 方法用于将能量分配给 TCL 设备集群。它将能量分配给 TCL 设备并更新 TCL 设备的状态。分配的能量将被用于供暖或冷却，从而调节室内温度。

- `TCLAggregator.get_number_of_tcls` 方法返回 TCL 设备集群的 TCL 数量。

- `TCLAggregator._get_desired_tcl_action` 方法用于获取 TCL 设备的期望动作。如果 TCL 设备的最大功率小于剩余的能量，那么它将被设为 ON（1），否则设为 OFF（0）。

这个 TCLAggregator 类将 TCL 设备集群作为一个组件，并允许模拟 TCL 设备的行为和其对环境状态的影响。该类可以与其他组件（例如主电网、分布式能源资源等）一起用于建立微电网仿真环境。

## environment.py

这些代码实现了一个微电网环境（Microgrid Environment），该环境是一个模拟环境，用于与 EMS（Energy Management System）代理进行交互。它由多个组件组合而成，包括热控装置（TCLs，Thermostatically Controlled Loads）、储能系统（ESS，Energy Storage System）、分布式能源资源（DER，Distributed Energy Resources）以及居民家庭等。

主要组件包括：

1. `Environment`：代表微电网环境，EMS代理与之交互。该环境根据代理的行动模拟一步，返回环境状态和奖励（生成的利润）。

2. `get_default_microgrid_params()`：获取微电网的默认参数。

3. `get_default_microgrid_env()`：创建一个默认的微电网环境。

4. 其他辅助函数和配置。

在环境中，EMS代理可以采取一系列控制动作，包括对TCL的操作、价格水平选择、优先级设置（如缺电和超出能源优先使用ESS还是主电网）等。环境会根据代理的控制动作来模拟电力系统的运行，并返回相应的状态信息和奖励。

该环境主要用于对能源管理系统的策略进行训练和评估。代理可以根据当前环境状态和奖励信息来学习并优化其控制策略，以实现最大的利润或能源效率。

这是`Environment`类的代码，它代表着EMS（Energy Management System）代理与之交互的微电网环境。让我们逐步解释这个类的功能：

1. `__init__()`方法：构造函数，用于初始化微电网环境。它接收微电网的参数字典`params_dict`、价格和温度数据文件的路径`prices_and_temps_path`和起始时间索引`start_time_idx`。该方法会根据参数字典和数据文件路径创建微电网环境的各个组件，比如热控装置（TCLs）、储能系统（ESS）、分布式能源资源（DER）和居民家庭等，并初始化时间步计数器和时间索引。

2. `step()`方法：模拟一次时间步。接收来自EMS代理的动作（action）作为输入，其中action是一个包含四个整数的元组。方法会根据动作应用到微电网环境中，并返回环境状态和生成的利润（reward）。

   在给定的代码片段中，`step`方法用于模拟一个时间步骤，使用给定的控制动作（`action`）。该方法的输入是一个四元组 `action`，其中包含四个整数值，表示控制动作：

   1. `tcl_action`: Thermostatically Controlled Load (TCL) 的动作，表示 TCL 的状态 (ON/OFF)。
   2. `price_level`: 当前的价格水平，表示价格响应负荷的信号，值在 {-2, -1, 0, 1, 2} 范围内。
   3. `def_prio`: 电池 (ESS) 的动作，1 表示使用电池，0 表示购买电力。
   4. `excess_prio`: 电池 (ESS) 的动作，1 表示将过剩电力存储到电池，0 表示将过剩电力卖出。

   方法逻辑如下：

   1. 通过调用计数器 `_timestep_counter` 来获取当前的时间步骤 `_idx`。

   2. 从 `action` 中获取 TCL 的动作 `tcl_action` 和当前的价格水平 `price_level`。

   3. 根据 `action` 中的 `def_prio` 和 `excess_prio` 来确定是否使用电池（ESS）以及如何处理过剩电力。

   4. 调用 `_apply_action` 方法，根据控制动作来执行相应的操作，并返回该时间步骤的奖励值 `reward`。

   5. 调用 `get_state` 方法获取当前环境的状态 `state`。

   6. 返回时间步骤结束后的环境状态 `state` 和奖励值 `reward`。

   总体来说，`step`方法用于执行一个时间步骤的模拟，根据给定的控制动作来更新环境状态，并返回该时间步骤的奖励值。

3. `_get_tcl_energy()`方法：根据给定的TCL动作（tcl_action）返回能源数量，其值为TCL最大消耗的0%、33%、67%或100%。

4. `_apply_action()`方法：应用EMS代理的动作到微电网环境中，返回相应的奖励。这里会根据动作来调整热控装置、居民家庭的用电和发电情况，以及能源的买卖情况。

   `_apply_action` 方法用于应用代理（agent）的选择，并返回奖励值。方法的输入参数包括：

   1. `tcl_action`: Thermostatically Controlled Load (TCL) 的动作，表示 TCL 的状态 (ON/OFF)。
   2. `price_level`: 当前的价格水平，表示价格响应负荷的信号，值在 {-2, -1, 0, 1, 2} 范围内。
   3. `deficiency_prio`: 字符串，表示电池 (ESS) 的动作，"ESS" 表示使用电池，"BUY" 表示购买电力。
   4. `excess_prio`: 字符串，表示电池 (ESS) 的动作，"ESS" 表示将过剩电力存储到电池，"SELL" 表示将过剩电力卖出。

   方法逻辑如下：

   1. 根据 TCL 的动作 `tcl_action` 和当前价格水平 `price_level`，调用 `_get_tcl_energy` 方法计算 TCL 的能量消耗。

   2. 调用 `tcl_aggregator` 的 `allocate_energy` 方法来分配 TCL 的能量消耗。

   3. 调用 `households_manager` 的 `get_consumption_and_profit` 方法来获取住户的能量消耗和收益。

   4. 调用 `der` 的 `get_generated_energy` 方法来获取该时间步骤产生的能量。

   5. 计算剩余的能量（过剩或不足）。

   6. 如果存在过剩能量，则根据 `excess_prio` 处理过剩能量；否则，根据 `deficiency_prio` 处理能量不足。

   7. 调用 `_compute_reward` 方法来计算奖励值。

   8. 返回计算得到的奖励值。

   总体来说，`_apply_action` 方法用于应用代理的选择（如 TCL 的状态、ESS 的动作等），并根据环境中的能量消耗和产生情况计算奖励值。

5. `_cover_energy_deficiency()`方法：处理能源不足情况，通过使用储能系统（ESS）和/或主电网来弥补不足的能源，并返回花费的成本。

6. `_handle_excess_energy()`方法：处理能源过剩情况，通过储能系统（ESS）和/或卖给主电网来储存或出售多余的能源，并返回相应的收益。

   在 `_handle_excess_energy` 函数中，代理（Agent）需要处理剩余的能源（excess energy）。这个函数将根据传入的优先级（priority）决定如何处理这些剩余能源，并返回相应的利润。

   如果优先级为 "SELL"，意味着代理选择将剩余能源出售给主电网。函数会调用 `self.components.main_grid.get_sold_profit` 方法来计算代理在当前时间步卖出能源获得的利润，并将该利润作为返回值。

   如果优先级不是 "SELL"，即为 "ESS"，意味着代理选择将剩余能源存储到能源存储系统（ESS）。函数首先将剩余能源传递给 `self.components.ess.charge` 方法来将能源存储到 ESS 中，并获取存储到 ESS 中的能源量。然后，函数再次调用 `self.components.main_grid.get_sold_profit` 方法来计算代理在当前时间步卖出存储到 ESS 之后的能源获得的利润，并将该利润作为返回值。

   总体而言，这个函数根据代理在当前时间步的优先级决定如何处理剩余能源，并返回相应的利润。如果优先级为 "SELL"，则代理选择将剩余能源出售给主电网；如果优先级为 "ESS"，则代理选择将剩余能源存储到 ESS 中，并随后再次考虑将存储到 ESS 中的能源出售给主电网。

7. `_compute_reward()`方法：计算并返回EMS代理在当前时间步的奖励（生成的利润），它包括热控装置的消耗成本、居民家庭的利润和主电网的利润。

8. `get_state()`方法：收集并返回EMS代理在当前时间步的环境状态，包括热控装置和储能系统的状态、居民家庭的价格计数器、室外温度、发电情况、主电网上调价格、当前时间步的小时以及居民家庭的基本用电负荷。

这个`Environment`类的设计是为了模拟能源管理系统在微电网中的运行，并且它可以用于训练和评估能源管理系统代理的控制策略。通过在环境中模拟不同的动作和状态，代理可以通过学习来优化其控制策略，以实现更好的能源利用和经济效益。

# test

## test_grid_v0.py

这是一个用于测试`GridV0Env`环境是否能与OpenAI Gym一起运行的脚本。脚本通过调用`gym.make`来创建`GridV0Env`环境，并运行多个回合进行测试。

在测试过程中，脚本将随机选择动作，并在每个步骤上渲染环境。每个回合结束后，脚本将输出该回合的步数、总奖励和终止信息。

请确保在运行测试之前，已经正确安装了依赖包`gym`和`custom_envs`。运行测试时，脚本将模拟执行10个回合，并输出每个回合的信息。

注意：在测试期间，可能会生成图形化界面显示环境的状态，如果出现问题，可以在`GridV0Env`的`render`函数中进行调整。

## print_prices_and_temps.py

这个脚本用于获取数据集中各列的最大值、最小值、平均值和标准差。数据集包括`wind_generation.csv`，`up_regulation.csv`，`down_regulation.csv`和`default_price_and_temperatures.npy`。

首先，脚本读取数据，并输出各列的最小值和最大值，以及平均值和标准差。

接下来，脚本展示了名为`base_hourly_loads`的小时基本负载数据，并输出了其平均值和标准差。

最后，注释掉了`main()`函数的调用，并调用`get_max_and_min_vals()`函数来获取数据集的统计信息。

## test_der.py

这是一个`DER`（分布式能源资源）类的单元测试脚本。在测试中，`DER`类被实例化，并使用`wind_generation.csv`中的数据初始化。然后，测试检查`get_generated_energy()`方法的输出是否是`float`类型。

测试在运行时会对`DER`类的`get_generated_energy()`方法进行多次调用，并对每次调用的结果进行类型断言，确保返回的值是`float`类型。

如果所有的断言都成功，测试将通过，否则会返回失败信息。

## test_microgrid_environment.py

这是一个测试脚本，用于测试`Environment`类的初始化和`step`方法。

在测试中，首先设置了用于初始化`Environment`类的参数，包括TCL、ESS、主电网、DER和住宅的参数。然后，从数据文件中加载价格和温度数据，并使用这些参数和数据创建了一个`Environment`类的实例。

接下来，测试调用了`Environment`类的`step`方法，传入一个预定义的动作。该方法返回一个新的状态向量和奖励值。最后，测试将状态向量和奖励值打印出来以供检查。

如果所有的断言都成功，测试将通过，否则会返回失败信息。

## test_params.py

这是一个测试脚本，用于测试`TCLParams`类的`from_dict`方法。

测试中首先创建了一个包含TCL参数的字典`params_dict`。然后，调用`TCLParams.from_dict`方法，将字典作为参数传入，创建一个`TCLParams`类的实例`params`。

接下来，测试检查`params`实例的属性是否正确设置。在这个例子中，测试检查了`num_tcls`、`out_temperatures[0]`、`min_temp`和`max_temp`这些属性的值是否符合预期。

如果所有的断言都成功，测试将通过，否则会返回失败信息。

## test_price_responsive.py

这是一个测试脚本，用于测试`PriceResponsiveLoad`类的两个方法：`_execute_load`和`get_load`。

在`_execute_load`方法中，测试了不同情况下的执行负载操作的结果是否正确。`_execute_load`方法模拟了一个随机概率（由`rand_out`参数控制），并根据该概率决定是否执行负载。测试用例涵盖了不同的概率情况，并检查结果是否符合预期。

在`get_load`方法中，测试了一个简单的用例。`get_load`方法会根据当前时间步和历史负载信息来计算并返回当前负载。测试用例中，模拟了当前时间步为4，历史负载信息为{-1: -1.0, 2: 1.0, 3: -1.0}的情况。根据模拟的概率（0.5），方法计算出当前负载为7.0，并更新了历史负载信息。

如果所有的断言都成功，测试将通过，否则会返回失败信息。

## test_run_neat_evolution.py

这是一个使用NEAT算法的测试脚本。在这个脚本中，使用了NEAT（NeuroEvolution of Augmenting Topologies）算法来优化神经网络的拓扑结构。

在测试中，首先定义了两个模拟的函数：

1. `mock_fitness_function`: 这是一个简单的适应度函数，用于演示目的。它为每个基因组随机分配一个适应度值。

2. `mock_species_fitness_function`: 这是一个简单的物种适应度函数，用于演示目的。它将所有物种的适应度值求平均作为整个物种的适应度。

然后，通过创建一个`NeatParams`对象来设置NEAT算法的参数。接下来，创建一个`Evolution`对象，并传入初始的输入和输出节点数量、NEAT算法的配置参数和物种适应度函数。然后，调用`run`方法来运行NEAT算法的演化过程，使用`mock_fitness_function`作为实际的适应度函数，并设置适应度目标为2.0，运行10代。

如果一切顺利，测试将通过，否则将返回失败信息。请注意，这里的演化过程只是为了演示目的，并没有实际的网络训练过程。在实际使用中，你需要根据具体问题来实现适应度函数和物种适应度函数，并在`run`方法中传入实际的适应度函数来进行网络的训练和优化。

## test_tcl.py

这是一个TCL（Thermostatically Controlled Load）类的单元测试脚本。TCL是一个模拟温度控制的组件，可以根据室内温度和外部温度来调整供暖功率。

在测试中，首先创建了一个TCL对象，以及与TCL相关的BackupController对象和TCLTemperatureModel对象。

接下来，分别测试了以下几个功能：

1. `test_backup_get_action`: 测试BackupController对象的`get_action`方法，以确保根据室内温度和指定的操作动作（0或1）正确地返回最终操作动作。

2. `test_get_soc`: 测试BackupController对象的`get_state_of_charge`方法，以确保根据室内温度正确地返回状态电荷值。

3. `test_model_update`: 测试TCLTemperatureModel对象的`update`方法，以确保室内温度的更新方向正确，根据外部温度和供暖功率的情况。

如果所有测试通过，测试脚本将不会输出任何错误消息。否则，将返回相关的错误信息。这些测试用例将确保TCL组件在不同情况下的行为正确，从而确保组件的可靠性和正确性。

## test_tcl_aggregator.py

这是一个TCLAggregator（TCL聚合器）类的单元测试脚本。TCLAggregator是一个组件，它可以聚合多个TCL组件，并根据它们的状态电荷和温度来分配能量。

在测试中，首先创建了三个TCL对象，并初始化它们的初始状态电荷和室内温度。

接下来，创建了TCLAggregator对象，并将上述三个TCL对象传递给它。

接下来，分别测试了以下两个功能：

1. `test_get_soc`: 测试TCLAggregator对象的`get_state_of_charge`方法，以确保正确地返回TCL聚合器的状态电荷。

2. `test_allocate_energy`: 测试TCLAggregator对象的`allocate_energy`方法，以确保根据指定的能量分配给TCL组件，并返回分配的能量。

如果所有测试通过，测试脚本将不会输出任何错误消息。否则，将返回相关的错误信息。这些测试用例将确保TCL聚合器在不同情况下的行为正确，从而确保组件的可靠性和正确性。