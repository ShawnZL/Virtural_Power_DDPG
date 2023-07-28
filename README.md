# Action space

TCL action:[0, 3] 原始action是[0,1] => [on, off]，根据当前TCL最大能量消耗大于当前剩余能量，那么就应该开始工作-> [0, 33%, 67%, 100%] 运转



Price action:[0, 4] 表示价格响应负荷的信号，值在{-2，-1，0，1，2}当中 正是的，`price_level` 表示当前的价格水平，它是用来指示价格响应负荷的信号的一个值。该值在范围 {-2, -1, 0, 1, 2} 内，代表着不同的价格信号：

- `-2`: 表示非常低的价格信号，通常意味着电力价格非常便宜，鼓励用户增加用电量。
- `-1`: 表示较低的价格信号，电力价格较便宜，鼓励用户增加用电量。
- `0`: 表示标准价格信号，电力价格正常，没有特殊调整。
- `1`: 表示较高的价格信号，电力价格较贵，鼓励用户减少用电量。
- `2`: 表示非常高的价格信号，通常意味着电力价格非常昂贵，鼓励用户大幅减少用电量。

根据当前的价格水平，代理（agent）可以调整其行为，例如控制 TCL 或 ESS 的状态，以及优化能量消耗或生产，从而获得更好的奖励值或利润。价格响应负荷是一种智能电网中的重要机制，可以通过调整价格信号来实现电力需求和供应的平衡，以提高能源的效率和经济性。



Energy deficiency action:[0, 1] 0表示购买电力，1表示使用电池能源。



Energy excess action:[0, 1] 0表示将过剩电力卖出，1表示将过剩电力存储到电池当中。

使用给定的控制动作（`action`）。该方法的输入是一个四元组 `action`，其中包含四个整数值，表示控制动作：

1. `tcl_action`: Thermostatically Controlled Load (TCL) 的动作，表示 TCL 的状态 (ON/OFF)。
2. `price_level`: 当前的价格水平，表示价格响应负荷的信号，值在 {-2, -1, 0, 1, 2} 范围内。
3. `def_prio`: 电池 (ESS) 的动作，1 表示使用电池，0 表示购买电力。
4. `excess_prio`: 电池 (ESS) 的动作，1 表示将过剩电力存储到电池，0 表示将过剩电力卖出。

# reward

在这个函数中，`_compute_reward` 用于计算每个时间步的奖励（reward）。奖励是由以下三个部分组成：

1. TCL 负荷消耗（tcl_consumption）乘以 DER（分布式能源资源）的发电成本（gen_cost）：这表示消耗的电力乘以单位电力的成本，以此作为一个奖励或惩罚。如果 TCL 负荷消耗的电力越少，那么奖励就越高；如果 TCL 负荷消耗的电力越多，那么奖励就越低。

2. 住户的盈利（residential_profit）：这表示住户根据当前价格水平（price_level）购买电力时的盈利。如果价格水平低，住户以较低的价格购买电力，其盈利会增加，从而增加奖励；如果价格水平高，住户以较高的价格购买电力，其盈利会减少，从而减少奖励。

3. 主电网的盈利（main_grid_profit）：这表示主电网根据当前价格水平（price_level）出售电力时的盈利。如果价格水平高，主电网以较高的价格出售电力，其盈利会增加，从而增加奖励；如果价格水平低，主电网以较低的价格出售电力，其盈利会减少，从而减少奖励。

最终，奖励是上述三个部分之和，表示代理（Agent）在当前时间步的总体表现。奖励越高，表示代理在当前状态下做出了更好的决策；奖励越低，表示代理在当前状态下做出了较差的决策。

# Debug

`./microgird_sim/environment` 中

```python
class Environment:
    """Environment that the EMS agent interacts with, combining the components together."""

    __slots__ = ("components", "_timestep_counter", "_idx")

    def __init__(self, params_dict: dict[str, dict[str, Any]], prices_and_temps_path: str, start_time_idx: int):
        tcl_params = params_dict["tcl_params"]
        ess_params = params_dict["ess_params"]
        main_grid_params = params_dict["main_grid_params"]
        der_params = params_dict["der_params"]
        residential_params = params_dict["residential_params"]

        prices_and_temps = np.load(prices_and_temps_path)
        residential_params["hourly_base_prices"] = prices_and_temps[:, 0]
        tcl_params["out_temps"] = prices_and_temps[:, 1]

        self.components = get_components_by_param_dicts(
            tcl_params, ess_params, main_grid_params, der_params, residential_params
        )
        self._timestep_counter = count(start_time_idx)
        self._idx = start_time_idx
        # print('begin time_step{} idx {}'.format(self._timestep_counter, self._idx))

    def step(
        self, action: tuple[int, int, int, int]
    ) -> tuple[tuple[float, float, float, float, float, float, int, int], float]:
        """
        Simulate one timestep with the given control actions.

        Returns state of the environment and reward (generated profit).
        """
        # 之前的bug，因为使用next会导致第一次参数没有办法更新
        # print('before_next idx {} 2 {}'.format(self._idx, next(self._timestep_counter)))
        # self._idx = next(self._timestep_counter)
        self._idx += 1
```

`self._idx = start_time_idx` 为初始设置idx参数

之后在step中，`self._idx = next(self._timestep_counter)` 中 `self._timestep_counter` 由0变为1，但是`_idx` 依旧是0。导致第一次跟0之后的小时状态依旧为0。