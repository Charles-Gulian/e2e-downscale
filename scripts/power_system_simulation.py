import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import pathlib

# Object Oriented CVXPY Model

T = 24  # Time window of optimization model(s)

### HELPERS ###

import numpy as np

def forced_outage_profile(FOR: float, MTTR: float, T: int,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a binary availability profile of length T.

    State convention:
        1 = available (up)
        0 = forced outage (down)

    Parameters
    ----------
    FOR : float
        Stationary forced outage rate (probability of being down).
    MTTR : float
        Mean time to repair (in time steps).
    T : int
        Simulation length.
    rng : numpy.random.Generator, optional

    Returns
    -------
    profile : np.ndarray (uint8)
        Length-T binary availability vector.
    """

    if not (0.0 < FOR < 1.0):
        raise ValueError("FOR must be in (0,1).")
    if MTTR <= 0.0:
        raise ValueError("MTTR must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    # Calibrate transition probabilities
    p10 = 1.0 / MTTR                       # down -> up
    p01 = (FOR / (1.0 - FOR)) * p10        # up -> down

    if p01 >= 1.0:
        raise ValueError("Implied failure probability >= 1. Check FOR/MTTR consistency.")

    profile = np.empty(T, dtype=np.uint8)

    # Random initial state from stationary distribution
    profile[0] = 1 if rng.random() < (1.0 - FOR) else 0

    # Pre-generate uniforms for speed
    u = rng.random(T - 1)

    for t in range(1, T):
        if profile[t - 1] == 1:
            # up -> down
            profile[t] = 0 if u[t - 1] < p01 else 1
        else:
            # down -> up
            profile[t] = 1 if u[t - 1] < p10 else 0

    return profile


### NODE ###
class Component:

    def __init__(self):
        self.system = None

    def link_system(self, system):
        self.system = system

    def create_parameters(self):
        pass

    def update_timeseries_parameters(self, date):
        pass

    def create_variables(self):
        pass

    def create_expressions(self):
        pass


### NODE ###
class Node(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack node attributes from data
        self.peak_load = data["MW Load"]
        self.node_type = data["Bus Type"]
        self.lat = data["lat"]
        self.long = data["lng"]
        self.area = data["Area"]
        self.sub_area = data["Sub Area"]
        self.zone = data["Zone"]

        # Set other, generic attributes
        self.VOLL = 1e4  # $/MWh of unserved energy

        # Initialize resources
        self.resources = []

        # Initialize in/out lines
        self.in_lines = []
        self.out_lines = []

        # Initialize load
        self.load_profile = None

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def get_load_profile(self, year):
        data_source = "NSRDB"
        ts_data_dir = self.system.base_dir / "load-data" / "profiles"
        ts_data_path = ts_data_dir / f"bus{self.name}" / f"{data_source}_load-profile_bus{self.name}_{year}.csv"
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    def create_parameters(self):
        self.load = cp.Parameter(T)
        self.perfect_capacity = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.load.value = self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values

    def update_perfect_capacity_parameter(self, perfect_capacity_value):
        self.perfect_capacity.value = np.repeat(perfect_capacity_value, T)

    def create_variables(self):
        self.perfect_generation = cp.Variable(T)
        self.unserved_energy = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        constraints = [
            cp.sum([r.p_out for r in self.resources])  # nodal generation
            + cp.sum([l.flow for l in self.in_lines])  # + line inflows
            - cp.sum([l.flow for l in self.out_lines])  # - line outflows
            + self.perfect_generation # + generation from perfect capacity
            + self.unserved_energy # + unserved energy
            == self.load, # = load (power balance constraint)
            self.perfect_generation <= self.perfect_capacity, # (max perfect capacity constraint)
        ]
        self.power_balance_constraint = constraints[0]
        self.max_perfect_capacity = constraints[1]
        return constraints

    @property
    def power_balance_duals(self):
        return self.power_balance_constraint.dual_value


### LINE ###

class Line(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack line attributes from data
        self.from_node_ID = data["From Bus"]
        self.to_node_ID = data["To Bus"]
        self.flow_limit = data[ "LTE Rating"]  # Long-term flow limit; alternatively can use continuous/short-term limits
        self.X = data["X"]  # Line susceptance # MVA base = 100?
        self.FOR = data["Perm OutRate"]
        self.MTTR = data["Duration"]

        # Initialize other parameters
        self.baseMVA = 100.0

        # Initialize from node / to node
        self.from_node = None
        self.to_node = None

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def link_nodes(self, nodes: dict):
        # Link nodes to line
        self.from_node = nodes[self.from_node_ID]
        self.to_node = nodes[self.to_node_ID]
        # Link line to nodes
        nodes[self.from_node_ID].out_lines.append(self)
        nodes[self.to_node_ID].in_lines.append(self)

    def create_parameters(self):
        self.max_flow = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.max_flow.value = self.flow_limit * forced_outage_profile(self.FOR, self.MTTR, T)

    def create_variables(self):
        self.flow = cp.Variable(T)

    def write_constraints(self):
        constraints = [
            self.flow <= self.max_flow,
            self.flow >= -self.max_flow,
        ]
        self.forward_flow_limit_constraint = constraints[0]
        self.reverse_flow_limit_constraint = constraints[1]
        return constraints

### RESOURCES ###

class Resource(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack basic resource attributes from data
        self.node_ID = data["Bus ID"]  # Bus ID
        self.unit_type = data["Unit Type"]  # Unit type
        self.nameplate_capacity = data["PMax MW"]  # Nameplate capacity (MW)
        self.VOM = data["VOM"]  # Variable O&M costs ($/MWh)

        # Initialize node
        self.node = None

    @classmethod
    def from_series(cls, df: pd.Series):
        return cls(df.name, df)

    def link_node(self, nodes: dict):
        # Link node to resource
        self.node = nodes[self.node_ID]
        # Link resource to node
        nodes[self.node_ID].resources.append(self)

    def create_parameters(self):
        self.pmax = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.pmax.value = np.repeat(self.nameplate_capacity, T)

    def create_variables(self):
        self.p_out = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        constraints = [self.p_out <= self.pmax]
        self.pmax_constraint = constraints[0]
        return constraints

    @property
    def pmax_duals(self):
        return self.pmax_constraint.dual_value


class ThermalResource(Resource):

    def __init__(self, name, data):
        super().__init__(name, data)

        # Unpack thermal resource attributes from data

        # Cost attributes
        self.fuel_price = data["Fuel Price $/MMBTU"]
        self.heat_rate = data["HR_avg_0"] / 1000  # BTU/kWh --> MMBTU/MWh
        self.variable_costs = self.fuel_price * self.heat_rate + self.VOM

        # Operational attributes
        self.pmin = data["PMin MW"]  # Minimum output (for unit commitment
        self.ramp_rate = 60 * data["Ramp Rate MW/Min"]
        self.FOR = data["FOR"]
        self.MTTR = data["MTTR Hr"]

    def update_timeseries_parameters(self, date: pd.Timestamp):
        forced_outage_profile
        self.pmax.value = self.nameplate_capacity * forced_outage_profile(self.FOR, self.MTTR, T)


class VariableResource(Resource):
    def __init__(self, name, data):
        super().__init__(name, data)

        # Resource type
        if self.unit_type in ["PV", "RTPV"]:
            self.resource_type = "solar"
        elif self.unit_type in ["WIND"]:
            self.resource_type = "wind"
        else:
            print(f"Unknown variable resource type: {self.unit_type}")
            self.resource_type = None

        # Cost attributes
        self.variable_costs = self.VOM

        # Initialize generation profile
        self.gen_profile = None

    def get_gen_profile(self, year):
        # HACK (temporary)
        if self.resource_type == "solar":
            data_source = "NSRDB"
        elif self.resource_type == "wind":
            data_source = "WTK-LED"
        ts_data_dir = self.system.base_dir / f"{self.resource_type}-data" / "profiles"
        ts_data_path = ts_data_dir / f"bus{self.node_ID}" / f"{data_source}_profile_bus{self.node_ID}_{year}.csv"
        df_profile = pd.read_csv(ts_data_path, index_col=[0])
        df_profile.index = pd.to_datetime(df_profile.index)
        self.gen_profile = df_profile.squeeze()

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.pmax.value = self.nameplate_capacity * self.gen_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values

    @property
    def pmax_duals(self):
        return self.pmax_constraint.dual_value * (self.pmax.value / self.nameplate_capacity)

class StorageResource(Resource):

    def __init__(self, name, data, duration=4.0):
        super().__init__(name, data)

        # Cost attributes
        self.variable_costs = self.VOM

        # Operational attributes
        self.duration = duration
        self.max_SOC = self.pmax * self.duration
        self.efficiency = data["Storage Roundtrip Efficiency"] / 100

    def create_variables(self):
        self.charge = cp.Variable(T, nonneg=True)
        self.discharge = cp.Variable(T, nonneg=True)
        self.SOC = cp.Variable(T, nonneg=True)

    def create_expressions(self):
        self.p_out = self.discharge - self.charge

    def write_constraints(self):
        constraints = [
            self.charge <= self.pmax,
            self.discharge <= self.pmax,
            self.SOC <= self.max_SOC
        ]
        constraints += [
            self.SOC[t] == self.SOC[np.mod(t - 1, T)] + self.efficiency * self.charge[np.mod(t - 1, T)]
            - self.discharge[np.mod(t - 1, T)]
            for t in range(T)
        ]
        self.pmax_charge_constraint = constraints[0]
        self.pmax_discharge_constraint = constraints[1]
        self.max_SOC_constraint = constraints[2]
        return constraints

    @property
    def pmax_duals(self):
        return self.pmax_charge_constraint.dual_value + self.pmax_discharge_constraint.dual_value

    @property
    def max_SOC_duals(self):
        return self.max_SOC_constraint.dual_value

### SYSTEM ###

class System:
    def __init__(self, base_dir, system_dir, system_config):

        # Save base directory, system directory
        self.base_dir = base_dir
        self.system_dir = system_dir
        self.system_config = system_config

        # Current date for OPF dispatch results (placeholder)
        self.opf_date = None

        # Level of perfect capacity
        self.perfect_capacity = 0

        # Read in data for IEEE RTS GMLC (Reliability Test System)
        df_bus = pd.read_csv(system_dir / system_config / "bus.csv", index_col=[0])
        df_line = pd.read_csv(system_dir / system_config / "branch.csv", index_col=[0])
        df_gen = pd.read_csv(system_dir / system_config / "gen.csv", index_col=[0])

        # Select unit types
        unit_types = ["CT", "STEAM", "CC", "NUCLEAR", "PV", "RTPV", "WIND", "STORAGE"]
        # Select columns
        columns = ["Bus ID", "Unit Type", "Category", "Fuel", "Fuel Price $/MMBTU", "HR_avg_0", "VOM", "PMax MW", "PMin MW", "Ramp Rate MW/Min", "FOR", "Storage Roundtrip Efficiency"]
        # Get final resource input data
        df_gen = df_gen.loc[df_gen["Unit Type"].isin(unit_types), columns]

        # 1. Instantiate nodes
        self.nodes = {}
        for node in df_bus.index:
            self.nodes[node] = Node.from_series(df_bus.loc[node])

        # 2. Instantiate lines
        self.lines = {}
        for line in df_line.index:
            self.lines[line] = Line.from_series(df_line.loc[line])

        # 3. Instantiate resources
        thermal_resource_types = ["CT", "STEAM", "CC", "NUCLEAR"]
        variable_resource_types = ["PV", "RTPV", "WIND"]
        storage_resource_types = ["STORAGE"]
        self.thermal_resources = {}
        self.variable_resources = {}
        self.storage_resources = {}
        for resource in df_gen.index:
            unit_type = df_gen.loc[resource, "Unit Type"]
            if unit_type in thermal_resource_types:
                self.thermal_resources[resource] = ThermalResource.from_series(df_gen.loc[resource])
            elif unit_type in variable_resource_types:
                self.variable_resources[resource] = VariableResource.from_series(df_gen.loc[resource])
            elif unit_type in storage_resource_types:
                self.storage_resources[resource] = StorageResource.from_series(df_gen.loc[resource])

        # 4. Link various components
        for obj in self.components:
            obj.link_system(self) # Link component to system
        for resource in self.resources.values():
            resource.link_node(self.nodes)  # Link resource to node
        for line in self.lines.values():
            line.link_nodes(self.nodes)  # Link line to nodes

    @property
    def resources(self):
        # Create dictionary of all resources
        return self.thermal_resources | self.variable_resources | self.storage_resources

    @property
    def components(self):
        # Create list of components
        return list(self.nodes.values()) + list(self.lines.values()) + list(self.resources.values())

    def read_timeseries(self, year):
        for node in self.nodes.values():
            node.get_load_profile(year)
        for resource in self.variable_resources.values():
            resource.get_gen_profile(year)

    def write_opf(self):
        # Create CVXPY model

        ### Parameters, Variables, Expressions ###

        # Create parameters, variables, and expressions for all objects
        for obj in self.components:
            obj.create_parameters()
            obj.create_variables()
            obj.create_expressions()

        ### Constraints ###

        # Write constraints for all system components
        self.constraints = []
        for obj in self.components:
            self.constraints += obj.write_constraints()

        ### Objective Function ###

        # Get total system variable costs
        self.total_variable_costs = cp.sum(cp.sum([cp.multiply(r.variable_costs, r.p_out) for r in self.resources.values()]))
        self.total_unserved_energy_costs = cp.sum(cp.sum([n.VOLL * n.unserved_energy for n in self.nodes.values()]))
        self.total_unserved_energy = cp.sum(cp.sum([n.unserved_energy for n in self.nodes.values()]))
        self.total_cost = self.total_variable_costs + self.total_unserved_energy_costs # + startup costs + ... etc.
        self.objective = cp.Minimize(self.total_unserved_energy)

        ### Problem ###
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve_opf(self, date):
        # Solve CVXPY model
        self.opf_date = date

        # Update time series parameters
        for obj in self.components:
            obj.update_timeseries_parameters(date)

        # Update perfect capacity at all nodes
        n = len(self.nodes)
        for node in self.nodes.values():
            node.update_perfect_capacity_parameter((1 / n) * self.perfect_capacity)

        # Solve model
        result = self.prob.solve(solver=cp.GUROBI)

        return result

    @property
    def dispatch_results(self):
        # Collect dispatch results
        df_dispatch = pd.DataFrame()
        df_dispatch["load"] = sum(node.load.value - node.perfect_capacity.value for node in self.nodes.values())
        df_dispatch["thermal"] = sum(resource.p_out.value for resource in self.thermal_resources.values())
        df_dispatch["solar"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "solar")
        df_dispatch["wind"] = sum(resource.p_out.value for resource in self.variable_resources.values() if resource.resource_type == "wind")
        df_dispatch["storage charge"] = sum(resource.charge.value for resource in self.storage_resources.values())
        df_dispatch["storage discharge"] = sum(resource.discharge.value for resource in self.storage_resources.values())
        df_dispatch["unserved energy"] = sum(node.unserved_energy.value for node in self.nodes.values())

        # Correct index
        df_dispatch.index = pd.date_range(start=self.opf_date, periods=24, freq="h", tz="UTC")

        return df_dispatch

    @staticmethod
    def dispatch_plot(df_dispatch):
        # Copy dataframe
        df_dispatch = df_dispatch.copy()

        # Correct index
        df_dispatch = pd.concat([df_dispatch.iloc[8:], df_dispatch.iloc[0:8]])
        df_dispatch = df_dispatch.reset_index()

        plt.figure(figsize=(9, 5))
        time = df_dispatch.index
        plt.plot(
            time,
            df_dispatch["load"],
            color="black",
            linewidth=3,
            label="load",
        )
        plt.plot(
            time,
            df_dispatch["load"] + df_dispatch["storage charge"],
            color="black",
            linewidth=2,
            linestyle="--",
            label="load + storage charge",
        )
        plt.stackplot(
            time,
            df_dispatch["solar"], df_dispatch["wind"], df_dispatch["thermal"], df_dispatch["storage discharge"],
            df_dispatch["unserved energy"],
            labels=("solar", "wind", "thermal", "storage discharge", "unserved energy"),
            colors=("gold", "skyblue", "lightgrey", "purple", "red"),
        )
        plt.legend(fontsize=9)
        plt.axis([0, 23, 0, None])
        plt.xlabel("Time (h)")
        plt.ylabel("MW")
        plt.show()

    def simulate_year(self, year):

        # Read in time series data for  year
        self.read_timeseries(year)

        # Create CVXPY OPF model
        self.write_opf()

        # Store results
        df_results = pd.DataFrame(
            index=pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59", freq="d"),
            columns=["Cost", "Unserved Energy"],
        )
        df_LMP = pd.DataFrame(
            index=pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59", freq="h"),
            columns=list(self.nodes.keys()),
        )
        df_MRI = pd.DataFrame(
            index=pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59", freq="h"),
            columns=list(self.nodes.keys()),
        )

        # Simulate each day
        for date in df_results.index:
            # Re-solve model
            self.solve_opf(date)

            # Save results
            df_results.loc[date, "Cost"] = self.total_variable_costs.value
            df_results.loc[date, "Unserved Energy"] = self.total_unserved_energy.value
            for n in self.nodes.keys():
                df_LMP.loc[date:date + pd.Timedelta(hours=T - 1), n] = -self.nodes[n].power_balance_constraint.dual_value

        return df_results, df_LMP