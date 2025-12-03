import numpy as np
import pandas as pd
import cvxpy as cp
import pathlib

# Object Oriented CVXPY Model

T = 24  # Time window of optimization model(s)

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
        self.theta_max = np.deg2rad(30)  # 30 degree maximum voltage angle

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

    # def get_load_profile(self, ts_pointer):
    #     # Open / store load profile for object (can definitely make this more efficient)
    #     ts_rel_path = ts_pointer.loc[
    #         (ts_pointer.Simulation == "REAL_TIME") &
    #         (ts_pointer.Parameter == "MW Load") &
    #         (ts_pointer.Object == str(self.area)),
    #         "Data File"
    #     ].values[0].strip("..")  # Get path to load profile from time series pointer
    #     ts_path = pathlib.Path(str(self.system.system_dir) + ts_rel_path)
    #     df_load = pd.read_csv(ts_path)  # Read in load data
    #     df_load = df_load.loc[::12, str(self.area)]  # Convert to hourly
    #     df_load /= df_load.max()  # Normalize
    #     df_load.index = pd.date_range(start="2020-01-01 00:00:00", end="2020-12-31 23:59:59",
    #                                   freq="h")  # Add timeseries index
    #     self.load_profile = df_load

    def get_load_profile(self, year):
        data_source = "NSRDB"
        ts_data_dir = self.system.base_dir / "load-data" / "profiles"
        ts_data_path = ts_data_dir / f"bus{self.name}" / f"{data_source}_load-profile_bus{self.name}_{year}.csv"
        df_load = pd.read_csv(ts_data_path, index_col=[0])
        df_load.index = pd.to_datetime(df_load.index)
        self.load_profile = df_load.squeeze()

    def create_parameters(self):
        self.load = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.load.value = self.peak_load * self.load_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values

    def create_variables(self):
        self.unserved_energy = cp.Variable(T, nonneg=True)
        self.theta = cp.Variable(T)

    def write_constraints(self):
        constraints = [
            cp.sum([r.p_out for r in self.resources])  # Nodal generation
            + cp.sum([l.flow for l in self.in_lines])  # + line inflows
            - cp.sum([l.flow for l in self.out_lines])  # - line outflows
            >= self.load - self.unserved_energy,  # = load minus unserved energy (power balance constraint)
            self.unserved_energy <= self.load,  # Limit unserved energy
            self.theta <= self.theta_max, self.theta >= -self.theta_max  # Voltage angle limits
        ]
        self.power_balance_constraint = constraints[0]
        if self.node_type == "Ref":
            constraints += [self.theta == 0.0]
        return constraints


### LINE ###

class Line(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack line attributes from data
        self.from_node_ID = data["From Bus"]
        self.to_node_ID = data["To Bus"]
        self.max_flow = data[ "LTE Rating"]  # Long-term flow limit; alternatively can use continuous/short-term limits
        self.X = data["X"]  # Line susceptance # MVA base = 100?

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

    def create_variables(self):
        self.flow = cp.Variable(T)

    def write_constraints(self):
        return [
            self.flow <= self.max_flow,
            self.flow >= -self.max_flow,
            self.flow == (1 / self.X) * self.baseMVA * (self.from_node.theta - self.to_node.theta)
        ]


### RESOURCES ###

class Resource(Component):

    def __init__(self, name, data):
        super().__init__()

        self.name = name
        self.data = data

        # Unpack basic resource attributes from data
        self.node_ID = data["Bus ID"]  # Bus ID
        self.unit_type = data["Unit Type"]  # Unit type
        self.pmax = data["PMax MW"]  # Nameplate capacity (MW)
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

    def create_variables(self):
        self.p_out = cp.Variable(T, nonneg=True)

    def write_constraints(self):
        return [self.p_out <= self.pmax]


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

        # Operational attributes
        self.nameplate_capacity = data["PMax MW"]

        # Initialize generation profile
        self.gen_profile = None

    # def get_gen_profile(self, ts_pointer):
    #     # ALREADY KNOWS RESOURCE TYPE, AND BUS ID
    #     ts_rel_path = ts_pointer.loc[
    #         (ts_pointer.Simulation == "REAL_TIME") &
    #         (ts_pointer.Parameter == "PMax MW") &
    #         (ts_pointer.Object == self.name),
    #         "Data File"
    #     ].values[0].strip("..")  # Get path to profile from time series pointer
    #     ts_path = pathlib.Path(str(self.system.system_dir) + ts_rel_path)
    #     df_profile = pd.read_csv(ts_path)  # Read in data
    #     df_profile = df_profile.loc[::12, self.name]  # Convert to hourly
    #     df_profile /= df_profile.max()  # Normalize
    #     df_profile.index = pd.date_range(start="2020-01-01 00:00:00", end="2020-12-31 23:59:59",
    #                                      freq="h")  # Add timeseries index
    #     self.gen_profile = df_profile

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

    def create_parameters(self):
        self.pmax = cp.Parameter(T)

    def update_timeseries_parameters(self, date: pd.Timestamp):
        self.pmax.value = self.nameplate_capacity * self.gen_profile.loc[date:date + pd.Timedelta(hours=T - 1)].values


class StorageResource(Resource):

    def __init__(self, name, data, duration=4):
        super().__init__(name, data)

        # Cost attributes
        self.variable_costs = self.VOM

        # Operational attributes
        self.duration = duration
        self.max_SOC = self.pmax * self.duration
        self.efficiency = data["Storage Roundtrip Efficiency"]

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
            self.SOC[t] == self.SOC[np.mod(t - 1, T)] + self.efficiency * self.charge[np.mod(t - 1, T)] -
            self.discharge[np.mod(t - 1, T)]
            for t in range(T)
        ]
        return constraints


### SYSTEM ###

class System:
    def __init__(self, base_dir, system_dir):

        # Save base directory, system directory
        self.base_dir = base_dir
        self.system_dir = system_dir

        # Read in data for IEEE RTS GMLC (Reliability Test System)
        df_bus = pd.read_csv(system_dir / "SourceData" / "bus.csv", index_col=[0])
        df_line = pd.read_csv(system_dir / "SourceData" / "branch.csv", index_col=[0])
        df_gen = pd.read_csv(system_dir / "SourceData" / "gen.csv", index_col=[0])
        df_ts_pointer = pd.read_csv(system_dir / "SourceData" / "timeseries_pointers.csv")

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
        self.total_cost = self.total_variable_costs + self.total_unserved_energy_costs # + startup costs + ... etc.
        self.objective = cp.Minimize(self.total_cost)

        ### Problem ###
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve_opf(self, date):
        # Solve CVXPY model

        # Update time series parameters
        for obj in self.components:
            obj.update_timeseries_parameters(date)

        # Solve model
        result = self.prob.solve(solver=cp.GUROBI)

        return result