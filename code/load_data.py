import numpy as np
import pandas as pd
import pathlib
import warnings
warnings.simplefilter("ignore", FutureWarning)

from demand_ninja.core import demand

curr_dir = pathlib.Path(".")
data_dir = curr_dir / "RTS-GMLC-master" / "RTS_Data"
load_data_dir = curr_dir / "load-data"

def create_load_profiles():

    ### Create load profiles; normalize to 2006 load data

    # Read in list of buses
    df_bus = pd.read_csv(curr_dir / "RTS-GMLC-master" / "RTS_Data" / "SourceData" / "bus.csv", index_col=[0])

    for bus in df_bus.index:
        print(bus)
        skip_bus = False
        years = np.arange(1998, 2024)
        df_in_concat = pd.DataFrame()
        for year in years:
            fname = load_data_dir / "inputs" / f"bus{bus}" / f"NSRDB_weather-inputs_bus{bus}_{year}.csv"
            if not fname.exists():
                skip_bus = True
                continue
            df_in = pd.read_csv(fname, index_col=["datetime"])
            df_in.index = pd.to_datetime(df_in.index)

            # Get subset of columns needed for load calculation in demand ninja
            df_in = df_in[["Temperature", "Relative Humidity", "GHI", "Wind Speed"]]
            df_in.columns = ["temperature", "humidity", "radiation_global_horizontal", "wind_speed_2m"]
            df_in_concat = pd.concat([df_in_concat, df_in])
        df_in = df_in_concat.copy()

        if skip_bus:
            print("Skipping bus", bus)
            continue

        # Calculate demand with demand ninja
        df_out = demand(
            df_in,
            base_power=10,
            heating_power=0.8,
            cooling_power=1.2,
            cooling_threshold=16.0,
            heating_threshold=24.0,
            humidity_discomfort=0.003,
            solar_gains=0.01,
            smoothing=0.5,
        )["total_demand"]

        # Normalize by peak load in 2006
        peak_2006 = df_out.loc[df_out.index.year == 2006].max()
        df_out /= peak_2006

        # Save data
        # Profiles directory
        bus_dir = load_data_dir / "profiles" / f"bus{bus}"
        bus_dir.mkdir(exist_ok=True, parents=True)
        for year in years:
            df_out_year = df_out.loc[df_out.index.year == year]
            df_out_year.to_csv(bus_dir / f"NSRDB_load-profile_bus{bus}_{year}.csv")