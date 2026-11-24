
import xarray as xr

FILES = {
    "ACE": "ace.nc",
    "Wave-Gauss": "wg.nc",
}

for name, path in FILES.items():
    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    ds = xr.open_dataset(path)

    print("\nVariables:")
    for var in ds.data_vars:
        print(f"  {var}: shape={ds[var].shape}, dims={ds[var].dims}, dtype={ds[var].dtype}")

    print("\nCoordinates:")
    for coord in ds.coords:
        print(f"  {coord}: shape={ds[coord].shape}, dims={ds[coord].dims}")

    print("\nAttributes:")
    print(ds.attrs)

    ds.close()