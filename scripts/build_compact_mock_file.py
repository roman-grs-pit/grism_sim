from glob import glob
from pathlib import Path

from astropy.io import fits
import h5py
import polars as pl

from tqdm import tqdm


OUTPUT_PARQUET = "galacticus_mock_compact.parquet"
HDF5_GLOB = "/Volumes/SSD/galacticus_4deg2_mock/galacticus_FOV_EVERY100_sub_*.hdf5"
FITS_FILE = "/Volumes/SSD/galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits"


def iter_datasets(group: h5py.Group, prefix: str = ""):
    """Yield all datasets in an HDF5 group recursively with full path names."""
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            yield path, item
        elif isinstance(item, h5py.Group):
            yield from iter_datasets(item, path)


def dataset_to_values(dataset: h5py.Dataset):
    """Return dataset values suitable for Polars, or None when unsupported."""
    values = dataset[()]

    # Scalar datasets cannot be joined row-wise with catalog arrays.
    if getattr(values, "shape", ()) == ():
        return None

    # Keep only 1D arrays to preserve a tabular layout.
    if len(values.shape) != 1:
        return None

    # Decode byte strings to UTF-8 strings for Parquet compatibility.
    if values.dtype.kind == "S":
        return [v.decode("utf-8", errors="ignore") for v in values]

    return values


def normalize_column_name(dataset_path: str) -> str:
    """Drop a leading Outputs group and convert path separators for flat columns."""
    parts = dataset_path.split("/", 1)
    if parts[0].lower() == "outputs" and len(parts) == 2:
        dataset_path = parts[1]

    return dataset_path.replace("/", "__").replace(":", "_")


def hdf5_to_frame(file_path: str, fits_data) -> pl.DataFrame | None:
    """Load one mock HDF5 file into a DataFrame, skipping SED datasets."""
    columns: dict[str, object] = {}
    row_count: int | None = None

    with h5py.File(file_path, "r") as h5f:
        for dataset_path, dataset in iter_datasets(h5f):
            # Exclude all SED content.
            if "sed" in dataset_path.lower():
                continue

            values = dataset_to_values(dataset)
            if values is None:
                continue

            if row_count is None:
                row_count = len(values)

            # Only keep columns that align with the inferred row count.
            if len(values) != row_count:
                continue

            col_name = normalize_column_name(dataset_path)
            columns[col_name] = values

    if not columns or row_count is None:
        return None

    sim = int(Path(file_path).stem.split("_")[-1])
    columns["sim"] = [sim] * row_count
    columns["src_index"] = list(fits_data["IDX"][fits_data["SIM"] == sim])
    columns["source_file"] = [Path(file_path).name] * row_count
    return pl.DataFrame(columns)


def main() -> None:
    file_paths = sorted(glob(HDF5_GLOB))
    if not file_paths:
        raise RuntimeError(f"No HDF5 files matched pattern: {HDF5_GLOB}")

    with fits.open(FITS_FILE) as f:
        fits_data = f[1].data  # type: ignore
        frames: list[pl.DataFrame] = []
        for file_path in tqdm(file_paths):
            frame = hdf5_to_frame(file_path, fits_data)
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No usable non-SED tabular datasets found in input files")

    compact = pl.concat(frames, how="diagonal_relaxed")
    compact.write_parquet(OUTPUT_PARQUET)
    print(f"Wrote {compact.height} rows and {compact.width} columns to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
