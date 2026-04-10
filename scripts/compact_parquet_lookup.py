import argparse
import json
from pathlib import Path

import polars as pl


def suggest_columns(compact_path: str) -> dict[str, list[str]]:
    """Suggest potentially useful compact columns for keying and line flux queries."""
    cols = pl.scan_parquet(compact_path).collect_schema().names()

    id_like = [c for c in cols if any(k in c.lower() for k in ("src", "idx", "id", "sim"))]
    line_like = [c for c in cols if any(k in c.lower() for k in ("line", "flux", "ha", "oiii", "hb", "nii", "sii"))]

    return {
        "id_like": id_like[:50],
        "line_like": line_like[:200],
    }


def lookup_compact_rows(
    compact_path: str,
    sim: int | list[int],
    src_index: int | list[int],
    compact_src_col: str | None = "src_index",
    compact_sim_col: str = "sim",
    select_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Retrieve rows from compact parquet associated with one or more (sim, src_index) pairs.
    """
    schema = pl.scan_parquet(compact_path).collect_schema()
    names = set(schema.names())

    if compact_src_col is None or compact_src_col not in names or compact_sim_col not in names:
        suggestions = suggest_columns(compact_path)
        raise ValueError(
            "Could not determine lookup key in compact parquet. "
            f"Expected {compact_src_col!r} or {compact_sim_col!r} with --fits. "
            f"Suggested id-like columns: {suggestions['id_like'][:10]}"
        )

    sims = sim if isinstance(sim, (list, tuple)) else [sim]
    src_indices = src_index if isinstance(src_index, (list, tuple)) else [src_index]
    if len(sims) != len(src_indices):
        raise ValueError("--sim and --src-index must have the same number of values")

    keys = pl.DataFrame({"__order": range(len(sims)), compact_sim_col: sims, compact_src_col: src_indices})
    lf = pl.scan_parquet(compact_path).join(keys.lazy(), on=[compact_sim_col, compact_src_col], how="inner")

    keep: list[str] | None = None
    if select_columns:
        keep = [c for c in select_columns if c in names]
        if not keep:
            raise ValueError("None of the requested --columns exist in compact parquet")

    result = lf.collect().sort("__order").drop("__order")
    if keep is not None:
        result = result.select(keep)
    return result


def parse_columns_arg(columns_arg: str | None) -> list[str] | None:
    """Parse comma-separated column list from CLI."""
    if columns_arg is None:
        return None

    cols = [c.strip() for c in columns_arg.split(",") if c.strip()]
    return cols or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lookup rows in compact parquet by src_index, or by catalog object id that resolves to src_index."
        )
    )
    parser.add_argument("--compact", required=True, help="Path to compact parquet")
    parser.add_argument("--sim", type=int, nargs="+", help="One or more sim numbers from catalog")
    parser.add_argument("--src-index", type=int, nargs="+", help="One or more src_index values")

    parser.add_argument("--catalog", help="Catalog parquet path (used with --object-id)")
    parser.add_argument("--object-id", help="Object identifier to resolve from catalog")
    parser.add_argument(
        "--object-id-col",
        default="id",
        help="Catalog column containing object ids",
    )
    parser.add_argument(
        "--catalog-src-col",
        default="src_index",
        help="Catalog column containing src_index",
    )

    parser.add_argument(
        "--compact-src-col",
        default="src_index",
        help="Compact parquet src_index column name (if present)",
    )
    parser.add_argument(
        "--compact-sim-col",
        default="sim",
        help="Compact parquet sim column name",
    )
    
    parser.add_argument(
        "--columns",
        help="Comma-separated subset of columns to return",
    )
    parser.add_argument(
        "--line-columns-only",
        action="store_true",
        help="Select only columns that look like line/flux measurements",
    )
    parser.add_argument(
        "--show-suggested-columns",
        action="store_true",
        help="Print suggested id/line columns and exit",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=20,
        help="Rows to print in terminal output",
    )
    parser.add_argument(
        "--out",
        help="Optional output path (.parquet, .csv, or .json)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    compact_path = args.compact

    if args.show_suggested_columns:
        suggestions = suggest_columns(compact_path)
        print("Suggested id-like columns:")
        for col in suggestions["id_like"]:
            print(f"  {col}")

        print("\nSuggested line/flux columns:")
        for col in suggestions["line_like"]:
            print(f"  {col}")
        return

    src_index = args.src_index
    sim = args.sim

    select_columns = parse_columns_arg(args.columns)
    if args.line_columns_only:
        line_like = suggest_columns(compact_path)["line_like"]
        if select_columns is None:
            select_columns = line_like
        else:
            select_columns = [c for c in select_columns if c in set(line_like)]

    result = lookup_compact_rows(
        compact_path=compact_path,
        sim=sim,
        src_index=src_index,
        compact_src_col=args.compact_src_col,
        compact_sim_col=args.compact_sim_col,
        select_columns=select_columns,
    )

    print(f"matched {result.height} rows, {result.width} columns")
    if result.height:
        print(result.head(args.head))

    if args.out:
        out_path = Path(args.out)
        suffix = out_path.suffix.lower()
        if suffix == ".parquet":
            result.write_parquet(out_path)
        elif suffix == ".csv":
            result.write_csv(out_path)
        elif suffix == ".json":
            out_path.write_text(json.dumps(result.to_dicts(), indent=2))
        else:
            raise ValueError("--out must end with .parquet, .csv, or .json")
        print(f"Wrote lookup result to {out_path}")


if __name__ == "__main__":
    main()
