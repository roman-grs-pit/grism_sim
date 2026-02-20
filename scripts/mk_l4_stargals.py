import numpy as np
from astropy.table import Table, Column, vstack, hstack
import roman_datamodels as rdm
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

def roman_datapath(*args):
    """
    Prepends the Roman data path. Looks to see if 
    the `github_dir` variable is set and uses that, otherwise
    tries `ROMAN_DATA` (which is what I use on the Yale Grace system)
    """
    if os.getenv("github_dir") is not None:
        base_path = os.path.join(os.getenv("github_dir"), *args)
    elif os.getenv("ROMAN_DATA") is not None:
        base_path = os.path.join(os.getenv("ROMAN_DATA"), *args)
    else:
        raise EnvironmentError("Neither 'github_dir' nor 'ROMAN_DATA' environment variables are set.")
    return base_path

def load_star_catalog(star_fn="stars/sim_star_cat_galacticus.ecsv"):
    """
    Load a star catalog from a given filename.
    """
    star_cat = Table.read(roman_datapath(star_fn), format='ascii.ecsv')
    return star_cat

def load_gal_catalog(gal_fn="galacticus_4deg2_mock/Euclid_Roman_4deg2_radec.fits"):
    """
    Load a galaxy catalog from a given filename.
    """
    gal_cat = Table.read(roman_datapath(gal_fn), format='fits')
    return gal_cat

def load_skycells(skycell_fn="misc/roman_wfi_skycells_0001.asdf"):
    """
    Load skycell data from a given filename. 

    If the path is not an absolute path, it is assumed to be
    relative to the Roman data path.
    """
    fn = skycell_fn
    if not os.path.isabs(fn):
        fn = roman_datapath(fn)
    skycell_dm = rdm.open(fn)
    return skycell_dm

def find_skycells(cat, skycell_dm=None):
    """
    Build an auxiliary table that maps stars to tiles and skycells.

    Arguments:
    cat : Astropy table with columns 'ra','dec' (allows for simple variants)
    skycell_dm: if none, reads in the default skycell file.
        if a string, reads in that file.
        if a datamodel, uses that directly.
    Returns:
    aux_table : Astropy table with columns 'tile_id', 'skycells' 
        mapping each star to its tile and skycell.

    This code is based on Roman-STScI-000708, Sec 3.7
    """
    # Load in the skycells if not passed
    if skycell_dm is None:
        skycell_dm = load_skycells()
    elif isinstance(skycell_dm, str):
        skycell_dm = load_skycells(skycell_dm)

    tiles = skycell_dm.projection_regions
    skycells = skycell_dm.skycells

    # Determine the maximum distance we would expect between an object
    # and the center of its skycell.
    nxy_skycell = skycell_dm.meta.nxy_skycell
    pixel_scale = skycell_dm.meta.pixel_scale  # arcsec/pixel
    max_dist_arcsec = 0.5 * np.sqrt(2.0) * nxy_skycell * pixel_scale

    # Get RA/Dec of objects
    if 'ra' in cat.colnames:
        racat = cat['ra']
    elif 'RA' in cat.colnames:
        racat = cat['RA']
    else:
        raise ValueError("Catalog must have 'ra' or 'RA' column.")
    
    if 'dec' in cat.colnames:
        deccat = cat['dec']
    elif 'DEC' in cat.colnames:
        deccat = cat['DEC']
    elif 'Dec' in cat.colnames:
        deccat = cat['Dec']
    else:
        raise ValueError("Catalog must have 'dec', 'DEC', or 'Dec' column.")

    # Loop over the tiles, matching objects to tiles
    tile_id = -1 * np.ones(len(cat), dtype=np.int64)
    cell_id = -1 * np.ones(len(cat), dtype=np.int64)
    match_dist = np.zeros(len(cat), dtype=np.float64)
    cell_name = np.empty(len(cat), dtype='<U16')
    for itile, tile in enumerate(tiles):
        ramin, ramax = tile['ra_min'], tile['ra_max']
        decmin, decmax = tile['dec_min'], tile['dec_max']
        in_tile = (racat >= ramin) & (racat < ramax) & (deccat > decmin) & (deccat <= decmax)

        # Do a check here to make sure that these objects have not 
        # already been assigned a tile/skycell
        # Raise a warning if they have
        if np.any(tile_id[in_tile] != -1):
            already_assigned = np.sum(tile_id[in_tile] != -1)
            print(f"Tile {itile} has {already_assigned} already assigned objects.")
            print("WARNING: Some objects have already been assigned a tile/skycell.")
            print("This may indicate overlapping tiles.")
            print("WARNING: TILE ASSIGNMENT ISSUE DETECTED.")
        if np.any(cell_id[in_tile] != -1):
            # Return the tile id and number of objects
            already_assigned = np.sum(cell_id[in_tile] != -1)
            print(f"Tile {itile} has {already_assigned} already assigned objects.")
            print("WARNING: Some objects have already been assigned a tile/skycell.")
            print("This may indicate overlapping tiles.")
            print("WARNING: SKYCELL ASSIGNMENT ISSUE DETECTED.")
        tile_id[in_tile] = itile
        
        # Trim down to objects in this tile
        obj_ras = racat[in_tile]
        obj_decs = deccat[in_tile]
        obj_coords = SkyCoord(ra=obj_ras*u.deg, dec=obj_decs*u.deg)

        # If no objects in this tile, skip
        if np.sum(in_tile) == 0:
            continue

        # Now loop over skycells within each tile    
        start, end = tile['skycell_start'], tile['skycell_end']
        # This logic is directly from Roman-STScI-000708, Sec 3.7 
        # and it seems that the end is exclusive. The datamodel does not say anything on this.
        cells = skycells[start:end]
        cell_centers  = SkyCoord(ra=cells['ra_center']*u.deg, dec=cells['dec_center']*u.deg)

        # Match objects to skycells
        # All objects will match a skycell since we are within the tile
        idx, d2d, _ = obj_coords.match_to_catalog_sky(cell_centers)
        cell_id[in_tile] = idx + start  # Adjust index to global skycell index
        match_dist[in_tile] = d2d.arcsec
        cell_name[in_tile] = cells['name'][idx]

    # Check that all objects were assigned
    if np.any(tile_id == -1):
        n_unassigned = np.sum(tile_id == -1)
        raise ValueError(f"{n_unassigned} objects were not assigned to any tile.")
    if np.any(cell_id == -1):
        n_unassigned = np.sum(cell_id == -1)
        raise ValueError(f"{n_unassigned} objects were not assigned to any skycell.")
    # A little padding here....
    if np.any(match_dist > 1.1 * max_dist_arcsec):
        n_too_far = np.sum(match_dist > 1.1 * max_dist_arcsec)
        raise ValueError(f"{n_too_far} objects were assigned to skycells farther than the maximum expected distance.")

    aux_table = Table()
    aux_table['tile_id'] = tile_id
    aux_table['skycell_id'] = cell_id
    aux_table['skycell_name'] = cell_name
    aux_table['match_dist_arcsec'] = match_dist

    return aux_table

# Write a function that creates an empty multiband source catalog
# with specified filters, and a specified number of rows. If filters are
# not specified, use all filters.
#
# This also adds in a column for "sourceid", sets the unit to None, and adds in a 
# description "Unique source identifier". Make this column an int64 type.
#
# Note that the sourceid column is not part of the standard Roman L4 catalog, nor is
# in the data model, but it seems to still work (at least for now).
def create_empty_L4_catalog(filters=None, n_rows=1):
    # Assert that n_rows > 0
    if n_rows < 1:
        raise ValueError("n_rows must be greater than or equal to 1.")

    if filters is None:
        filters = ['f062', 'f087', 'f106', 'f129', 'f146', 'f158', 'f184', 'f213']
    m = rdm.datamodels.MultibandSourceCatalogModel()
    tab = m.create_empty_catalog(filters=filters)

    # Add sourceid column
    sourceid_col = Column(
        np.zeros(0, dtype=np.int64), 
        name="sourceid",
        unit='none',
        description="Unique source identifier"
    )
    tab.add_column(sourceid_col)

    # Create a dictionary with column names as keys, arrays as values
    if n_rows > 0:
        # Create a dictionary with column names as keys, arrays as values
        # Preserve metadata by using Column objects
        col_dict = {}
        for col in tab.colnames:
            col_dict[col] = Column(
                np.zeros(n_rows, dtype=tab[col].dtype),
                name=col,
                unit=tab[col].unit,
                description=tab[col].description
            )
    
    # Create new table from column dictionary and stack
    new_rows = Table(col_dict)
    tab = vstack([tab, new_rows])
    return tab


# The code below writes out L4 catalogs in parquet format 
# for stars, based on the our trimmed star catalog.
# 
# Here are the mappings from the star catalog data
#
# The ASDF data model defines 
#   fwhm = 2*sqrt(ln(2) * (semimajor**2 + semiminor**2)
# We only populate the following columns
# 
#  ra, dec = RA, DEC
#  f158_kron_abmag = magnitude
#  f158_kron_abmag_err = 0
#  fwhm = 0.158 # PSF for f158
#  semimajor = 0.158 / 2*sqrt(2 ln(2))
#  semiminor = 0.158 / 2*sqrt(2 ln(2))
#  is_extended_f158 = False
#  orientation_sky = 0
#  sourceid = index # Not a column in the datamodel, but useful to track
#
#
# References:
#  https://rad.readthedocs.io/en/latest/generated/schemas/tables/source_catalog_columns-1.0.0.html

# 
def mk_stargal_L4_catalogs():
    outpath=roman_datapath("skycell_catalogs")
    print(f"Writing star and galaxy L4 catalogs to {outpath}")
    # Check to see if the output path exists, and create it if not
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Build the star catalog and auxiliary table
    star_cat = load_star_catalog()
    star_aux_table = find_skycells(star_cat)
    full_star_cat = hstack([star_cat, star_aux_table])

    # Do the same for the galaxy catalog
    gal_cat = load_gal_catalog()
    gal_aux_table = find_skycells(gal_cat)
    full_gal_cat = hstack([gal_cat, gal_aux_table])

    # Combine the skycell names from the star and galaxy catalogs and get 
    # the unique set of skycells
    all_skycells = np.unique(np.concatenate([full_star_cat['skycell_name'], full_gal_cat['skycell_name']]))
    print(f"Found {len(all_skycells)} unique skycells across stars and galaxies.")

    # Loop over the skycells
    star_fwhm_value = 0.158  # arcsec
    gal_r_eff_value = 2.5*0.11 # arcsec, based on half-light radius of 2.5 pixels and pixel scale of 0.11 arcsec/pixel
    gal_fwhm_value = 0.83 * gal_r_eff_value # arcsec, based on FWHM ~ 0.83 * r_eff for a Sersic profile with n=1
    for skycell_name in all_skycells:    
        # Select the stars and galaxies in this skycell
        in_skycell_stars = full_star_cat['skycell_name'] == skycell_name
        in_skycell_gals = full_gal_cat['skycell_name'] == skycell_name
        cat1 = full_star_cat[in_skycell_stars]
        cat2 = full_gal_cat[in_skycell_gals]

        n_stars = len(cat1)
        n_gals = len(cat2)
        n_total = n_stars + n_gals
        print(f"Processing skycell {skycell_name} with {n_stars} stars, {n_gals} galaxies, {n_total} total objects.")

        # Create empty L4 catalog
        l4_cat = create_empty_L4_catalog(n_rows=n_total)

        # Populate the columns, first with stars
        l4_cat['ra'][0:n_stars] = cat1['RA']
        l4_cat['dec'][0:n_stars] = cat1['DEC']
        l4_cat['kron_f158_abmag'][0:n_stars] = cat1['magnitude']
        l4_cat['kron_f158_abmag_err'][0:n_stars] = 0.0
        l4_cat['fwhm'][0:n_stars] = star_fwhm_value
        semimajor_value = star_fwhm_value / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        l4_cat['semimajor'][0:n_stars] = semimajor_value
        l4_cat['semiminor'][0:n_stars] = semimajor_value
        l4_cat['is_extended_f158'][0:n_stars] = False
        l4_cat['orientation_sky'][0:n_stars] = 0.0
        l4_cat['sourceid'][0:n_stars] = np.arange(n_stars, dtype=np.int64)

        # Now populate the columns for galaxies
        l4_cat['ra'][n_stars:n_total] = cat2['RA']
        l4_cat['dec'][n_stars:n_total] = cat2['DEC']
        l4_cat['kron_f158_abmag'][n_stars:n_total] = cat2['mag_F158_Av1.6523']
        l4_cat['kron_f158_abmag_err'][n_stars:n_total] = 0
        l4_cat['fwhm'][n_stars:n_total] = gal_fwhm_value
        l4_cat['semimajor'][n_stars:n_total] = gal_fwhm_value / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        l4_cat['semiminor'][n_stars:n_total] = gal_fwhm_value / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        l4_cat['is_extended_f158'][n_stars:n_total] = True
        l4_cat['orientation_sky'][n_stars:n_total] = 0.0
        l4_cat['sourceid'][n_stars:n_total] = cat2['unique_ID']

        # Write out the catalog in parquet format
        out_fn = os.path.join(outpath, f"stargal_catalog_{skycell_name}.parquet")
        l4_cat.write(out_fn, format='parquet', overwrite=True)
        print(f"Wrote star and galaxy L4 catalog to {out_fn}")

if __name__ == "__main__":
    mk_stargal_L4_catalogs()