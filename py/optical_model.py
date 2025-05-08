import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


class RomanOpticalModel:
    """
    Unit and label system used --
    SCA: Sensor Chip Assembly (Invid. Detectors); position defined in [pixels]
    FPA: Focal Plane Assembly; position defined in [deg] from center of the SCA
    MPA: Mosaic Plate Assembly; position defined in [mm]
    """

    def __init__(self, config_file="../data/Roman_OpticalModel_v0.5.yaml"):

        # Load the specified model file (YAML format)
        self.config_file = None
        self.load_model(config_file=config_file)

        # Setup the SCA polygons
        self.sca_list = np.array(
            sorted(list(self.config["detector_model"]["xy_centers"].keys())),
            dtype=int,
        )
        self.define_sca_polygons()

        # Vectorize functions
        self.get_beam_coeff_vec = np.vectorize(self._get_beam_coeff)
        self.get_beam_trace_vec = np.vectorize(self._get_beam_trace)

        self.get_beam_coeff = lambda *args, **kwargs: np.atleast_1d(
            self.get_beam_coeff_vec(*args, **kwargs)
        )
        self.get_beam_trace = lambda *args, **kwargs: np.atleast_1d(
            self.get_beam_trace_vec(*args, **kwargs)
        )

    def load_model(self, config_file=None):
        """
        Reads in the configuration file for the optical model
        """
        # Read in the YAML config file
        if config_file is not None:
            self.config_file = config_file

        with open(self.config_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Define some default parameters
        self.pixel_per_mm = self.config["detector_model"]["pixel_per_mm"]
        self.pixel_scale = self.config["detector_model"]["pixel_scale"]
        self.disperser = self.config["optical_model"]["disperser"]
        self.ref_lambda = self.config["optical_model"]["ref_lambda"]

        # Define the wavelength resolution of the returned beam trace info
        self.wvl_min = self.config["optical_model"]["lambda_min"]
        self.wvl_max = self.config["optical_model"]["lambda_max"]
        self.wvl_res = self.config["optical_model"]["lambda_res"]
        self.def_lambdas = np.arange(
            self.wvl_min, self.wvl_max + 1e-6, self.wvl_res
        )

        # Define the polynomial orders
        self.dimen_map = self.config["optical_model"]["dimen_map"]
        self.dimen_crv = self.config["optical_model"]["dimen_crv"]
        self.dimen_ids = self.config["optical_model"]["dimen_ids"]

        # Define the polynomial orders
        self.power_crv = self.config["optical_model"]["power_crv"]
        self.power_ids = self.config["optical_model"]["power_ids"]

        # Cast the polynomial coefficients as arrays
        self.beam_coeffs = {}
        for order in ["+1", "0", "-1", "+2", "+3", "+4"]:
            order_str = f"order_{order:s}"
            if order_str in self.config["optical_model"]:
                self.beam_coeffs[order] = {}
                self.beam_coeffs[order]["X_ij"] = np.array(
                    self.config["optical_model"][order_str]["X_ij"], dtype=float
                )
                self.beam_coeffs[order]["Y_ij"] = np.array(
                    self.config["optical_model"][order_str]["Y_ij"], dtype=float
                )
                self.beam_coeffs[order]["C_ijk"] = np.array(
                    self.config["optical_model"][order_str]["C_ijk"],
                    dtype=float,
                )
                self.beam_coeffs[order]["D_ijk"] = np.array(
                    self.config["optical_model"][order_str]["D_ijk"],
                    dtype=float,
                )

    def define_sca_polygons(self):
        """
        Defines the SCA locations on the MPA (in mm) and their sizes (in pix)
        """
        from matplotlib.patches import Rectangle

        self.sca_polygons = {}
        for sca in self.sca_list:
            xy_cen = self.config["detector_model"]["xy_centers"][sca]
            xy_vrt = np.array(xy_cen) - np.array(
                [
                    self.config["detector_model"]["crpix1"] / self.pixel_per_mm,
                    self.config["detector_model"]["crpix2"] / self.pixel_per_mm,
                ]
            )
            self.sca_polygons[sca] = Rectangle(
                xy_vrt,
                self.config["detector_model"]["x_size"] / self.pixel_per_mm,
                self.config["detector_model"]["y_size"] / self.pixel_per_mm,
                angle=self.config["detector_model"]["pa"],
                rotation_point="center",
                lw=2,
                facecolor="none",
                edgecolor="k",
                alpha=0.9,
            )

    def check_point_in_sca(self, xmpa, ympa):
        """
        Returns matching SCA in which the given point(s) lie
        Units are in 'mm'
        """
        xmpa, ympa = np.atleast_1d(xmpa), np.atleast_1d(ympa)
        pos = np.array([xmpa, ympa]).T
        cond = np.array(
            [
                self.sca_polygons[sca].contains_points(
                    self.sca_polygons[sca].get_data_transform().transform(pos)
                )
                for sca in self.sca_list
            ],
            dtype=bool,
        )
        j, i = np.where(cond)

        sca_match = np.zeros(len(xmpa), dtype=int)
        sca_match[i] = self.sca_list[j]
        return sca_match if len(sca_match) > 1 else sca_match[0]

    def convert_sca_to_mpa(self, xsca, ysca, sca):
        """
        Returns MPA position [mm] for reference position in the SCA [pixel]
        Input: xsca, ysca are source position, in px, in detector plane (SCA)
        Output: xmpa, ympa are source position, in mm, in focal plane (MPA)
        """
        dx = xsca - self.config["detector_model"]["crpix1"]
        dy = ysca - self.config["detector_model"]["crpix2"]

        # Rotation terms might be needed here
        xoff = dx / self.pixel_per_mm
        yoff = dy / self.pixel_per_mm

        xmpa = xoff + self.config["detector_model"]["xy_centers"][sca][0]
        ympa = yoff + self.config["detector_model"]["xy_centers"][sca][1]

        return xmpa, ympa

    def convert_sca_to_fpa(self, xsca, ysca, sca):
        """
        Returns FPA position [degree] for reference position in the SCA [px]
        Input: xsca, ysca are source position, in px, in detector plane (SCA)
        Output: xfpa, yfpa are source position, in deg, in focal plane (FPA)
        """
        dx = xsca - self.config["detector_model"]["crpix1"]
        dy = ysca - self.config["detector_model"]["crpix2"]

        # Get absolute X_FPA and Y_FPA, in degrees
        xcen, ycen = self.config["detector_model"]["xy_centers"][sca]
        xfpa = (xcen * self.pixel_per_mm + dx) * self.pixel_scale / 3600
        yfpa = (ycen * self.pixel_per_mm + dy) * self.pixel_scale / 3600

        return xfpa, yfpa

    def convert_mpa_to_sca(self, xmpa, ympa, sca):
        """
        Returns SCA position [pixel] for reference position in the MPA [mm]
        Input: xmpa, ympa are source position, in px, in detector plane (SCA)
        Output: xfpa, yfpa are source position, in deg, in focal plane (FPA)
        """
        xoff = xmpa - self.config["detector_model"]["xy_centers"][sca][0]
        yoff = ympa - self.config["detector_model"]["xy_centers"][sca][1]

        # Rotation terms might be needed here
        xrot = xoff * self.pixel_per_mm
        yrot = yoff * self.pixel_per_mm

        xdet = xrot + self.config["detector_model"]["crpix1"]
        ydet = yrot + self.config["detector_model"]["crpix2"]

        return xdet, ydet

    def convert_fpa_to_sca(self, xfpa, yfpa, sca):
        """
        Returns SCA position [pixel] for reference position in the FPA [degree]
        Input: xfpa, yfpa are source position, in deg, in focal plane (FPA)
        Output: xsca, ysca are source position, in px, in detector plane (SCA)
        """
        # Get absolute X_FPA and Y_FPA, in degrees
        xcen, ycen = self.config["detector_model"]["xy_centers"][sca]
        dx = (xfpa * 3600 / self.pixel_scale) - (xcen * self.pixel_per_mm)
        dy = (yfpa * 3600 / self.pixel_scale) - (ycen * self.pixel_per_mm)

        xsca = dx + self.config["detector_model"]["crpix1"]
        ysca = dy + self.config["detector_model"]["crpix2"]

        return xsca, ysca

    def get_map_coords(self, xfpa, yfpa, order="+1"):
        """
        Method to return the trace offset location for a given reference pixel
        The trace offset location returned defines the "ref_lambda" wavelength
        Input: reference position in deg from get_FPA_coords
        Output: trace offset position in mm
        """
        x = np.atleast_1d(xfpa)[:, np.newaxis] ** np.arange(self.dimen_map)
        y = np.atleast_1d(yfpa)[:, np.newaxis] ** np.arange(self.dimen_map)

        xmpa = np.diagonal(x @ self.beam_coeffs[order]["X_ij"] @ y.T)
        ympa = np.diagonal(x @ self.beam_coeffs[order]["Y_ij"] @ y.T)

        return xmpa, ympa

    def get_trace_coeffs(self, xmpa, ympa, order="+1"):
        """
        Method to return the curvature and inverse dispersion soln coefficients
        for a given trace offset location
        Input: trace offset location from get_map_coords() and trace length
        Output: curvature and inverse dispersion soln coefficients
        """
        xtrc = xmpa * self.pixel_per_mm
        ytrc = ympa * self.pixel_per_mm

        x = np.atleast_1d(xtrc)[:, np.newaxis] ** np.arange(self.dimen_crv)
        y = np.atleast_1d(ytrc)[:, np.newaxis] ** np.arange(self.dimen_crv)
        crv = (x @ self.beam_coeffs[order]["C_ijk"] @ y.T)[:, :, 0]

        x = np.atleast_1d(xtrc)[:, np.newaxis] ** np.arange(self.dimen_ids)
        y = np.atleast_1d(ytrc)[:, np.newaxis] ** np.arange(self.dimen_ids)
        ids = (x @ self.beam_coeffs[order]["D_ijk"] @ y.T)[:, :, 0]

        return crv, ids

    def integerize_width(self, width):
        """
        Utility to convert the width to a valid integer (rounds up)
        """
        return int(np.ceil(width))

    def _get_beam_coeff(self, xref, yref, sca, width=1, order="+1"):
        """
        Computes the reference pixel location of the trace for the source
        position as well as the wavelength soln and curvature coefficients
        """
        # Setup the reference position according to the specified width
        width = self.integerize_width(width=width)
        xref_f = xref - np.floor(width / 2) + np.arange(width)
        yref_f = np.full(width, yref)

        # Initial transform to FPA coordinates for the transformation soln
        xfpa, yfpa = self.convert_sca_to_fpa(xsca=xref_f, ysca=yref_f, sca=sca)

        # Apply transformation soln to get the reference position of the trace
        xmpa, ympa = self.get_map_coords(xfpa=xfpa, yfpa=yfpa, order=order)

        # Apply the wavelength and curvature soln to get the trace
        crv, ids = self.get_trace_coeffs(xmpa=xmpa, ympa=ympa, order=order)

        # Transform the reference position of the trace back to pixels
        xpix, ypix = self.convert_mpa_to_sca(xmpa=xmpa, ympa=ympa, sca=sca)

        return {
            # Input values
            "ref_pix_direct": [xref, yref],
            "width": width,
            "sca": sca,
            "order": order,
            # Reference positions of the trace
            "ref_pix_trace": np.array([xpix, ypix]).T,
            # Coefficients for the curvature and wavelength soln
            "crv": crv,
            "ids": ids,
        }

    def _get_beam_trace(self, xref, yref, sca, width=1, order="+1"):
        """
        Main workhorse function to produce the full trace (x, y, wavelength)
        using the coefficients from _get_beam_coeff()
        """
        # Ensure width is an integer
        width = self.integerize_width(width=width)

        # Get the required coefficients
        coeff = self._get_beam_coeff(
            xref=xref,
            yref=yref,
            sca=sca,
            width=width,
            order=order,
        )

        # Setup the wavelength array
        wave = self.def_lambdas - self.ref_lambda

        # Get the position along the trace
        w = wave[:, np.newaxis] ** np.arange(self.power_ids)
        yarr = w @ coeff["ids"]

        # Get the curvature along the trace
        y = yarr[:, :, np.newaxis] ** np.arange(self.power_crv)
        xarr = np.diagonal(y @ coeff["crv"], axis1=1, axis2=2)

        # Populate the relevant trace info
        coeff["trace_pix_x"] = (
            xarr.T + coeff["ref_pix_trace"][:, 0][:, np.newaxis]
        )
        coeff["trace_pix_y"] = (
            yarr.T + coeff["ref_pix_trace"][:, 1][:, np.newaxis]
        )
        coeff["trace_wvl"] = np.full((width, len(wave)), self.def_lambdas)

        return coeff

    def get_beam_trace_wvl(
        self, xpos, ypos, xref, yref, sca, width=25, order="+1"
    ):
        """
        Inverts beam trace solution returning wavelength at a given pixel
        for a reference position and source width
        """
        coeff = self._get_beam_trace(
            xref=xref, yref=yref, sca=sca, width=width, order=order
        )

        interp = LinearNDInterpolator(
            (coeff["trace_pix_x"].ravel(), coeff["trace_pix_y"].ravel()),
            coeff["trace_wvl"].ravel(),
            fill_value=np.nan,
        )

        return interp(xpos, ypos)

    def plot_quick_look(self, N=8, width=3, order="+1", cmap=plt.cm.Spectral_r):
        """
        Quick-view plot to visualize the optical model in the MPA coordinates
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=75, tight_layout=True)

        ax.set_xlim(-140, 140)
        ax.set_ylim(-120, 75)

        for sca in self.sca_list:
            ax.add_patch(self.sca_polygons[sca])
            ax.text(
                *self.config["detector_model"]["xy_centers"][sca],
                f"SCA{sca:d}",
                fontsize=18,
                fontweight=600,
                va="center",
                ha="center",
                color="dimgray",
            )

        for sca in self.sca_list:
            xref, yref = np.random.rand(N * 2).reshape(2, N)
            xref *= self.config["detector_model"]["x_size"]
            yref *= self.config["detector_model"]["y_size"]

            coeffs = self.get_beam_trace(
                xref=xref, yref=yref, sca=sca, width=width, order=order
            )

            for coeff in coeffs:
                ax.scatter(
                    *self.convert_sca_to_mpa(
                        xsca=coeff["ref_pix_direct"][0],
                        ysca=coeff["ref_pix_direct"][1],
                        sca=sca,
                    ),
                    marker="*",
                    facecolor="none",
                    edgecolor="k",
                    s=200,
                    zorder=10,
                )
                ax.scatter(
                    *self.convert_sca_to_mpa(
                        xsca=coeff["ref_pix_trace"][:, 0],
                        ysca=coeff["ref_pix_trace"][:, 1],
                        sca=sca,
                    ),
                    marker="*",
                    color="k",
                    s=100,
                    zorder=10,
                )
                ax.scatter(
                    *self.convert_sca_to_mpa(
                        xsca=coeff["trace_pix_x"].ravel(),
                        ysca=coeff["trace_pix_y"].ravel(),
                        sca=sca,
                    ),
                    c=coeff["trace_wvl"].ravel(),
                    cmap=cmap,
                    vmin=self.wvl_min,
                    vmax=self.wvl_max,
                    s=3,
                    lw=0,
                    alpha=0.8,
                    zorder=9,
                )
