# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:54:30 2020

@author: Quinten Giglio
"""
import os
import random
import fabio
import pyFAI
import numpy as np
from scipy import signal
from lmfit import models
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from background_larch import XrayBackground


class element:
    """Class to hold all of the data."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.name = os.path.splitext(os.path.basename(file_name))[0]
        # self.img = fabio.open(file_name)
        self.data = fabio.open(file_name).data

    def filt(self, zmax):
        """
        Harsh cut off of values above a zmax.

        keeps original data intact and adds data_filt attribute.

        Parameters
        ----------
        zmax : int
            Maximum acceptable value for data,
            anything above this is set to zero.

        Returns
        -------
        None.

        """
        self.data_filt = self.data
        self.data_filt[self.data > zmax] = 0

    def find_background(self, width):
        """
        Use XrayBackground() method to find the background of histogram data.

        Parameters
        ----------
        width : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        data = [int(self.hist['y'][i]) for i in range(len(self.hist['y']))]
        self.bgr = XrayBackground(data=data, width=width).bgr

    def gen_figs(self, tth0, zmax=400):
        """
        Create figures to be displayed on dashboard.

        Parameters
        ----------
        tth0 : list
            predicted two theta values.
        zmax : int, optional
            maximum for colorscale. The default is 400.

        Returns
        -------
        None.

        """
        img_fig = px.imshow(self.data, origin='lower', range_color=[0, zmax])
        img_fig.layout.coloraxis.showscale = False
        img_fig.update_layout(title=self.name)
        self.img_fig = img_fig

        hist_fig = go.Figure(
            go.Scatter(x=self.hist['x'],
                       y=self.hist['y'],
                       mode='markers',
                       name='Data'))

        for i in tth0:
            hist_fig.add_shape(
                # Line Vertical
                dict(type="line",
                     x0=i,
                     y0=0,
                     x1=i,
                     y1=self.hist['y'].max(),
                     line=dict(
                         color="MediumPurple",
                         width=2,
                         dash="dot",
                     )))
        for i, model in enumerate(self.ycomps):
            center = np.round(self.fit_values['center (degrees)'][i], decimals=2)
            hist_fig.add_trace(
                go.Scatter(x=self.hist['x'],
                           y=model,
                           mode="lines",
                           name=f'Peak: {center}'))

        hist_fig.update_layout(title=self.name,
                               xaxis_title="Two Theta (degrees)",
                               yaxis_title="Counts",
                               template='plotly_white')

        hist_fig.update_yaxes(range=[0, self.hist['y'].max() + 10])
        self.hist_fig = hist_fig

        hist_fig_2D = go.Figure(data=go.Heatmap(
            z=self.polar['I'],
            x=self.polar['tth'],
            y=self.polar['chi'],
            showscale=False,
            colorscale="Plasma",
            zmax=zmax,
            zmin=0,
            name='Data',
            hovertemplate='two theta %{x}' + '<br>chi %{y}',
        ))
        for i, strain in enumerate(self.fit_values['strain']):
            short = np.round(strain, decimals=5)
            d = np.round(self.fit_values['d (Angstroms)'][i], decimals=5)
            hist_fig_2D.add_trace(
                # Line Vertical
                go.Scatter(
                    x=[self.fit_values['center (degrees)'][i], self.fit_values['center (degrees)'][i]],
                    y=[self.polar['chi'].min(), self.polar['chi'].max()],
                    mode='lines',
                    name='Peak',
                    text=f'Fitted Values: <br>d spacing: {d}' +
                    f'<br>strain: {short}',
                    opacity=0.5,
                    marker=dict(color='white'),
                    hoverinfo='text',
                    line=dict(dash='dash')))

        hist_fig_2D.update_layout(
            title=self.name,
            xaxis_nticks=36,
            xaxis_title="Two Theta (degrees)",
            yaxis_title="Chi (degrees)",
            hovermode='x',
            showlegend=False,
            autosize=True,
            height=650,
            template='plotly_white',
        )
        self.hist_fig_2D = hist_fig_2D

        r, theta = np.meshgrid(self.polar['tth'], self.polar['chi'])
        r = r.ravel()
        theta = theta.ravel()
        polar_fig = go.Figure(
            go.Scatterpolargl(
                r=r,
                theta=theta,
                fill='toself',
                mode='markers',
                marker=dict(
                    color=self.polar['I'].ravel(),
                    size=4,
                    cmin=0,
                    cmax=zmax,
                    # colorbar=dict(title="Intensity"),
                    # colorscale="Viridis"
                )))
        polar_fig.update_layout(width=800, height=800, title=self.name)
        polar_fig.layout.coloraxis.showscale = False
        self.polar_fig = polar_fig


def d_from_hkl(hkls, a):
    """
    Calculate d spacing from miller indices.

    Parameters
    ----------
    hkls : dict
        lattice type: [list of [h,k,l]].
    a : float
        side length.

    Returns
    -------
    d : array
        array of d spacings.

    """
    d = [None] * 10
    for i, hkl in enumerate(hkls):
        d[i] = a / np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
    d = np.round(d, decimals=4)
    return d


def tth_from_d(d, lamb):
    """
    Calculate two theta values corresponding to dspacing.

    Parameters
    ----------
    d : array, list like
        d spacing values.
    lamb : float
        wavelength.

    Returns
    -------
    tth : array
        two theta values in degrees.

    """
    tth = np.empty_like(d)
    for i in range(len(d)):
        tth[i] = np.arcsin((1 * lamb) / (2 * d[i]))
    tth = np.round(2 * np.degrees(tth), decimals=2)
    return tth


def d_from_tth(tth, lamb):
    """
    Calculate d spacing from a list of two theta values.

    Parameters
    ----------
    tth : array, list like
        list of two theta values.
    lamb : float
        wavelength.

    Returns
    -------
    d : array
        d spacing values.

    """
    theta = np.deg2rad(tth / 2)
    d = np.empty_like(tth)
    for i, theta in enumerate(theta):
        d[i] = (lamb) / (2 * np.sin(theta))
    return d


def integrateHistAndPolar(data, poni):
    """
    Use pyFAI to perform azimuthal integration of 2D XRD data.

    Parameters
    ----------
    data : array.
        2D RGB image data array.
    poni : string
        location of .poni file containing callibration data from pyFAI GUI.

    Returns
    -------
    res_dict : dict
        1D results dictionary.
    res2_dict : dict
        2D results dictionary.

    """
    # create an instance of the pyFAi azimuthal Integrator
    # this object can also be manually defined
    ai = pyFAI.load(poni)
    # result in just the 2 theta space
    # array, number of bins, unit
    res = ai.integrate1d(data, 2000, unit='2th_deg')
    res_dict = {
        'x': res[0],
        'y': res[1],
    }

    # result in both angles
    # array, 2theta bins, chi bins, unit
    res2 = ai.integrate2d(data, 1000, 3600, unit='2th_deg')
    r, theta = np.meshgrid(res2[1], res2[2])
    r = r.ravel()
    theta = theta.ravel()
    I = res2[0]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = I.ravel()
    res2_dict = {
        'I': I,
        'tth': res2[1],
        'chi': res2[2],
        'x': x,
        'y': y,
        'z': z,
    }

    return res_dict, res2_dict


def create_spectrum(tth, x, y):
    """
    Create a prediction of what the spectrum should look like.

    The number of peaks/gaussian functions put into the dict
    comes directly from the length of the predicted tth input
    so if you dont have a peak where you want one to be,
    make sure there are enough tth

    Parameters
    ----------
    tth : list like
        list of expected two theta values.
    x : array
        x data from histogram.
    y : array
        y data from histogram.

    Returns
    -------
    spec : dict
        dictionary with one model dictionary for each expected peak/tth.

    """
    spec = {
        'x': x,
        'y': y,
    }
    length = range(len(tth))
    model = [dict(type='GaussianModel') for x in length]
    paramslist = [dict() for x in length]
    for i, theta in enumerate(tth):
        paramslist[i]['center'] = theta
        paramslist[i]['sigma'] = 0.1
    for i in length:
        model[i].update({'params': paramslist[i]})
    spec.update({'model': model})
    return spec


def update_spec_from_peaks(spec, peak_widths=(3, 10), **kwargs):
    """
    Update the spectrum dictionary by looking at the data.

    Possibly redundant in this case due to how the spectrum is generated.
    Uses scipy.signal.find_peaks()

    Parameters
    ----------
    spec : dict
        contains x y data and one model dictionary for each expected peak/tth.
    peak_widths : tuple, optional
        predicted widths. The default is (3, 10).
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    NotImplemented
        DESCRIPTION.

    Returns
    -------
    peak_indicies : array
        fitted local maximums.

    """
    x = spec['x']
    y = spec['y']
    model_indicies = list(range(len(spec['model'])))
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(),
                                           model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplementedError('model not implemented yet')
    return peak_indicies


def generate_model(spec):
    """
    Compile the model for lmfit.

    Parameters
    ----------
    spec : dict
        contains x y data and one model dictionary for each expected peak/tth.

    Raises
    ------
    NotImplemented
        DESCRIPTION.

    Returns
    -------
    composite_model : dict
        model to fit to the data.
    params : dict
        params to fit to the data.

    """
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in [
                'GaussianModel', 'LorentzianModel', 'VoigtModel'
        ]:  # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1 * y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix + 'center': x_min + x_range * random.random(),
                prefix + 'height': y_max * random.random(),
                prefix + 'sigma': x_range * random.random()
            }
        else:
            raise NotImplementedError(
                f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params,
                                         **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def get_output(model, params, spec):
    """
    Fit the model to the data.

    Parameters
    ----------
    model : dict
        from generate_model().
    params : dict
        from_generate_model().
    spec : dict
        spectrum including x y data.

    Returns
    -------
    ycomps : nested array
        y components for each peak curve.
    values_df : pandas dataframe
        best fit values and parameters.

    """
    x = spec['x']
    y = spec['y']
    output = model.fit(y, params, x=x)
    components = output.eval_components(x=x)
    # find y vals with components[f'm{i}_']
    ycomps = [None] * len(spec['model'])
    for i, mod in enumerate(spec['model']):
        ycomps[i] = components[f'm{i}_']
    best_values = output.best_values
    model_params = {
        'GaussianModel': ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel': ['amplitude', 'sigma', 'gamma']
    }
    values = {
        'order': [],
        'amplitude (counts)': [],
        'center (degrees)': [],
        'sigma (width in degrees)': [],
        'FWHM (degrees)': [],
        'model': [],
    }

    for i, mod in enumerate(spec['model']):
        prefix = f'm{i}_'
        index = f'm{i}'
        for param in model_params[mod["type"]]:
            if param == 'amplitude':
                values['amplitude (counts)'].append(best_values[prefix + param])
            elif param == 'sigma':
                values['sigma (width in degrees)'].append(best_values[prefix + param])
                fwhm = 2 * best_values[prefix + param] * np.sqrt(2 * np.log(2))
                values['FWHM (degrees)'].append(fwhm)
        values['center (degrees)'].append(best_values[prefix + 'center'])
        values['order'].append(index)
        values['model'].append(mod['type'])
    values_df = pd.DataFrame.from_dict(values)
    values_df.set_index('order', inplace=True)

    return ycomps, values_df


def fit_hist(tth0, x, y):
    """
    Run all of the fitting functions with one call.

    Parameters
    ----------
    tth0 : list
        predicted two theta values.
    x : TYPE
        x data for histogram.
    y : TYPE
        y data for histogram.

    Returns
    -------
    ycomps : dict
        y components for each peak.
    fit_vals_df : pandas dataframe
        best fit values and params.

    """
    spec = create_spectrum(tth0, x, y)
    peak_indices = update_spec_from_peaks(spec)
    model, params = generate_model(spec)
    ycomps, fit_vals_df = get_output(model, params, spec)
    return ycomps, fit_vals_df


def show_one(element_obj):
    """
    Generate html for dashboard overview page.

    Figure for 2D histogram, histogram with fit and table of values.

    Parameters
    ----------
    element_obj : element class
        element class with figures generated.

    Returns
    -------
    dash bootsrap components container with html figs inside
        DESCRIPTION.

    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=element_obj.hist_fig_2D, )],
                    width={
                        "size": 7,
                    }),
            dbc.Col([
                html.Div([dcc.Graph(figure=element_obj.hist_fig)]),
                html.Div(
                    [
                        dbc.Table.from_dataframe(
                            element_obj.fit_values.round(4).drop(
                                ["model", "amplitude (counts)"], axis=1),
                            striped=False,
                            bordered=False,
                            className="table table-hover",
                            size='sm',
                            responsive=True),
                    ],
                    className="table-wrapper-scroll-y my-custom-scrollbar",
                    style={
                        'marginBottom': 20,
                        'marginTop': 10,
                    },
                ),
            ],
                    width={'size': 5}),
        ]),
    ],
                         fluid="True",
                         style={
                             "margin": "20px",
                             "border": "2px solid black",
                         })
