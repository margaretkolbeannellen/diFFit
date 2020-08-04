# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:54:30 2020

@author: starlord
"""
import fabio
import pyFAI
import numpy as np
from scipy import signal
from lmfit import models
import random
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class element:
    def __init__(self, file_name):
        self.file_name = file_name
        name = os.path.basename(file_name)
        self.name = os.path.splitext(name)[0]
        self.img = fabio.open(file_name)
        self.data = fabio.open(file_name).data
        self.hkls = [
            [1, 1, 1],
            [2, 0, 0],
            [2, 2, 0],
            [3, 1, 1],
            [2, 2, 2],
            [4, 0, 0],
            [3, 3, 1],
            [4, 2, 0],
            [4, 2, 2],
            [3, 3, 3],
        ]

    def filt(self, zmax):
        self.data_filt = self.data
        self.data_filt[self.data > zmax] = 0

    def gen_figs(self, Filter=False, zmax=400):
        img_fig = px.imshow(self.data, origin='lower', range_color=[0, zmax])
        img_fig.layout.coloraxis.showscale = False
        img_fig.update_layout(title=self.name)
        self.img_fig = img_fig

        hist_fig = go.Figure(
            go.Scatter(
                x=self.res[0],
                y=self.res[1],
                mode='markers',
            ))

        for i in self.tth:
            hist_fig.add_shape(
                # Line Vertical
                dict(type="line",
                     x0=i,
                     y0=0,
                     x1=i,
                     y1=self.res[1].max(),
                     line=dict(
                         color="MediumPurple",
                         width=2,
                         dash="dot",
                     )))
        for i, model in enumerate(self.ycomps):
            hist_fig.add_trace(
                go.Scatter(x=self.spec['x'], y=model, mode="lines"))

        hist_fig.update_layout(
            title=self.name,
            xaxis_title="Two Theta",
            yaxis_title="Counts",
        )

        hist_fig.update_yaxes(range=[0, self.res[1].max()+10])
        self.hist_fig = hist_fig

        hist_fig_2D = go.Figure(data=go.Heatmap(
            z=self.res2[0],
            x=self.res2[1],
            y=self.res2[2],
            zmax=zmax,
            zmin=0,
        ))
        hist_fig_2D.update_layout(
            title=self.name,
            xaxis_nticks=36,
            xaxis_title="Two Theta",
            yaxis_title="Chi",
        )
        self.hist_fig_2D = hist_fig_2D

        r, theta = np.meshgrid(self.res2[1], self.res2[2])
        r = r.ravel()
        theta = theta.ravel()
        size = np.ones_like(r)

        polar_fig = go.Figure(
            go.Scatterpolargl(
                r=r,
                theta=theta,
                mode='markers',
                marker=dict(
                    color=self.res2[0].ravel(),
                    size=4,
                    cmin=0,
                    cmax=zmax,
                    #colorbar=dict(title="Intensity"),
                    # colorscale="Viridis"
                )))
        polar_fig.update_layout(width=800, height=800, title=self.name)
        polar_fig.layout.coloraxis.showscale = False
        self.polar_fig = polar_fig
        
        
        
def process(data, poni):
    """
    Find angles from 2D x-ray diffraction data.

    Parameters
    ----------
    tiff : 2D array
        2 dimensional array of diffraction data
    poni : .poni file
        The name of file path of the callibration file.
        This can be generated using the pyFAI GUI.
        To launch run pyFAI-calib2 in the terminal.

    Returns
    -------
    res : list-like
        Two theta diffraction data. [two-theta, intensity]
    res2 : list-like
        2D diffraction data. [I, two-theta, chi]

    """
    # create an instance of the pyFAi azimuthal Integrator
    # this object can also be manually defined
    ai = pyFAI.load(poni)
    # result in just the 2 theta space
    # array, number of bins, unit
    res = ai.integrate1d(data, 1000, unit='2th_deg')
    # result in both angles
    # array, 2theta bins, chi bins, unit
    res2 = ai.integrate2d(data, 500, 360, unit='2th_deg')

    return res, res2



def d_from_hkl(hkls, a):
    d = [None]*10
    for i, hkl in enumerate(hkls):
        d[i] = a/np.sqrt(hkl[0]**2+hkl[1]**2+hkl[2]**2)
    d = np.round(d, decimals=4)
    return d


def tth_from_d(d,lamb):
    tth = np.empty_like(d)
    for i in range(len(d)):        
        tth[i] = np.arcsin((1*lamb)/(2*d[i]))
    tth = np.round(2*np.degrees(tth),decimals=2)    
    return tth


def create_spectrum(tth, x, y):
    spec = {
        'x': x,
        'y': y,
    }
    length = range(len(tth))
    model = [dict(type='GaussianModel') for x in length]
    params_keys = ['center', 'sigma']
    paramslist = [dict() for x in length]
    for i, tth in enumerate(tth):
        paramslist[i]['center'] = tth
        paramslist[i]['sigma'] = 0.1
    for i in length:
        model[i].update({'params': paramslist[i]})
    spec.update({'model': model})
    return spec

def update_spec_from_peaks(spec, peak_widths=(3, 10), **kwargs):
    x = spec['x']
    y = spec['y']
    model_indicies = list(range(len(spec['model'])))
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
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
            raise NotImplemented('model not implemented yet')
    return peak_indicies


def generate_model(spec):
    '''
    

    Parameters
    ----------
    spec : TYPE
        DESCRIPTION.

    Raises
    ------
    NotImplemented
        DESCRIPTION.

    Returns
    -------
    composite_model : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    '''
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
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def get_output(model, params, x, y, spec):
    output = model.fit(y, params, x=x)
    components = output.eval_components(x=x)
    # find y vals with components[f'm{i}_']
    ycomps = [None]*len(spec['model'])
    for i, model in enumerate(spec['model']):
        ycomps[i] = components[f'm{i}_']
    best_values = output.best_values
    return ycomps, best_values


def get_vals_df(spec, best_values):
    model_params = {
        'GaussianModel': ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel': ['amplitude', 'sigma', 'gamma']
    }
    values = {
        'order': [],
        'amplitude': [],
        'center': [],
        'sigma': [],
        'model': []
    }

    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        index = f'm{i}'
        for param in model_params[model["type"]]:
            if param == 'amplitude':
                values['amplitude'].append(best_values[prefix + param])
            elif param == 'sigma':
                values['sigma'].append(best_values[prefix + param])
        values['center'].append(best_values[prefix + 'center'])
        values['order'].append(index)
        values['model'].append(model['type'])
    df = pd.DataFrame.from_dict(values)
    df.set_index('order', inplace=True)
    return df        
        