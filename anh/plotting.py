import subprocess
from cycler import cycler
from copy import copy,deepcopy
import tempfile
import shutil

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import gca,gcf
from scipy import constants

from . import tools
from .tools import *

golden_ratio = 1.61803398874989

## standard papersize for figures - in inches
papersize={
    'a4':(8.3,11.7),
    'a4_portrait':(8.3,11.7),
    'a4_landscape':(11.7,8.3),
    'a5':(5.87,8.3),
    'a5landscape':(8.3,5.87),
    'letter':(8.5,11),
    'letter_portrait':(8.5,11),
    'letter_landscape':(11,8.5),
    'latexTiny':(golden_ratio*1.2,1.2),
    'latexSmall':(golden_ratio*1.7,1.7),
    'latexMedium':(golden_ratio*3.,3.),
    'latexLarge':(golden_ratio*4.,4.),
    'squareMedium':(5.,5.),
    'squareLarge':(8.,8.),
    'article_full_page_width':6.77,
    'article_single_column_width':3.5,
    'article_full_page_height':8.66,
}

def presetRcParams(
        preset='base',      # name of the preset to use
        # make_fig=False, # make a figure and axes and return (fig,ax)
        **params, # a dictionay containing any valid rcparams, or abbreviated by removing xxx.yyy. etc
):
    """Call this function wit some presets before figure object is
    created. If make_fig = True return (fig,ax) figure and axis
    objects. Additional kwargs are applied directly to rcParams"""
    ## try and get screen size
    try:
        xscreen,yscreen = get_screensize() # pts
        ## subtrace some screenspace for toolbar etc
        yscreen -= 50
        xscreen -= 15
        xscreen,yscreen = xscreen/matplotlib.rcParams['figure.dpi'],yscreen/matplotlib.rcParams['figure.dpi'] # inches
    except:
        warnings.warn("could not get screensize")
        xscreen,yscreen = 5,5
    ## dicitionary of dictionaries containing keyval pairs to
    ## substitute into rcParams
    presets = dict()
    ## the base params
    presets['base'] = {
        'legend.handlelength'  :1.5,
        'legend.handletextpad' :0.4,
        'legend.labelspacing'  :0.,
        # 'legend.loc'           :'best', # setting this to best makes things slooooow
        'legend.numpoints'     :1,
        'font.family'          :'serif',
        'text.usetex'          :False,
        'text.latex.preamble'  : r'''\usepackage{mhchem}\usepackage[np]{numprint}\npdecimalsign{\ensuremath{.}} \npthousandsep{\,} \npproductsign{\times} \npfourdigitnosep''',
        'mathtext.fontset'     :'cm',
        'lines.markeredgewidth': 1,
        # 'axes.prop_cycle': cycler('color',linecolors_colorblind_safe),
        'axes.prop_cycle': cycler('color',linecolors_print),
        # 'axes.color_cycle': linecolors_colorblind_safe,
        'patch.edgecolor': 'none',
        'xtick.minor.top': True,
        'xtick.minor.bottom': True,
        'xtick.minor.visible': True ,
        'xtick.top': True ,
        'xtick.bottom': True ,
        'ytick.minor.right': True,
        'ytick.minor.left': True,
        'ytick.minor.visible': True ,
        'ytick.right': True ,
        'ytick.left': True ,
        'path.simplify'      :False, # whether or not to speed up plots by joining line segments
        'path.simplify_threshold' :1, # how much to do so
        'agg.path.chunksize': 10000,  # antialisin speed up -- does not seem to do anything over path.simplify
        ## set up axes tick marks and labels
        'axes.formatter.limits' : (-3, 6), # use scientific notation if log10 of the axis range is smaller than the first or larger than the second
        'axes.formatter.use_mathtext' : True, # When True, use mathtext for scientific notation.
        'axes.formatter.useoffset'      : False,    # If True, the tick label formatter # will default to labeling ticks relative # to an offset when the data range is # small compared to the minimum absolute # value of the data.
        'axes.formatter.offset_threshold' : 4,     # When useoffset is True, the offset # will be used when it can remove # at least this number of significant # digits from tick labels.
        }
    presets['screen']=deepcopy(presets['base'])
    presets['screen'].update({
        'figure.figsize'     :(xscreen,yscreen),
        'figure.subplot.left':0.05,
        'figure.subplot.right':0.95,
        'figure.subplot.bottom':0.05,
        'figure.subplot.top':0.95,
        'figure.subplot.wspace':0.2,
        'figure.subplot.hspace':0.2,
        'figure.autolayout'  : True, # reset tight_layout everytime figure is redrawn -- seems to cause problems with long title and label strings
        # 'toolbar'  :'none' , # hides toolbar but also disables keyboard shortcuts
        'legend.handlelength':4,
        'text.usetex'        :False,
        'lines.linewidth'    : 2,
        'lines.markersize'   : 10.0,
        'grid.alpha'         : 1.0,
        'grid.color'         : 'gray',
        'grid.linestyle'     : ':',
        'legend.fontsize'    :12,
        'axes.titlesize'     :12,
        'axes.labelsize'     :12,
        'xtick.labelsize'    :12,
        'ytick.labelsize'    :12,
        'font.size'          :12,
        'axes.prop_cycle'    : cycler('color',linecolors_screen),
        'path.simplify'      :True , # whether or not to speed up plots by joining line segments
        'path.simplify_threshold' :1, # how much to do so
        'agg.path.chunksize': 10000,  # antialisin speed up -- does not seem to do anything over path.simplify
    })
    ## screen without axes or text
    presets['fast']=deepcopy(presets['screen'])
    presets['fast'].update({
        'figure.autolayout'  :False,
        'lines.antialiased' : False,
        'patch.antialiased' : False,
        'figure.subplot.left':0,
        'figure.subplot.right':1,
        'figure.subplot.bottom':0,
        'figure.subplot.top':1,
        'figure.subplot.wspace':0,
        'figure.subplot.hspace':0,
        'figure.frameon':False,
        'grid.linestyle'     : '',
        'legend.fontsize'    :0,
        'axes.titlesize'     :0,
        'axes.labelsize'     :0,
        'xtick.labelsize'    :0,
        'ytick.labelsize'    :0,
        'font.size'          :0,
        'axes.prop_cycle'    : cycler('color',linecolors_screen),
        'path.simplify'      :True , # whether or not to speed up plots by joining line segments
        'path.simplify_threshold' :1, # how much to do so
        'agg.path.chunksize': 10000,  # antialisin speed up -- does not seem to do anything over path.simplify
    })
    ## single column figure in a journal artile
    presets['article_single_column']=deepcopy(presets['base'])
    presets['article_single_column'].update({
        'text.usetex'          :False,
        'figure.figsize'       :(papersize['article_single_column_width'],papersize['article_single_column_width']/constants.golden_ratio),
        # 'lines.linewidth'    : 0.5,
        'lines.linewidth'    : 1,
        'figure.subplot.left'  :0.14,
        'figure.subplot.right' :0.97,
        'figure.subplot.bottom':0.20,
        'figure.subplot.top'   :0.95,
        'figure.subplot.wspace':0.35,
        'figure.subplot.hspace':0.3,
        'legend.fontsize'      :9.,
        'axes.titlesize'       :10.,
        'axes.labelsize'       :10.,
        'lines.markersize'     :4.,
        'xtick.labelsize'      :9.,
        'ytick.labelsize'      :9.,
        'font.size'            :10.,
        # 'axes.formatter.use_mathtext': True, # use math text for scientific notation . i.e,. not 1e-9
    })
    presets['article_single_column_one_third_page']=deepcopy(presets['article_single_column'])
    presets['article_single_column_one_third_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']/3.),
            'figure.subplot.bottom':0.15,
            'figure.subplot.top'   :0.95,
            })
    presets['article_single_column_half_page']=deepcopy(presets['article_single_column'])
    presets['article_single_column_half_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']/2.),
            'figure.subplot.bottom':0.1,
            'figure.subplot.top'   :0.95,
            })
    presets['article_single_column_two_thirds_page']=deepcopy(presets['article_single_column_half_page'])
    presets['article_single_column_two_thirds_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']*2./3.),
            'figure.subplot.bottom':0.1,
            'figure.subplot.top'   :0.95,
            })
    presets['article_single_column_full_page']=deepcopy(presets['article_single_column'])
    presets['article_single_column_full_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']),
            'figure.subplot.bottom':0.05,
            'figure.subplot.top'   :0.97,
            })
    presets['article_double_column']=deepcopy(presets['article_single_column'])
    presets['article_double_column'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_single_column_width']),
            'figure.subplot.left':0.1,
            'lines.linewidth'    : 0.5,
            'figure.subplot.bottom':0.14,})
    presets['article_double_column_half_page']=deepcopy(presets['article_double_column'])
    presets['article_double_column_half_page'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_full_page_height']/2.),
            # 'figure.figsize':(6.,5.),
            'figure.subplot.left':0.1,
            'figure.subplot.right':0.95,
            'figure.subplot.bottom':0.10,
            'figure.subplot.top':0.95,
            'figure.subplot.wspace':0.3,
            'figure.subplot.hspace':0.3,})
    presets['article_double_column_two_thirds_page']=deepcopy(presets['article_double_column'])
    presets['article_double_column_two_thirds_page'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_full_page_height']*2./3.),
            'figure.subplot.left':0.1,
            'figure.subplot.right':0.95,
            'figure.subplot.bottom':0.07,
            'figure.subplot.top':0.95,
            'figure.subplot.wspace':0.3,
            'figure.subplot.hspace':0.3,
            })
    presets['article_full_page']=deepcopy(presets['article_double_column'])
    presets['article_full_page'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_full_page_height']),
            'figure.figsize':(6.9,9.2),
            'figure.subplot.bottom':0.05,
            'figure.subplot.top'   :0.97,
            # 'figure.subplot.hspace':0.15,
            'figure.subplot.hspace':0.27,
            'figure.subplot.left':0.1,
            })
    presets['article_double_column_full_page'] = presets['article_full_page']
    presets['article_full_page_landscape']=deepcopy(presets['article_full_page'])
    presets['article_full_page_landscape'].update({
            'figure.figsize':(papersize['article_full_page_height'],papersize['article_full_page_width']),
            'figure.subplot.left':0.05,
            'figure.subplot.right'   :0.97,
            'figure.subplot.bottom':0.07,
            })
    presets['beamer_base']=deepcopy(presets['base'])
    presets['beamer_base'].update({
        'text.usetex'          :False,
        'font.size'            :8,
        'xtick.labelsize'      :8,
        'ytick.labelsize'      :8,
        'ytick.labelsize'      :8,
        'lines.linewidth'      :0.5,
        'lines.markersize'     :4,
    })
    ## good for a simgle image slide
    presets['beamer_large']=deepcopy(presets['base'])
    presets['beamer_large'].update({
        'figure.figsize'       :(4.5,2.5), # 5.0393701,3.7795276 beamer size
        'figure.subplot.left':0.15,
        'figure.subplot.right':0.92,
        'figure.subplot.bottom':0.17,
        'figure.subplot.top'   :0.93,
        'figure.subplot.wspace':0.20,
        'figure.subplot.hspace':0.37,
        'xtick.labelsize'      :8.,
        'ytick.labelsize'      :8.,
        'ytick.labelsize'      :8.,
    })
    presets['beamer'] = presets['beamer_large']
    presets['beamer_large_twinx']=deepcopy(presets['beamer_large'])
    presets['beamer_large_twinx'].update({
        'figure.figsize'       :(4.5,2.5), # 5.0393701,3.7795276 beamer size
        'figure.subplot.left':0.15,
        'figure.subplot.right':0.85,
    })
    ## good for single imag ewith more text
    presets['beamer_medium']=deepcopy(presets['beamer_base'])
    presets['beamer_medium'].update({
            'figure.figsize'       :(constants.golden*1.8,1.8),
            'figure.subplot.left'  :0.18,
            'figure.subplot.right' :0.95,
            'figure.subplot.bottom':0.2,
            'figure.subplot.top'   :0.9,
            'figure.subplot.wspace':0.20,
            'figure.subplot.hspace':0.37,
            })
    ## good to fill one quadrant
    presets['beamer_small']=deepcopy(presets['beamer_base'])
    presets['beamer_small'].update({
        'figure.figsize'       :(2.25,2.25/constants.golden),
        'figure.subplot.left'  :0.25,
        'figure.subplot.right' :0.95,
        'figure.subplot.bottom':0.25,
        'figure.subplot.top'   :0.95,
        'figure.subplot.wspace':0.20,
        'figure.subplot.hspace':0.37,
        'axes.labelpad': 1,
    })
    ## maximise
    presets['beamer_entire_slide']=deepcopy(presets['beamer_large'])
    presets['beamer_entire_slide'].update({
        'figure.figsize'       :(5.0393701,3.7795276), 
    })
    ## fit more text beside
    presets['beamer_wide']=deepcopy(presets['beamer_large'])
    presets['beamer_wide'].update({
            'figure.figsize'       :(4.5,1.5),
            'figure.subplot.bottom':0.25,
            })
    ## fit more text beside
    presets['beamer_tall']=deepcopy(presets['beamer_large'])
    presets['beamer_tall'].update({
            'figure.figsize'       :(2.2,3.2),
            'figure.subplot.left':0.25,
            'figure.subplot.bottom':0.14,
            'figure.subplot.top':0.95,
            })
    ## good for two column
    presets['beamer_half_width']=deepcopy(presets['beamer_base'])
    presets['beamer_half_width'].update({
            'figure.figsize'       :(2.25,2.5),
            'figure.subplot.left':0.2,
            'figure.subplot.bottom':0.15,
            'figure.subplot.top':0.95,
            'figure.subplot.right':0.95,
            })
    presets['a4_portrait'] = deepcopy(presets['base'])
    presets['a4_portrait'].update({
            'text.usetex'          :False,
            'figure.figsize':papersize['a4_portrait'],
            'figure.subplot.left':0.11,
            'figure.subplot.right':0.92,
            'figure.subplot.top':0.94,
            'figure.subplot.bottom':0.08,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            'lines.markersize':2.,
            'legend.fontsize':'large',
            'font.size':10,
            })
    presets['a4_landscape'] = deepcopy(presets['a4_portrait'])
    presets['a4_landscape'].update({
            'figure.figsize':papersize['a4_landscape'],
            'figure.subplot.left':0.07,
            'figure.subplot.right':0.95,
            'figure.subplot.top':0.95,
            'figure.subplot.bottom':0.08,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            })
    presets['letter_portrait'] = deepcopy(presets['a4_portrait'])
    presets['letter_portrait'].update({
            'figure.figsize':papersize['letter_portrait'],
            'figure.subplot.left':0.11,
            'figure.subplot.right':0.92,
            'figure.subplot.top':0.94,
            'figure.subplot.bottom':0.08,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            })
    presets['letter_landscape'] = deepcopy(presets['a4_portrait'])
    presets['letter_landscape'].update({
            'figure.figsize':papersize['letter_landscape'],
            'figure.subplot.left':0.08,
            'figure.subplot.right':0.93,
            'figure.subplot.top':0.92,
            'figure.subplot.bottom':0.1,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            })
    ## synonyms 
    presets['a4'] = presets['a4_portrait']
    presets['letter'] = presets['letter_portrait']
    presets['a4landscape'] = presets['a4_landscape'] # deprecated
    presets['a4portrait'] = presets['a4_portrait']   # deprecated
    ## find key in presets to match the requested preset
    for key in presets[preset]:
        matplotlib.rcParams[key] = presets[preset][key]
    ## extra keys -- potentially with assuming prefixes in rcParams
    ## hierarchy until an existing key is found
    set_rcparam(params)
    # ## create figure and axes objects
    # if make_fig:
        # fig = plt.figure()
        # ax = fig.gca()
        # return fig,ax 

def set_rcparam(params):
    """Set a matplotlib rcParam, possibly guessing the prefix of an
    abbreviated form.  E.g., linewith=1 is guess as
    lines.linewidth=1."""
    for key,val in params.items():
        if key in matplotlib.rcParams:
            ## not abbreviated
            matplotlib.rcParams[key] = val
        else:
            for prefix in ('','figure.','figure.subplot.','axes.',
                            'lines.','font.','xtick.','ytick.',): 
                if prefix+key in matplotlib.rcParams:
                    matplotlib.rcParams[prefix+key] = val
                    break
            else:
                raise Exception(f"Could not interpret abbreviated rcParam: {repr(key)}")

def qfig(
        n=None,
        preset='screen',
        figsize=None,
        hide_toolbar=True,
        fullscreen=False,
        show=False,
        **preset_kwargs):
    """quick figure preparation."""
    figure_exists = plt.fignum_exists(n)
    presetRcParams(preset,**preset_kwargs)
    ## get / make figure
    if n is None:
        fig = plt.figure()
    else:
        fig = plt.figure(n);
    if fullscreen and not figure_exists:
        set_figsize_fullscreen()
    ## use this as a marker that this figure was created by this function 
    fig._my_fig = True        
    newcolor(reset=True)
    newlinestyle(reset=True)
    newmarker(reset=True)
    extra_interaction()
    if figsize=='full screen':
        set_figsize_fullscreen(fig=fig)
    elif figsize=='quarter screen':
        set_figsize_fullscreen(fig=fig,scale=0.25)
    elif figsize is not None:
        set_figsize_in_pixels(*figsize,fig=fig)
    # if hide_toolbar:
        # # from PyQt5 import QtWidgets 
        # # from PyQt4 import QtGui as QtWidgets 
        # try:
            # win = fig.canvas.manager.window
        # except AttributeError:
            # win = fig.canvas.window()
        # # toolbar = win.findChild(QtWidgets.QToolBar)
        # # toolbar.setVisible(False)
    def format_coord(x,y):
        if x<1e-5 or x>1e5:
            xstr = f'{x:0.18e}'
        else:
            xstr = f'{x:0.18f}'
        if y<1e-5 or y>1e5:
            ystr = f'{y:0.18e}'
        else:
            ystr = f'{y:0.18f}'
        return f'x={xstr:<25s} y={ystr:<25s}'
    ## show or update the figure
    if show:
        if figure_exists:
            qupdate(fig)
        else:
            plt.show(block=False)
            plt.pause(1e-10)
    fig.clf()
    return fig

def qax(*qfig_args,**qfig_kwargs):
    return qfig(*qfig_args,**qfig_kwargs).gca()

def qfigax(fig=None,ax=None,**qfig_kwargs):
    """Get a figure and ax object."""
    if ax is None:
        if fig is None:
            fig = gcf()
        elif isinstance(fig,int):
            fig = qfig(n=fig,**qfig_kwargs)
        elif isinstance(fig,matplotlib.figure.Figure):
            pass
        else:
            raise Exception('Invalid input: {fig=}. Must be None, Figure, or an int')
        ax = fig.gca()
    else:
        fig = ax.get_figure()
    return fig,ax

def qupdate(fig=None):
    """Exisint figures will be updated figure without blocking or
raising. New figures non-blocking show and be raised."""
    if fig is None:
        ## update all figures
        for fig in plt.get_fignums():
            qupdate(fig)
    elif isinstance(fig,int):
        if not plt.fignum_exists(fig):
            ## new figure
            qfig(fig,show=True)
        else:
            ## get figure from number and update
            qupdate(plt.figure(fig))
    else:
        ## update existing figure
        fig.canvas.draw()
        fig.canvas.start_event_loop(1e-10)

def figax(*args,**kwargs):
    f = fig(*args,**kwargs)
    ax = f.gca()
    return(f,ax)

def get_screensize():
    """In pixels."""
    status,output = subprocess.getstatusoutput(r"set_window.py get_current_screen_dimensions")
    if status!=0:
        raise Exception("Could not determine screensize")
    x,y = output.split()
    return(int(x),int(y))

def update_figure_without_raising(fig=None):
    """A hack to redraw a figure without raising the window. From this stack overflow: https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7"""
    if fig is None:
        fig = plt.gcf()
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)

def set_figsize_fullscreen(fig=None,scale=1):
    """Set figsize in pixels for screen display aiming for full
    screen."""
    try:
        x,y = get_screensize()
        x,y = x*scale,y*scale
        return(set_figsize_in_pixels(
            x=x,
            # y=y-55,        # a bit less to fit in toolbar
            y=y-400,        # a bit less to fit in toolbar
            fig=fig))
    except:
        pass
        # warnings.warn('set_figsize_fullscreen failed')
        

def set_figsize_in_pixels(x,y,fig=None):
    """Set figsize in pixels for screen display."""
    if fig is None:
        fig = plt.gcf()
    dpi = fig.get_dpi()
    figsize=(x/dpi,y/dpi)
    fig.set_size_inches(*figsize)
    return(figsize)


def get_data_range(axes=None):
    """Return (xbeg,xend,ybeg,yend) encompassing all data plotted in
    axes."""
    if axes is None:
        axes = plt.gca()
    lines  = [line for line in axes.get_lines()
              if line.get_visible() and isinstance(line.get_xdata(),np.ndarray)]
    ybeg = np.nanmin([line.get_ydata().min() for line in lines])
    yend = np.nanmax([line.get_ydata().max() for line in lines])
    xbeg = np.nanmin([line.get_xdata().min() for line in lines])
    xend = np.nanmax([line.get_xdata().max() for line in lines])
    return xbeg,xend,ybeg,yend

def extra_interaction(fig=None,pickradius=5):
    """Call this to customise the matplotlib interactive gui experience
    for this figure."""
    ## select figure
    if fig is None:
        fig = plt.gcf() # determine figure object
    if hasattr(fig,'_my_extra_interaction'):
        ## extra interaction already set up, do not add again
        return                  
    else:
        fig._my_extra_interaction = {
            'pickradius':pickradius,
        }
        # _extra_interaction_select_axes(fig.gca())
    ## watch for events
    fig.canvas.mpl_connect('key_press_event', _extra_interaction_on_key)
    fig.canvas.mpl_connect('pick_event', _extra_interaction_on_pick)
    fig.canvas.mpl_connect('button_press_event', _extra_interaction_on_button_press)
    fig.canvas.mpl_connect('scroll_event', _extra_interaction_on_scroll_event)

def _extra_interaction_initialise_axes_if_necessary(axes):
    """Setup this dictionary if not set."""
    if not hasattr(axes,'_my_extra_interaction'):
        axes._my_extra_interaction = dict(
            selected_line = None,
            selected_line_annotation = None,
            currently_selecting_points = False,
            make_point_annotations = False,
            list_of_point_annotations = [],) 

def _extra_interaction_select_axes(axes):
    """"""
    fig = axes.figure          
    fig.sca(axes) # set clicked axes to gca
    ## initialise dictionrary to store useful data
    _extra_interaction_initialise_axes_if_necessary(axes)
    ## set all lines to requested picker values
    if fig._my_extra_interaction['pickradius'] is not None:
        for line in axes.lines:
            line.set_picker(True) 
            line.set_pickradius(fig._my_extra_interaction['pickradius'])
    ## set this as selected axes
    fig._my_extra_interaction['axes'] = axes

def _extra_interaction_zoom_in(x_or_y,axes,factor=2/3):
    if x_or_y == 'x':
        xmin,xmax = axes.get_xlim()
        axes.set_xlim(
            (xmin+xmax)/2-(xmax-xmin)/2*factor,
            (xmin+xmax)/2+(xmax-xmin)/2*factor)
    elif x_or_y == 'y':
        ymin,ymax = axes.get_ylim()
        axes.set_ylim(
            (ymin+ymax)/2-(ymax-ymin)/2*factor,
            (ymin+ymax)/2+(ymax-ymin)/2*factor)
 
def _extra_interaction_zoom_out(x_or_y,axes,factor=2/3):
    if x_or_y == 'x':
        xmin,xmax = axes.get_xlim()
        axes.set_xlim(
            (xmin+xmax)/2-(xmax-xmin)/2/factor,
            (xmin+xmax)/2+(xmax-xmin)/2/factor)
    elif x_or_y == 'y':
        ymin,ymax = axes.get_ylim()
        axes.set_ylim(
            (ymin+ymax)/2-(ymax-ymin)/2/factor,
            (ymin+ymax)/2+(ymax-ymin)/2/factor)

def _extra_interaction_select_line(line):
    """some data stored in figure to facilitate actions below
    what to do when a deselected line is picked"""
    print(line.get_label())
    axes = line.axes
    _extra_interaction_initialise_axes_if_necessary(axes)
    axes._my_extra_interaction['selected_line_annotation'] = axes.annotate(
        line.get_label(),(0.1,0.1),xycoords='axes fraction',
        ha='left',va='top',fontsize='medium',in_layout=False)
    axes._my_extra_interaction['selected_line'] = line
    line.set_linewidth(line.get_linewidth()*2)
    line.set_markersize(line.get_markersize()*2)

def _extra_interaction_deselect_line(line):    
    """what to do when a selected line is picked"""
    axes = line.axes
    line.set_linewidth(line.get_linewidth()/2)
    line.set_markersize(line.get_markersize()/2)
    _extra_interaction_initialise_axes_if_necessary(axes)
    if axes._my_extra_interaction['selected_line_annotation']!=None:
        axes._my_extra_interaction['selected_line_annotation'].remove()
    axes._my_extra_interaction['selected_line_annotation'] = None
    axes._my_extra_interaction['selected_line'] = None

def _extra_interaction_on_button_press(event):
    """If turned on annotate click point coordinates."""
    if event.inaxes:
        axes = event.inaxes
        _extra_interaction_initialise_axes_if_necessary(axes)
        _extra_interaction_select_axes(axes)
        if axes._my_extra_interaction['make_point_annotations']:
            ## annotate point
            x,y = event.xdata,event.ydata
            point = axes.plot(x,y,marker='x',color='red')[0]
            annotation = plt.annotate(
                f"({x:0.12e}, {y:0.12e})",
                (x,y),
                # (1,1), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='left',
                fontsize='x-small', color='red',)
            axes._my_extra_interaction['list_of_point_annotations'].extend((point,annotation))
            plt.draw()
        elif event.button == 2:
            tools.set_clipboard(format(event.xdata),target='primary')

def _extra_interaction_on_scroll_event(event):
    """what do to when mouse wheel is rolled -- zoom in and out"""
    if event.inaxes:
        axes = event.inaxes
        if event.button == 'up':
            _extra_interaction_zoom_in('x',axes,)
            _extra_interaction_zoom_in('y',axes,)
        if event.button == 'down':
            _extra_interaction_zoom_out('x',axes,)
            _extra_interaction_zoom_out('y',axes,)
        plt.draw()

def _extra_interaction_on_pick(event):
    """on picking of line etc"""
    line = event.artist
    axes = line.axes
    _extra_interaction_initialise_axes_if_necessary(axes)
    if axes._my_extra_interaction['currently_selecting_points'] is True:
        pass
    elif axes._my_extra_interaction['selected_line'] is None:
        _extra_interaction_select_line(line)
    elif line==axes._my_extra_interaction['selected_line']:
        _extra_interaction_deselect_line(line)
    else:
        _extra_interaction_deselect_line(axes._my_extra_interaction['selected_line'])
        _extra_interaction_select_line(line)
    plt.draw()
    return

def _extra_interaction_on_key(event):
    """key options"""
    if event.key=='q':
        ## quit already handled
        return              
    axes = plt.gca()
    _extra_interaction_initialise_axes_if_necessary(axes)
    move_factor = 0.2
    zoom_factor = 1.5
    if event.key=='d':
        ## hide line
        if axes._my_extra_interaction['selected_line'] is not None:
            line = axes._my_extra_interaction['selected_line']
            _extra_interaction_deselect_line(line)
            line.set_visible(False)
    elif event.key=='D':
        ## unhide all lines
        for line in axes.lines:
            line.set_visible( True)
    elif event.key=='a':
        ## autoscale
        axes.autoscale(enable=True,axis='both',tight=True)
    elif event.key=='z': 
        ## zoom to all data
        lines  = (axes.get_lines()
                  if axes._my_extra_interaction['selected_line'] is None
                  else [axes._my_extra_interaction['selected_line']])  # use all lines if not selected
        xmin,xmax,ymin,ymax = np.inf,-np.inf,np.inf,-np.inf
        for line in lines:
            if not line.get_visible(): continue
            if not isinstance(line.get_xdata(),np.ndarray): continue # hack to avoid things I dont know what they are
            xmin = min(xmin,line.get_xdata().min())
            xmax = max(xmax,line.get_xdata().max())
            ymin = min(ymin,line.get_ydata().min())
            ymax = max(ymax,line.get_ydata().max())
        if not np.isinf(xmin):
            axes.set_xlim(xmin=xmin) 
        if not np.isinf(xmax):
            axes.set_xlim(xmax=xmax)
        if not np.isinf(ymin):
            axes.set_ylim(ymin=ymin)
        if not np.isinf(ymax):
            axes.set_ylim(ymax=ymax)
    elif event.key=='y':
        ## zoom to full yscale
        ## use all lines if none selected
        if axes._my_extra_interaction['selected_line'] is None:
            lines  = axes.get_lines()
        else:
             lines = [axes._my_extra_interaction['selected_line']]
        xmin,xmax = axes.get_xlim()
        ymin,ymax = np.inf,-np.inf
        for line in lines:
            if not line.get_visible():
                continue
            if not isinstance(line.get_xdata(),np.ndarray):
                ## hack to avoid things I dont know what they are
                continue
            i = np.argwhere((line.get_xdata()>=xmin)&(line.get_xdata()<=xmax))
            if not any(i):
                continue
            ymin = min(ymin,(line.get_ydata()[i]).min())
            ymax = max(ymax,(line.get_ydata()[i]).max())
        if not np.isinf(ymin):
            axes.set_ylim(ymin=ymin)
        if not np.isinf(ymax):
            axes.set_ylim(ymax=ymax)
    elif event.key=='x': 
        ## zoom to full xscale 
        lines  = (axes.get_lines()
                  if axes._my_extra_interaction['selected_line'] is None
                  else [axes._my_extra_interaction['selected_line']])  # use all lines if not selected
        xmin,xmax = np.inf,-np.inf
        ymin,ymax = axes.get_ylim()
        for line in lines:
            if not line.get_visible():
                continue
            if not isinstance(line.get_xdata(),np.ndarray):
                ## hack to avoid things I dont know what they are
                continue
            ## get only data in current ylim
            i = (line.get_ydata()>=ymin)&(line.get_ydata()<=ymax) 
            if not any(i):
                continue
            xmin = min(xmin,np.min(line.get_xdata()[i])) # get new limits
            xmax = max(xmax,np.max(line.get_xdata()[i]))
        if not np.isinf(xmin):
            axes.set_xlim(xmin=xmin) 
        if not np.isinf(xmax):
            axes.set_xlim(xmax=xmax)
    elif event.key=='m':
        ## set to annotate points
        if axes._my_extra_interaction['make_point_annotations']:
            axes._my_extra_interaction['make_point_annotations'] = False
        else:
            axes._my_extra_interaction['make_point_annotations'] = True
    elif event.key=='M':
        ## delete point annotations
        if axes._my_extra_interaction['list_of_point_annotations'] is not None:
            for artist in axes._my_extra_interaction['list_of_point_annotations']:
                artist.remove()
            axes._my_extra_interaction['list_of_point_annotations'].clear()
    elif event.key=='P':
        ## select (x,y) points and save to clipboard, enter to quit
        clginput()
    elif event.key=='X':
        ## select x points and save to clipboard, enter to quit
        clginput('x')
    elif event.key=='Y':
        ## select y points and save to clipboard, enter to quit
        clginput('y')
    ## move with arrow keys
    elif event.key=='right':
        xmin,xmax = axes.get_xlim()
        shift = (xmax-xmin)*move_factor
        axes.set_xlim(xmin+shift,xmax+shift)
    elif event.key=='left':
        xmin,xmax = axes.get_xlim()
        shift = (xmax-xmin)*move_factor
        axes.set_xlim(xmin-shift,xmax-shift)
    elif event.key=='up':
        ymin,ymax = axes.get_ylim()
        shift = (ymax-ymin)*move_factor
        axes.set_ylim(ymin+shift,ymax+shift)
    elif event.key=='down':
        ymin,ymax = axes.get_ylim()
        shift = (ymax-ymin)*move_factor
        axes.set_ylim(ymin-shift,ymax-shift)
    ## move with arrow keys one entire range
    elif event.key=='ctrl+right':
        xmin,xmax = axes.get_xlim()
        shift = (xmax-xmin)
        axes.set_xlim(xmin+shift,xmax+shift)
    elif event.key=='ctrl+left':
        xmin,xmax = axes.get_xlim()
        shift = (xmax-xmin)
        axes.set_xlim(xmin-shift,xmax-shift)
    elif event.key=='ctrol+up':
        ymin,ymax = axes.get_ylim()
        shift = (ymax-ymin)
        axes.set_ylim(ymin+shift,ymax+shift)
    elif event.key=='ctrl+down':
        ymin,ymax = axes.get_ylim()
        shift = (ymax-ymin)
        axes.set_ylim(ymin-shift,ymax-shift)
    ## zoom with arrow keys
    elif event.key=='shift+right':
        _extra_interaction_zoom_out('x',axes)
    elif event.key=='shift+left':
        _extra_interaction_zoom_in('x',axes)
    elif event.key=='shift+up':
        _extra_interaction_zoom_out('y',axes)
    elif event.key=='shift+down':
        _extra_interaction_zoom_in('y',axes)
    ## zoom with +/=/- keys
    elif event.key=='+' or event.key=='=':
        _extra_interaction_zoom_in('x',axes)
        _extra_interaction_zoom_in('y',axes)
    elif event.key=='-':
        _extra_interaction_zoom_out('x',axes)
        _extra_interaction_zoom_out('y',axes)
    ## toggle x or y log -- preserve limits
    elif event.key=='l':
        _extra_interaction_toggle_log('y',axes)
    elif event.key=='L':
        _extra_interaction_toggle_log('x',axes)
    ## redraw
    plt.draw()
    return
    
def _extra_interaction_toggle_log(x_or_y,axes):
    assert x_or_y in ('x','y')
    if x_or_y == 'x':
        if not hasattr(axes,'_extra_interaction_toggle_log_x'):
            axes._extra_interaction_toggle_log_x = axes.get_xscale()
        loglin = axes._extra_interaction_toggle_log_x
        lim = list(axes.get_xlim())
        axis = axes.xaxis
    else:
        if not hasattr(axes,'_extra_interaction_toggle_log_y'):
            axes._extra_interaction_toggle_log_y = axes.get_yscale()
        lim = list(axes.get_ylim())
        axis = axes.yaxis
        loglin = axes._extra_interaction_toggle_log_y
    if loglin == 'linear':
        ## make log - deal with negative limits
        lim = sorted(lim)
        if lim[1] <= 0:
            lim[1] = 1
        if lim[0] <= 0:
            lim[0] = lim[1]/1e10
        ## make linear
        if x_or_y == 'x':
            axes.set_xscale('log')
            axes._extra_interaction_toggle_log_x = 'log'
        else:
            axes.set_yscale('log')
            axes._extra_interaction_toggle_log_y = 'log'
    else:
        if x_or_y == 'x':
            axes.set_xscale('linear')
            axes._extra_interaction_toggle_log_x = 'linear'
        else:
            axes.set_yscale('linear')
            axes._extra_interaction_toggle_log_y = 'linear'
    ## rest axis limits
    if x_or_y == 'x':
        axes.set_xlim(lim)
    else:
        axes.set_ylim(lim)

_newcolor_nextcolor=0
linecolors_screen=(
    'red',
    'blue',
    'green',
    'black',
    'orange',
    'magenta',
    # 'aqua',
    'indigo',
    'brown',
    # 'grey',
    # 'aliceblue',
    # 'aquamarine',
    ## 'azure',
    ## 'beige',
    ## 'bisque',
    ## 'blanchedalmond',
    ## 'blue',
    ## 'blueviolet',
    ## 'brown',
    'burlywood',
    'cadetblue',
    'chartreuse',
    'chocolate',
    'coral',
    'cornflowerblue',
    ## 'cornsilk',
    'crimson',
    'cyan',
    # 'darkblue',
    # 'darkcyan',
    # 'darkgoldenrod',
    # 'darkgray',
    # 'darkgreen',
    # 'darkkhaki',
    # 'darkmagenta',
    # 'darkolivegreen',
    # 'darkorange',
    # 'darkorchid',
    # 'darkred',
    # 'darksalmon',
    # 'darkseagreen',
    # 'darkslateblue',
    # 'darkslategray',
    # 'darkturquoise',
    # 'darkviolet',
    # 'deeppink',
    # 'deepskyblue',
    # 'dimgray',
    # 'dodgerblue',
    # 'firebrick',
    # 'forestgreen',
    # 'fuchsia',
    # 'gainsboro',
    # 'gold',
    # 'goldenrod',
    # 'gray',
    # 'green',
    # 'greenyellow',
    # 'honeydew',
    # 'hotpink',
    # 'indianred',
    # 'indigo',
    # 'ivory',
    # 'khaki',
    # 'lavender',
    # 'lavenderblush',
    # 'lawngreen',
    # 'lemonchiffon',
    # 'lightblue',
    # 'lightcoral',
    # 'lightcyan',
    # 'lightgoldenrodyellow',
    # 'lightgreen',
    # 'lightgray',
    # 'lightpink',
    # 'lightsalmon',
    # 'lightseagreen',
    # 'lightskyblue',
    # 'lightslategray',
    # 'lightsteelblue',
    # 'lightyellow',
    # 'lime',
    # 'limegreen',
    # 'linen',
    # 'magenta',
    # 'maroon',
    # 'mediumaquamarine',
    # 'mediumblue',
    # 'mediumorchid',
    # 'mediumpurple',
    # 'mediumseagreen',
    # 'mediumslateblue',
    # 'mediumspringgreen',
    # 'mediumturquoise',
    # 'mediumvioletred',
    # 'midnightblue',
    # 'mintcream',
    # 'mistyrose',
    # 'moccasin',
    # 'navajowhite',
    # 'olive',
    # 'olivedrab',
    # 'orange',
    # 'orangered',
    # 'orchid',
    # 'palegoldenrod',
    # 'palegreen',
    # 'paleturquoise',
    # 'palevioletred',
    # 'papayawhip',
    # 'peachpuff',
    # 'peru',
    # 'pink',
    # 'plum',
    # 'powderblue',
    # 'purple',
    # 'red',
    # 'rosybrown',
    # 'royalblue',
    # 'saddlebrown',
    # 'salmon',
    # 'sandybrown',
    # 'seagreen',
    # 'seashell',
    # 'sienna',
    # 'silver',
    # 'skyblue',
    # 'slateblue',
    # 'slategray',
    # 'snow',
    # 'springgreen',
    # 'steelblue',
    # 'tan',
    # 'teal',
    # 'thistle',
    # 'tomato',
    # 'turquoise',
    # 'violet',
    # 'wheat',
    # 'yellow',
    # 'yellowgreen',
    # # 'floralwhite', 'ghostwhite', 'navy','oldlace', 'white','whitesmoke','antiquewhite',
)

linecolors_colorblind_safe=(
    (204./256.,102./256.,119./256.),
    (61./256., 170./256.,153./256.),
    (51./256., 34./256., 136./256.),
    ## (17./256., 119./256.,51./256.),
    (170./256.,68./256., 153./256.),
    ## (136./256.,34./256., 85./256.),
    (153./256.,153./256.,51./256.),
    (136./256.,204./256.,238./256.),
    (221./256.,204./256.,199./256.),
    (51./256., 102./256.,170./256.),
    (17./256., 170./256.,153./256.),
    (102./256.,170./256.,85./256.),
    (153./256.,34./256., 136./256.),
    (238./256.,51./256., 51./256.),
    (238./256.,119./256.,34./256.),
    ## (204./256.,204./256.,85./256.),
    ## (255./256.,238./256.,51./256.),
    ## (119./256.,119./256.,119./256.),
)   ## from http://www.sron.nl/~pault/

## from http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=6
linecolors_print=(
    # ## attempt1
    # '#a50026',
    # '#f46d43',
    # '#fdae61',
    # '#fee090',
    # '#74add1',
    # '#4575b4',
    # '#4575b4',
    # '#313695',
    # '#d73027',
    # '#abd9e9',
    # '#e0f3f8',
    ## attempt 2
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#a65628',
    '#f781bf',
    '#ffff33', # light yellow
)

# linecolors = mpl.rcParams['axes.color_cycle']
linecolors = [f['color'] for f in matplotlib.rcParams['axes.prop_cycle']]

def newcolor(index=None,reset=None,linecolors=None):
    """Retuns a color string, different to the last one, from the list
    linecolors. If reset is set, returns to first element of
    linecolors, no color is returned. If index (int) is supplied
    return this color. If index is supplied and reset=True, set index
    to this color. """
    global _newcolor_nextcolor
    if linecolors is None:
        linecolors = [f['color'] for f in matplotlib.rcParams['axes.prop_cycle']]
        # linecolors = [f for f in mpl.rcParams['axes.color_cycle']]
    if reset!=None or index in ['None','none','']:
        _newcolor_nextcolor=0
        return
    if index is not None:
        ## index should be an int -- but make it work for anything
        try:
            index = int(index)
        except (TypeError,ValueError):
             # index = id(index)
             index = hash(index)
        if reset:
            _newcolor_nextcolor = (index) % len(linecolors)
        return(linecolors[(index) % len(linecolors)])

    retval = linecolors[_newcolor_nextcolor]
    _newcolor_nextcolor = (_newcolor_nextcolor+1) % len(linecolors)
    return retval


_newlinestyle_nextstyle=0
linestyles=(
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    (0,(3,2,1,2,1,2)),          # -..
    (0,(1,2,3,2,3,2)),          # --.
)
def newlinestyle(index=None,reset=None):
    """Retuns a style string, different to the last one, from the list
    linestyles. If reset is set, returns to first element of
    linestyles, no style is returned."""
    global linestyles,_newlinestyle_nextstyle
    if reset!=None:
        _newlinestyle_nextstyle=0
        return
    if index is not None:
        _newlinestyle_nextstyle = (index) % len(linestyles)
    retval = linestyles[_newlinestyle_nextstyle]
    _newlinestyle_nextstyle = (_newlinestyle_nextstyle+1) % len(linestyles)
    return retval

_newmarker_nextmarker=0
# markers=('o','x','t','d')
markers=("o","s","d","p","v","^","<",">","1","2","3","4","8","*","+","x","h","H","D","|","_",)
def newmarker(index=None,reset=None):
    """Retuns a marker type string, different to the last one, from
    the list markers. If reset is set, returns to first element of
    markers, no style is returned."""
    global markers,_newmarker_nextmarker
    if reset!=None:
        _newmarker_nextmarker=0
        return 
    if index is not None:
        index = int(index)
        _newmarker_nextmarker = (index) % len(markers)
    retval = markers[_newmarker_nextmarker]
    _newmarker_nextmarker = (_newmarker_nextmarker+1) % len(markers)
    return retval

def newcolorlinestyle(index=None,reset=None):
    """Returns a new color and line style if necessary. Cycles colors
first."""
    ## reset built in counters and do nothing else
    if reset is not None:
        return(newcolor(reset=reset),newlinestyle(reset=reset))
    ## return combination according to an index
    if index is not None:
        color_index = index%len(linecolors)
        linestyle_index = (index-color_index)%len(linestyles)
        return(linecolors[color_index],linestyles[linestyle_index])
    ## iterate and get a new combination
    if _newcolor_nextcolor==len(linecolors)-1:
        return(newcolor(),newlinestyle())
    else:
        return(newcolor(),linestyles[_newlinestyle_nextstyle])
    
# def newcolormarker(reset=None):
#     """Returns a new color and line style if necessary."""
#     if reset is not None:
#         return(newcolor(reset=reset),newmarker(reset=reset))
#     if _newcolor_nextcolor==len(linecolors)-1:
#         return(newcolor(),newmarker())
#     else:
#         return(newcolor(),markers[_newmarker_nextmarker])

def newcolormarker(reset=None):
    """Returns a new color and line style if necessary."""
    if reset is not None:
        newcolor(reset=reset)
        newmarker(reset=reset)
    if _newcolor_nextcolor==len(linecolors)-1:
        color,marker = newcolor(),newmarker()
    else:
        color,marker = newcolor(),markers[_newmarker_nextmarker]
    return({'color':color,'marker':marker})

# def subplot(*args,**kwargs):
    # """Work out reasonable dimensions for an array of subplots with
    # total number n.\n
    # subplot(i,j,n) - nth subplot of i,j grid, like normal\n
    # subplot(i*j,n)   - switch to nth subplot out of a total of i*j in a sensible arrangement\n
    # subplot((i,j)) - add next subplot in i,j grid\n
    # subplot(i*j)   - total number of subplots, guess a good arrangment\n
    # All other kwargs are passed onto pyplot.subplot."""
    # assert len(args) in [1,2,3], 'Bad number of inputs.'
    # if 'fig' in kwargs:
        # fig = kwargs.pop('fig')
    # else:
        # fig=plt.gcf();
    # ## determine next subplot from how many are already drawn
    # if len(args)==1: 
        # if isinstance(args[0],tuple):
            # args=(args[0][0],args[0][1],len(fig.axes)+1,)
        # else:
            # args=(args[0],len(fig.axes)+1,)
    # ## send straight to subplot
    # if len(args)==3:
        # return fig.add_subplot(*args,**kwargs)
    # ## nice rounded dimensions
    # nsubplots,isubplot = args
    # rows = int(np.floor(np.sqrt(nsubplots)))
    # columns = int(np.ceil(float(nsubplots)/float(rows)))
    # ## if landscapish reverse rows/columns
    # if fig.get_figheight()>fig.get_figwidth(): (rows,columns,)=(columns,rows,)
    # return fig.add_subplot(rows,columns,isubplot,**kwargs)
# add_subplot=mysubplot=subplot
# mySubplot=subplot

def subplot(
        n=None,                 # subplot index, begins at 0, if None adds a new subplot
        ncolumns=None,          # how many colums (otherwise adaptive)
        nrows=None,          # how many colums (otherwise adaptive)
        ntotal=None,         # how many to draw in total (at least)
        fig=None,            # Figure object or number identifier
        **add_subplot_kwargs
):
    """Return axes n from figure fig containing subplots.\n If subplot
    n does not exist, return a new axes object, possibly shifting all
    subplots to make room for ti. If axes n already exists, then
    return that. If ncolumns is specified then use that value,
    otherwise use internal heuristics.  n IS ZERO INDEXED"""
    if fig is None:
        ## default to current fkgure
        fig = plt.gcf()
    elif isinstance(fig,int):
        fig = plt.figure(fig)
    old_axes = fig.axes           # list of axes originally in figure
    old_nsubplots = len(fig.axes) # number of subplots originally in figure
    if n is None:
        ## new subplot
        n = old_nsubplots
    if ntotal is not None and ntotal > old_nsubplots:
        ## current number of subplot is below ntotal, make empty subplots up to this number
        ax = subplot(ntotal-1,ncolumns,nrows,fig=fig,**add_subplot_kwargs)
        old_nsubplots = len(fig.axes) # number of subplots originally in figure
    if n < old_nsubplots:
        ## indexes an already existing subplot - return that axes
        ax = fig.axes[n]
    elif n > old_nsubplots:
        ## creating empty intervening subplot then add the requested new one
        for i in range(old_nsubplots,n):
            ax = subplot(i,ncolumns,nrows,fig=fig,**add_subplot_kwargs)
    else:
        ## add the nenew subplot 
        nsubplots = old_nsubplots+1
        if ncolumns is not None and nrows is not None:
            columns,rows = ncolumns,nrows
        elif ncolumns is None and nrows is None:
            rows = int(np.floor(np.sqrt(nsubplots)))
            columns = int(np.ceil(float(nsubplots)/float(rows)))
            if fig.get_figheight()>fig.get_figwidth(): 
                rows,columns = columns,rows
        elif ncolumns is None and nrows is not None:
            rows = nrows
            columns = int(np.ceil(float(nsubplots)/float(rows)))
        elif ncolumns is not None and nrows is None:
            columns = ncolumns
            rows = int(np.ceil(float(nsubplots)/float(columns)))
        else:
            raise Exception("Impossible")
        ## create new subplot
        ax = fig.add_subplot(rows,columns,nsubplots,**add_subplot_kwargs)
        ## adjust old axes to new grid of subplots
        gridspec = matplotlib.gridspec.GridSpec(rows,columns)
        for axi,gridspeci in zip(fig.axes,gridspec):
            axi.set_subplotspec(gridspeci)
    ## set to current axes
    fig.sca(ax)  
    ## set some other things to do if this is a qfig
    if hasattr(fig,'_my_fig') and fig._my_fig is True:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.10g'))
        ax.grid(True)
    return ax 

def get_axes_position(ax=None):
    """Get coordinates of top left and bottom right corner of
    axes. Defaults to gca(). Returns an array not a Bbox."""
    if ax is None: ax = plt.gca()
    return np.array(ax.get_position())
    
def set_axes_position(x0,y0,x1,y1,ax=None):
    """Set coordinates of bottom left and top right corner of
    axes. Defaults to gca()."""
    if ax is None: ax = plt.gca()
    return ax.set_position(matplotlib.transforms.Bbox(np.array([[x0,y0],[x1,y1]])))

def transform_points_into_axis_fraction(x,y,ax=None):
    """Convert width and height in points to an axes fraction."""
    if ax is None: ax = plt.gca()
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # size of window in inches
    width, height = bbox.width*72, bbox.height*72 # size of window in pts
    return(x/width,y/height)

def transform_points_into_data_coords(x,y,ax=None):
    """Convert width and height in points to to data coordinates."""
    if ax is None: ax = plt.gca()
    xaxes,yaxes = transform_points_into_axis_fraction(x,y,ax)
    return(xaxes*np.abs(np.diff(ax.get_xlim())),yaxes*np.abs(np.diff(ax.get_ylim())))

def transform_axis_fraction_into_data_coords(x,y,ax=None):
    """Convert width and height in points to to data coordinates."""
    if ax is None: ax = plt.gca()
    (xbeg,xend),(ybeg,yend) = ax.get_xlim(),ax.get_ylim()
    xout = xbeg + x*(xend-xbeg)
    yout = ybeg + y*(yend-ybeg)
    return(xout,yout)

def subplotsCommonAxes(fig=None,):
    """Issue this comman with a optional Figure object and all x and y
    ticks and labels will be turned off for all subplots except the
    leftmost and bottommost. (Hopefully.)"""
    ## havent yet implemented turning off labels
    if fig is None: fig = plt.gcf()
    for ax in fig.axes:
        bbox = ax.get_position()
        if abs(bbox.x0-fig.subplotpars.left)>1e-2:
            ax.set_yticklabels([],visible=False)
        if abs(bbox.y0-fig.subplotpars.bottom)>1e-2:
            ax.set_xticklabels([],visible=False)

def add_xaxis_alternative_units(
        ax,
        transform,              # a function, from existing units to new
        inverse_transform=None,
        fmt='0.3g',             # for tick labels
        label='',               # axis label
        labelpad=None,               # axis label
        ticks=None,             # in original units
        minor=False,            # by default minor tick labels are turned off
        **set_tick_params_kwargs
):
    """Make an alternative x-axis (on top of plot) with units
    transformed by a provided function. Scale will always be linear."""
    ax2 = ax.twiny()
    if ticks is not None:
        assert inverse_transform is not None,'inverse_transform required if ticks are specified'
        ax2.xaxis.set_ticks(np.sort(inverse_transform(np.array(ticks))))
    ax2.xaxis.set_tick_params(**set_tick_params_kwargs)
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticklabels([format(transform(t),fmt) for t in ax2.xaxis.get_ticklocs()])
    if not minor: ax2.xaxis.set_ticks([],minor=True)
    ax2.set_xlabel(label,labelpad=labelpad)   # label
    return(ax2)

def add_yaxis_alternative_units(
        ax,
        transform,              # a function, from existing units to new
        fmt='0.3g',             # for tick labels
        label='',               # axis label
        labelpad=None,               # axis label
        ticks=None,             # in original units
        minor=False,            # by default minor tick labels are turned off
        **set_tick_params_kwargs
):
    """Make an alternative x-axis (on top of plot) with units
    transformed by a provided function. Scale will always be linear."""
    ax2 = ax.twinx(sharex=True)
    if ticks is not None: ax2.yaxis.set_ticks(np.sort(ticks))
    ax2.yaxis.set_tick_params(**set_tick_params_kwargs)
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_ticklabels([format(transform(t),fmt) for t in ax2.yaxis.get_ticklocs()])
    if not minor: ax2.yaxis.set_ticks([],minor=True)
    ax2.set_ylabel(label,labelpad=labelpad)   # label
    return(ax2)

def make_axes_limits_even(ax=None,beg=None,end=None,square=None):
    """Make x and y limits the same"""
    if ax is None:
        ax = plt.gca()
    if square is not None:
        beg,end = -square,square
    if beg is None:
        beg = min(ax.get_xlim()[0],ax.get_ylim()[0])
        end = max(ax.get_xlim()[1],ax.get_ylim()[1])
    ax.set_xlim(beg,end)
    ax.set_ylim(beg,end)
    return(beg,end)


def set_tick_interval(x_or_y,major_interval=1,major_minor_ratio=1,minor_interval=None,ax=None):
    """Set ticks to a certain spacing."""
    ## process inputs
    if ax is None:
        ax = plt.gca()
    if minor_interval is None:
        minor_interval = major_interval/major_minor_ratio
    ## set major
    xbeg,xend = ax.get_xlim()
    xbeg,xend = int(xbeg-xbeg%major_interval+major_interval),int(xend-xend%major_interval)
    xticks = np.arange(xbeg,xend+major_interval/2,major_interval)
    ax.set_xticks(xticks,minor=False)
    ## set minor
    xbeg,xend = ax.get_xlim()
    xbeg,xend = int(xbeg-xbeg%minor_interval+minor_interval),int(xend-xend%minor_interval)
    xticks = np.arange(xbeg,xend+minor_interval/2,minor_interval)
    ax.set_xticks(xticks,minor= True)

# def add_yaxis_alternative_units(ax,transform,label='',fmt='0.3g',ticks=None,minor=False):
    # """Make an alternative y-axis (on top of plot) with units
    # transformed by a provided function."""
    # ax2 = ax.twinx()
    # if ticks!=None:
        # ax2.yaxis.set_ticks(ticks)
    # ax2.set_ylim(ax.get_ylim())
    # ax2.yaxis.set_ticklabels([format(transform(t),fmt) for t in ax2.yaxis.get_ticklocs()])
    # if not minor: ax2.yaxis.set_ticks([],minor=True)
    # ax2.set_ylabel(label)   # label
    # return(ax2)
            
def get_range_of_lines(line):
    """Find the maximum of a list of line2D obejts. I.e. a plotted
    line or errorbar, or an axes.. Returns (xmin,ymin,xmax,ymax)."""
    ymin = xmin = ymax = xmax = np.nan
    for t in line:
        x = t.get_xdata()
        y = t.get_ydata()
        i = ~np.isnan(y*x)
        if not any(i): continue
        x,y = x[i],y[i]
        if np.isnan(ymin):
            ymin,ymax = y.min(),y.max()
            xmin,xmax = x.min(),x.max()
        else:
            ymax,ymin = max(y.max(),ymax),min(y.min(),ymin)
            xmax,xmin = max(x.max(),xmax),min(x.min(),xmin)
    return(xmin,ymin,xmax,ymax)

def texEscape(s):
    """Escape TeX special characters in string. For filenames and
    special characters when using usetex."""
    return (re.sub(r'([_${}\\])', r'\\\1', s))
tex_escape=texEscape # remove one day


def legend(
        *plot_kwargs_or_lines,  # can be dicts of plot kwargs including label
        ax=None,                # axis to add legend to
        include_ax_lines=True, # add current lines in axis to legend
        color_text= True,     # color the label text
        show_style=False,      # hide the markers
        in_layout=False,       # constraining tight_layout or not
        allow_multiple_axes=False,
        **legend_kwargs,        # passed to legend
):
    """Make a legend and add to axis. Operates completely outside the
    normal scheme."""
    if ax is None:
        ax = plt.gca()
    def _reproduce_line(line): # Makes a new empty line with the properties of the input 'line'
        new_line = plt.Line2D([],[]) #  the new line to fill with properties
        for key in ('alpha','color','fillstyle','label',
                    'linestyle','linewidth','marker',
                    'markeredgecolor','markeredgewidth','markerfacecolor',
                    'markerfacecoloralt','markersize','markevery',
                    'solid_capstyle','solid_joinstyle',): # add all these properties
            if hasattr(line,'get_'+key):                  # if the input line has this property
                getattr(new_line,'set_'+key)(getattr(line,'get_'+key)()) # copy it to the new line
            elif hasattr(line,'get_children'): # if it does not but has children (i.e., and errorbar) then search in them for property
                for child in line.get_children():
                    if hasattr(child,'get_'+key): # property found
                        try:                      # try to set in new line, if ValueError then it has an invalid valye for a Line2D
                            getattr(new_line,'set_'+key)(getattr(child,'get_'+key)())
                            break # property added successfully search no more children
                        except ValueError:
                            pass
                    else:       # what to do if property not found anywhere
                        pass    # nothing!
        return(new_line)
    ## collect line handles and labels
    handles,labels = [],[]
    ## add existing lines in axis to legend
    if include_ax_lines:
        for handle,label in zip(*ax.get_legend_handles_labels()):
            if label!='_nolegend':
                labels.append(label)
                handles.append(_reproduce_line(handle))
    ## add get input lines or kwargs
    for i,t in enumerate(plot_kwargs_or_lines):
        if isinstance(t,matplotlib.lines.Line2D) or isinstance(t,matplotlib.container.ErrorbarContainer):
            raise Exception("Does not currently work for some reason.")
            t = t[0]
            if t.get_label()!='_nolegend':
                labels.append(t.get_label())
                handles.append(_reproduce_line(t))
        elif isinstance(t,dict):
            if t['label']!='_nolegend_':
                labels.append(t['label'])
                handles.append(plt.Line2D([],[],**t))
        else:
            raise Exception(f'Unhandled plot container type: {type(t)}')
    ## hide markers if desired
    if not show_style:
        for t in handles:
            t.set_linestyle('')
            t.set_marker('')
        legend_kwargs['handlelength'] = 0
        legend_kwargs['handletextpad'] = 0
    ## make a legend
    legend_kwargs.setdefault('handlelength',2)
    legend_kwargs.setdefault('loc','best')
    legend_kwargs.setdefault('frameon',False)
    legend_kwargs.setdefault('framealpha',1)
    legend_kwargs.setdefault('edgecolor','black')
    legend_kwargs.setdefault('fontsize','medium')
    if len(labels)==0: return(None)
    leg = ax.legend(labels=labels,handles=handles,**legend_kwargs)
    leg.set_in_layout(False)
    ## color the text
    if color_text:
        for text,handle in zip(leg.get_texts(),handles):
            if isinstance(handle,matplotlib.container.ErrorbarContainer):
                color = handle[0].get_color()
            else:
                color = handle.get_color()
            text.set_color(color)
    ## add to axis
    if allow_multiple_axes:
        ax.add_artist(leg) 
    return leg

legend_from_kwargs = legend
legend_colored_text = legend


def supylabel(text,fig=None,x=0.01,y=0.5,**kwargs):
    """Set ylabel for entire figure at bottom. x,y to adjust position
    in figure fraction."""
    kwargs.setdefault('va','center')
    kwargs.setdefault('ha','left')
    kwargs.setdefault('rotation',90)
    # kwargs.setdefault('fontsize','large')
    if fig is None:
        fig = plt.gcf()
    fig.text(x,y,text,**kwargs)

def supxlabel(text,fig=None,x=0.5,y=0.02,loc='bottom',**kwargs):
    """Set xlabel for entire figure at bottom. x,y to adjust position
    in figure fraction."""
    if loc=='bottom':
        x,y=0.5,0.01
        kwargs.setdefault('va','bottom')
        kwargs.setdefault('ha','center')
    if loc=='top':
        x,y=0.95,0.98
        kwargs.setdefault('va','top')
        kwargs.setdefault('ha','center')
    kwargs.setdefault('rotation',0)
    # kwargs.setdefault('fontsize','large')
    if fig is None: fig = plt.gcf()
    fig.text(x,y,text,**kwargs)

def suplegend(
        fig=None,
        lines=None,
        labels=None,
        ax=None,
        loc='below',
        frame_on=True,
        **legend_kwargs
):
    """Plot a legend for the entire figure. This is useful when
    several subplots have the same series and common legend is more
    efficient. """
    if fig    is None: fig    = plt.gcf()
    if ax     is None: ax     = plt.gca()
    if lines  is None: lines  = [l for l in ax.get_lines() if l.get_label() != '_nolegend_']
    if labels is None: labels = [l.get_label() for l in lines]
    legend_kwargs.setdefault('numpoints',1)
    legend_kwargs.setdefault('fontsize','medium')
    legend_kwargs.setdefault('borderaxespad',1)   #try to squeeze between edge of figure and axes title
    ## not accpeted directly by fig.legend
    fontsize = legend_kwargs.pop('fontsize')   
    ## put legend
    if loc in ('below','bottom'):
        legend_kwargs.setdefault('ncol',3)
        legend_kwargs['loc'] = 'lower center'
    elif loc in ('above','top'):
        legend_kwargs.setdefault('ncol',3)
        legend_kwargs['loc'] = 'upper center'
    elif loc=='right':
        legend_kwargs.setdefault('ncol',1)
        legend_kwargs['loc'] = 'center right'
    else:
        legend_kwargs['loc'] = loc
    leg = fig.legend(lines,labels,**legend_kwargs)
    ## remove frame and resize text
    if not frame_on:
        leg.draw_frame(False)
    for t in leg.get_texts(): 
        t.set_fontsize(fontsize)
    return(leg)

def turnOffOffsetTicklabels(ax=None):
    if ax is None: ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    

def noTickMarksOrLabels(axes=None,axis=None):
    """Turn off tick labels and marks, axis='x' or axis='y' affects
    only that axis."""
    if axes is None: axes=gca()
    if (axis is None) or (axis=='x'):
        axes.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        axes.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if (axis is None) or (axis=='y'):
        axes.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        axes.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
noTicMarksOrLabels = noTickMarksOrLabels

def simple_tick_labels(ax=None,axis=None,fmt='%g'):
    """Turn off fancy - but confusing scientific notation. Optional
    axis='x' or axis='y' to affect one axis o. Also sets the format of
    the mouse indicator to be 0.12g. """
    if ax is None: ax=plt.gca()
    # if (axis is None) | (axis=='x'):
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
        # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # if (axis is None) | (axis=='y'):
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
        # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax.format_coord = lambda x,y : "x={:0.12g} y={:0.12g}".format(x, y) 
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(fmt))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(fmt))
    ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(style='plain')
        
def tick_label_format(fmt,axis='both',ax=None):
    """Use a standard string format to format tick labels. Requires %
    symbol. Axis can be 'both', 'x',' or 'y'. Axes defaults to gca."""
    if ax is None: ax=plt.gca()
    if (axis=='both') | (axis=='x'):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
    if (axis=='both') | (axis=='y'):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))

def noTickLabels(axes=None,axis=None):
    """Turn off tick labels, axis='x' or axis='y' affects only that axis."""
    if axes is None: axes=plt.gca()
    if (axis is None) | (axis=='x'):
        axes.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if (axis is None) | (axis=='y'):
        axes.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def connect_zoom_in_axis(ax1,ax2,**line_kwargs):
    """Join two plots, one a zoom in of the other, by lines indicating
    the zoom. Requires ax1 to be located above ax2 on the
    figure. Requires that the xlim of one axis (either one) is
    completely within the xlim of the other."""
    ## Defining the line to draw
    line_kwargs.setdefault('lw',1)
    line_kwargs.setdefault('color','red')
    line_kwargs.setdefault('ls','-')
    line_kwargs.setdefault('zorder',-5)
    ## get locations in figure coordinates then draw line
    fig = ax1.figure
    transFigure = fig.transFigure.inverted()
    data1_beg,data1_end = ax1.get_xlim()
    data2_beg,data2_end = ax2.get_xlim()
    coord1 = transFigure.transform(ax1.transAxes.transform([(max(data1_beg,data2_beg)-data1_beg)/(data1_end-data1_beg),0]))
    coord2 = transFigure.transform(ax2.transAxes.transform([(max(data1_beg,data2_beg)-data2_beg)/(data2_end-data2_beg),1]))
    line1 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]), transform=fig.transFigure,**line_kwargs)
    fig.lines.append(line1)
    coord1 = transFigure.transform(ax1.transAxes.transform([(min(data1_end,data2_end)-data1_beg)/(data1_end-data1_beg),0]))
    coord2 = transFigure.transform(ax2.transAxes.transform([(min(data1_end,data2_end)-data2_beg)/(data2_end-data2_beg),1]))
    line2 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]), transform=fig.transFigure,**line_kwargs)
    fig.lines.append(line2)
    ## add vertical lines
    line_kwargs['lw'] = 0
    if (data1_beg>data2_beg)&(data1_end<data2_end):
        ax2.axvline(data2_beg,**line_kwargs)
        ax2.axvline(data2_end,**line_kwargs)
    if (data2_beg>data1_beg)&(data2_end<data1_end):
        # ax1.axvline(data2_beg,**line_kwargs)
        # ax1.axvline(data2_end,**line_kwargs)
        
        ax1.axvspan(data2_beg,data2_end,**line_kwargs)
        # ax1.axvline(data2_end,**line_kwargs)
    return((line1,line2))
        
def arrow(
        x1y1,
        x2y2,
        arrowstyle='->',
        label=None,
        xcoords='data',      # affects both ends of arrow
        ycoords='data',      # affects both ends of arrow
        ax=None,
        color='black',
        labelpad=1,
        fontsize=10,
        **arrow_kwargs):
    """
    Trying to make a nice simple arrow drawing function.  kwargs are
    passed directly to matplotlib.Patches.FancyArrowPatch. There are
    some custom defaults for these.
    """
    arrowParams={
        'linewidth':1.,
        'linestyle':'solid',
        'edgecolor':color, # also colors line
        'facecolor':color,
        'arrowstyle':arrowstyle, # simple,wedge,<->,-] etc
        'mutation_scale':10, # something to with head size
        'shrinkA':0.0,       # don't truncate ends
        'shrinkB':0.0,       # don't truncate ends
        }
    arrowParams.update(arrow_kwargs)
    if ax is None: ax = plt.gca()
    if xcoords=='data':
        xtransform = ax.transData
    elif xcoords=='axes fraction':
        xtransform = ax.transAxes
    else:
        raise Exception("unkonwn xcoords "+repr(xcoords))
    if ycoords=='data':
        ytransform = ax.transData
    elif ycoords=='axes fraction':
        ytransform = ax.transAxes
    else:
        raise Exception("unkonwn ycoords "+repr(ycoords))
    arrow = ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            x1y1,x2y2,
            transform=matplotlib.transforms.blended_transform_factory(xtransform,ytransform),
            **arrowParams),)
    ## add label parallel to arrow
    if label is not None:
        ## transform to display coordinates
        x1,y1 = ax.transData.transform(x1y1)
        x2,y2 = ax.transData.transform(x2y2)
        midpoint = (0.5*(x1+x2),0.5*(y1+y2))
        try:
            angle  = np.arctan((y2-y1)/(x2-x1))
        except ZeroDivisionError:
            angle = np.pi/2.
        ax.annotate(str(label),
                    ax.transData.inverted().transform(midpoint),
                    xycoords='data',
                    xytext=(-labelpad*fontsize*np.sin(angle),labelpad*fontsize*np.cos(angle)),
                    # xytext=(0,0),
                    textcoords='offset points',
                    rotation=angle/np.pi*180,
                    ha='center',va='center',fontsize=fontsize,color=color)
    return(arrow)

myArrow=arrow

def annotate_corner(
        string,
        loc='top left',
        ax=None,
        fig=None,
        xoffset=5,
        yoffset=5,
        **kwargs
):
    """Put string in the corner of the axis. Location as in
    legend. xoffset and yoffset in points."""
    if fig is None: fig = plt.gcf()
    if ax is None: ax = fig.gca()
    ## get x and y offsets from axes edge 
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) #bbox in inches
    dx = xoffset/(bbox.width*72)          #in axes fraction
    dy = yoffset/(bbox.height*72)         #in axes fraction
    if loc in ['top left','tl','upper left']:
        xy,ha,va = (dx,1.-dy),'left','top'
    elif loc in ['bottom left','bl','lower left']:
        xy,ha,va = (dx,dy),'left','bottom'
    elif loc in ['top right','tr','upper right']:
        xy,ha,va = (1-dx,1-dy),'right','top'
    elif loc in ['bottom right','br','lower right']:
        xy,ha,va = (1-dx,dy),'right','bottom'
    elif loc in ['center left' ,'centre left' ]:
        xy,ha,va = (dx,0.5),'left','center'
    elif loc in ['top center','top centre','center top','centre top','upper center','upper centre','center upper','centre upper']:
        xy,ha,va = (0.5,1-dy),'center','top'
    elif loc in ['bottom center','bottom centre','center bottom','centre bottom','lower center','lower centre','center lower','centre lower']:
        xy,ha,va = (1-dx,0.5),'center','bottom'
    elif loc in ['center right','centre right']:
        xy,ha,va = (1-dx,0.5),'right','center'
    else:
        ValueError('Bad input loc')
    return ax.annotate(string,xy,xycoords='axes fraction',ha=ha,va=va,**kwargs)

def annotate_line(
        string=None,
        xpos='ymax',            # ymax,ymin,left,right,peak,minimum
        ypos='above',
        line=None,
        ax=None,
        color=None,
        xoffset=0,yoffset=0,   # pts
        text_auto_offset=False,
        text_x_offset=0,text_y_offset=0,
        **annotate_kwargs):
    """Put a label above or below a line. Defaults to legend label and
    all lines in axes.  xpos can be [min,max,left,right,peak,a
    value]. ypos in [above,below,a value].  First plots next to first
    point, last to last. Left/right fixes to left/rigth axis."""
    ## default to current axes
    if ax is None and line is None:
        ax = plt.gca()
    ## no lines, get list from current axes
    if line is None:
        line = ax.get_lines()
    ## list of lines, annotate each individually
    if np.iterable(line):
        return([annotate_line(
            string=string,xpos=xpos,ypos=ypos,line=l,
            ax=ax,color=color,xoffset=xoffset,yoffset=yoffset,**annotate_kwargs) for l in line])
    if ax is None: ax = line.axes
    if string is None:
        if re.match('_line[0-9]+',line.get_label()): return None
        string = line.get_label()
    ## a shift to make space around text
    if text_auto_offset:
        text_x_offset,text_y_offset = transform_points_into_axis_fraction(
            matplotlib.rcParams['font.size']/2,matplotlib.rcParams['font.size']/2)
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    xdata,ydata = line.get_data()
    xdata,ydata = np.asarray(xdata),np.asarray(ydata)
    if len(xdata)==0: return    # no line
    if xpos in (None,'left'):
        annotate_kwargs['xycoords'] = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
        xpos = text_x_offset
        annotate_kwargs.setdefault('ha','left')
    elif xpos in (None,'right'):
        annotate_kwargs['xycoords'] = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
        xpos = 1-text_x_offset
        annotate_kwargs.setdefault('ha','right')
    elif xpos in ('min','xmin',):
        xpos = xdata.min()
        annotate_kwargs.setdefault('ha','right')
    elif xpos in ('max','xmax'):
        xpos = xdata.max()
        annotate_kwargs.setdefault('ha','left')
    elif xpos in ('peak','ymax',):
        xpos = line.get_xdata()[np.argmax(ydata)]
        annotate_kwargs.setdefault('ha','center')
    elif xpos in ('minimum','ymin',):
        xpos = line.get_xdata()[np.argmin(ydata)]
        annotate_kwargs.setdefault('ha','center')
    elif tools.isnumeric(xpos):
        annotate_kwargs.setdefault('ha','center')
    else:
        raise Exception('bad xpos: ',repr(xpos))
    i = np.argmin(np.abs(xdata-xpos))
    if ypos in ['above','top']:
        annotate_kwargs.setdefault('va','bottom')
        ypos = ydata[i] + text_y_offset
    elif ypos in ['below','bottom']:
        annotate_kwargs.setdefault('va','top')
        ypos = ydata[i] - text_y_offset
    elif ypos in ['center']:
        annotate_kwargs.setdefault('va','center')
        ypos = ydata[i]
    elif ypos == None or tools.isnumeric(ypos):
        annotate_kwargs.setdefault('va','center')
    else:
        raise Exception('bad ypos: ',repr(ypos))
    if color is None: 
        color = line.get_color()
    ## draw label
    if string=='_nolegend_': string=''
    xoffset,yoffset = transform_points_into_data_coords(xoffset,yoffset)
    annotate_kwargs.setdefault('in_layout',False)
    annotation = ax.annotate(string,(float(xpos+xoffset),float(ypos+yoffset)),color=color,**annotate_kwargs)
    return annotation

def annotate(*args,**kwargs):
    """Adds default arrow style to kwargs, otherwise the same as
    numpy.annotate, which by default doesn't draw an arrow
    (annoying!)."""
    ## default arrow style, if no arrows coords then draw no arrow
    kwargs.setdefault('arrowprops',dict(arrowstyle="->",))
    if 'linewidth' in kwargs:
        kwargs['arrowprops']['linewidth']=kwargs.pop('linewidth')
    if 'color' in kwargs:
        kwargs['arrowprops']['edgecolor']=kwargs['color']
        kwargs['arrowprops']['facecolor']=kwargs['color']
    if len(args)<3:
        kwargs.pop('arrowprops')
    else:
        if not (isiterableNotString(args[1]) and isiterableNotString(args[2])):
            kwargs.pop('arrowprops')
    ax = plt.gca()
    a =  ax.annotate(*args,**kwargs)
    a.set_in_layout(False)
    return(a)

def annotate_point(label,x,y,ax=None,fontsize='medium',marker='o',linestyle='',color='black',**plot_kwargs):
    """Annotate and draw point."""
    if ax is None:
        ax = plt.gca()
    l = ax.plot(x,y,marker=marker,linestyle=linestyle,color=color,**plot_kwargs)
    a = ax.annotate(str(label),(x,y),fontsize=fontsize,color=color)
    return(l,a)

def clginput(return_coord='both',with_comma=False):
    """Get ginput and add to clipboard. return_coord can be 'both', 'x', or 'y'."""
    x = np.array(plt.ginput(-1))
    if return_coord=='both': 
        tools.cl(x)
    elif return_coord=='x':
        if with_comma:
            tools.cl('\n'.join([format(t,'#.15g')+',' for t in x[:,0]]))
        else:
            tools.cl(x[:,0])
    elif return_coord=='y':
        tools.cl(x[:,1])
    else: raise Exception("Unknown return_coord: ",repr(return_coord))

def ginput_spline_curve(x,y):
    """Interactively add points to a figure, whilst updating a spline
    curve connecting them. Returns selected points. Hit 'enter' to
    break the interactive loop. Middle button to select a point."""
    ## create figure and axes, and draw curve
    presetRcParams('screen',autolayout=True)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x,y,color=newcolor(0))
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    ## to break loop below on close figure window
    # def f(event): raise KeyboardInterrupt
    # fig.canvas.matplotlib_connect('close_event',f)
    ## main loop selecting spline points
    xbg,ybg = np.array([],dtype=float),np.array([],dtype=float) 
    while True:
        ## get a new background point selection, any funny business here
        ## indicates to exit the loop. For example "enter".
        try:
            xi,yi = plt.ginput(n=1,timeout=0)[0]
        except:
            break
        ## compute if close to an existing point
        distance = np.sqrt(((ybg-yi)/(ylim[-1]-ylim[0]))**2 + ((xbg-xi)/(xlim[-1]-xlim[0]))**2)
        i = distance<0.02
        ## if so remove the existing point, else add the new point
        if np.any(i):
            xbg,ybg = xbg[~i],ybg[~i]
        else:
            xbg = np.concatenate((xbg,[xi]))
            ybg = np.concatenate((ybg,[yi]))
        ## sort
        i = np.argsort(xbg)
        xbg,ybg = xbg[i],ybg[i]
        ## redraw curve
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.autoscale(False)
        ax.plot(x,y,color=newcolor(0))
        ## draw points
        ax.plot(xbg,ybg,ls='',marker='x',color=newcolor(1))
        ## draw spline background
        if len(xbg)>1:
            xspline = x[(x>=xbg[0])&(x<=xbg[-1])]
            yspline = spline(xbg,ybg,xspline)
            ax.plot(xspline,yspline,ls='-',color=newcolor(1))
        plt.draw()
    return(xbg,ybg)

def annotate_ginput(label='',n=1):
    """Get a point on figure with click, and save a line of code to
    clipboard which sets an annotation at this point."""
    d = plt.ginput(n=n)
    annotate_command = '\n'.join([
        'ax.annotate("{label:s}",({x:g},{y:g}))'.format(label=label,x=x,y=y)
        for x,y in d])
    cl(annotate_command)
    return(annotate_command)

def annotate_vline(label,xpos,ax=None,color='black',fontsize='medium',
                   label_ypos=0.98,labelpad=0,zorder=None,alpha=1,
                   annotate_kwargs=None,**axvline_kwargs):
    """Draw a vertical line at xpos, and label it."""
    if ax==None: ax = plt.gca()
    axvline_kwargs.setdefault('alpha',alpha)
    axvline_kwargs.setdefault('zorder',zorder)
    line_object = ax.axvline(xpos,color=color,**axvline_kwargs)
    transform = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ## align labels to top or bottom
    ## the label
    if annotate_kwargs is None: annotate_kwargs = {}
    annotate_kwargs.setdefault('color',color)
    annotate_kwargs.setdefault('alpha',alpha)
    annotate_kwargs.setdefault('zorder',zorder)
    if label_ypos>0.5:
        annotate_kwargs.setdefault('va','top')
    else:
        annotate_kwargs.setdefault('va','bottom')
    label_object = ax.annotate(label,
                               xy=(xpos+labelpad, label_ypos),
                               xycoords=transform,
                               rotation=90,
                               fontsize=fontsize,
                               # ha='right',
                               # color=color,
                               # backgroundcolor='white',
                               **annotate_kwargs
    )
    label_object.set_in_layout(False)
    return(line_object,label_object)

def annotate_hline(label,ypos,ax=None,color='black',fontsize='medium',va='bottom',
                   loc='right', # or 'left'
                   **axhline_kwargs):
    """Draw a vertical line at xpos, and label it."""
    if ax==None: ax = plt.gca()
    line_object = ax.axhline(ypos,color=color,**axhline_kwargs)
    transform = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData,)
    label_object = ax.annotate(
        label,
        xy=((0.98 if loc=='right' else 0.02), ypos),
        xycoords=transform,
        ha=('right' if loc=='right' else 'left'),
        va=va,color=color,fontsize=fontsize) 
    return(line_object,label_object)

def annotate_hspan(
        label,
        y0,y1,
        labelpos='bottom right',
        ax=None,
        color='black',
        fontsize='medium',
        alpha=0.3,
        zorder=-10,
        **axhspan_kwargs,
):
    """Draw a horizontal hspan with a label."""
    axhspan_kwargs.setdefault('linewidth',0) # no frame line
    if ax==None:
        ax = plt.gca()
    line_object = ax.axhspan(y0,y1,color=color,alpha=alpha,zorder=zorder,**axhspan_kwargs)
    if labelpos=='bottom right':
        ylabel,va = min(y0,y1),'bottom'
        xlabel,ha = 1,'right'
        label = label + ' '
    elif labelpos=='bottom left':
        ylabel,va = min(y0,y1),'bottom'
        xlabel,ha = 0,'left'
        label = ' '+label
    elif labelpos=='top right':
        ylabel,va = max(y0,y1),'top'
        xlabel,ha = 1,'right'
        label = label + ' '
    elif labelpos=='top left':
        ylabel,va = max(y0,y1),'top'
        xlabel,ha = 0,'left'
        label = ' '+label
    elif labelpos=='center':
        ylabel,va = 0.5*(y0+y1),'center'
        xlabel,ha = 0.5,'center'
    else:
        raise Exception(f'Bad labelpos: {repr(labelpos)}')
    transform = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData,)
    label_object = ax.annotate(
        label, xy=(xlabel,ylabel), xycoords=transform,
        ha=ha, va=va, color=color, fontsize=fontsize,
        zorder=zorder,) 
    return(line_object,label_object)


def set_text_border(text_object,border_color='black',face_color='white',border_width=1):
    """Set text_object to have a border."""
    import matplotlib.patheffects as path_effects
    text_object.set_color(face_color)
    text_object.set_path_effects([
        path_effects.Stroke(linewidth=border_width,foreground=border_color), # draw border
         path_effects.Normal(), # draw text on top
    ])

def plot_and_label_points(x,y,labels,ax=None,fontsize='medium',**plot_kwargs):
    """Plot like normal but also label each data point."""
    if ax is None: ax = plt.gca()
    l = ax.plot(x,y,**plot_kwargs)
    a = [ax.annotate(label,(xx,yy),fontsize=fontsize) for (xx,yy,label) in zip(x,y,labels)]
    return(l,a)


def plot_lines_and_disjointed_points(x,y,max_separation=1,ax=None,**plot_kwargs):
    """Plot continguous data (x separation < max_separation) as line
    segments and disjointed points with markers."""
    if ax is None: ax = plt.gca()
    plot_kwargs.setdefault('marker','o')
    plot_kwargs.setdefault('linestyle','-')
    plot_kwargs.setdefault('color',newcolor(0))
    if 'ls' in plot_kwargs:
        plot_kwargs['linestyle'] = plot_kwargs.pop('ls')
    i = np.argsort(x)
    x,y = x[i],y[i]
    d = np.diff(x)
    ## plot line segments
    kwargs = copy(plot_kwargs)
    kwargs['marker']=''
    for i in find(d<=max_separation):
        ax.plot(x[[i,i+1]],y[[i,i+1]],**kwargs)
    ## plot markers
    kwargs = copy(plot_kwargs)
    kwargs['linestyle']=''
    ## one point only
    if len(x)==1:
        ax.plot(x,y,**kwargs)
    ## disjointed points
    for i in find(d>max_separation):
        ## first point along
        if i==0:
            ax.plot(x[i],y[i],**kwargs)
        ## second jump
        if d[i-1]>max_separation:
            ax.plot(x[i],y[i],**kwargs)
        ## last point alone
        if i==len(d)-1:
            ax.plot(x[-1],y[-1],**kwargs)

def plot_sticks(x,y,ax=None,**plot_kwargs):
    """Plot as vertical lines"""
    assert len(x)==len(y)
    if ax is None:
        ax = plt.gca()
    x = np.row_stack((x,x,x))
    t = np.zeros(y.shape)
    y = np.row_stack((t,y,t))
    x = np.reshape(x.transpose(),np.prod(x.shape))                                                                                                                                                              
    y = np.reshape(y.transpose(),np.prod(y.shape))
    ax.plot(x,y,**plot_kwargs)

def plot_cumtrapz(x,y,ax=None,**plot_kwargs):
    if ax is None:
        ax = plt.gca()
    y = tools.cumtrapz(y,x)
    return ax.plot(x,y,**plot_kwargs)

def hist_with_normal_distribution(
        y,
        bins=None,
        ax=None,
):
    """Plot data y as a histogram along with a fitted normal
    distribution."""
    if ax is None:
        ax = plotting.gca()
    ax.cla()
    if bins is None:
        ## guess a sensible number of bins
        bins = max(10,int(len(y)/200))
    ax.hist(y,bins=bins,density=True)
    σ,μ = np.std(y),np.mean(y)
    x = np.linspace(*ax.get_xlim(),1000)
    yf = 1/(σ*np.sqrt(2*constants.pi))*np.exp(-1/2*((x-μ)/σ)**2)
    ax.plot(x,yf)
    ax.set_title(f'μ={μ:0.5g}, σ={σ:0.5g}')
    return ax

def axesText(x,y,s,**kwargs):
    """Just like matplotlib ax.text, except defaults to axes fraction
    coordinates, and centers text."""
    ax = plt.gca()
    kwargs.setdefault('transform',ax.transAxes)
    kwargs.setdefault('verticalalignment','center')
    kwargs.setdefault('horizontalalignment','center')
    return ax.text(x,y,s,**kwargs)

def set_ticks(
        x_or_y='x',
        locs=None,
        labels=None,
        spacing=None,
        divisions=None,
        ax=None,               # axes
        fontsize=None,
        rotation=None,
        **labels_kwargs
):
    ## get an axis
    if ax is None:
        ax = plt.gca()
    assert x_or_y in ('x','y')
    axis = (ax.xaxis if x_or_y=='x' else ax.yaxis)
    ## set major and minor locs
    if locs is not None:
        axis.set_ticks(locs)
    elif spacing is not None:
        beg,end = (ax.get_xlim() if x_or_y=='x' else ax.get_ylim())
        axis.set_ticks(np.arange(((beg+spacing*0.9999)//spacing)*spacing, end, spacing))
        if divisions is not None:
            minor_spacing = spacing/divisions
            axis.set_ticks(np.arange(((beg+minor_spacing*0.9999)//minor_spacing)*minor_spacing, end, minor_spacing),minor=True)
    ## set labels
    if labels is not None:
        axis.set_ticklabels(labels,**labels_kwargs)
    ## set rotation
    if rotation is not None:
        if x_or_y=='x':
            verticalalignment = 'top'
            horizontalalignment = 'right'
        else:
            verticalalignment = 'center'
            horizontalalignment = 'right'
        for label in axis.get_ticklabels():
            label.set_rotation(rotation)
            label.set_verticalalignment(verticalalignment)
            label.set_horizontalalignment(horizontalalignment)
   ## fontsize
    if fontsize is not None:
       for label in axis.get_ticklabels():
           label.set_size(fontsize)

def rotate_tick_labels(
        x_or_y='x',
        rotation=90,
        ax=None,
        verticalalignment=None,
        horizontalalignment=None,

):
    if ax is None: ax=plt.gca()
    assert x_or_y in ('x','y')
    if x_or_y=='x':
        labels = ax.xaxis.get_ticklabels()
        if verticalalignment is None:
            verticalalignment = 'top'
        if horizontalalignment is None:
            horizontalalignment = 'right'
    elif x_or_y=='y':
        labels = ax.yaxis.get_ticklabels()
        verticalalignment = 'center'
        horizontalalignment = 'right'
    else:
        raise Exception(f'Bad x_or_y: {repr(x_or_y)}')
    for t in labels:
        t.set_rotation(rotation)
        t.set_verticalalignment(verticalalignment)
        t.set_horizontalalignment(horizontalalignment)

def set_tick_labels_text(
        strings,
        locations=None,
        axis='x',
        ax=None,
        rotation=70,
        **set_ticklabels_kwargs,):
    """Set a list of strings as text labels."""
    if ax is None:
        ax = plt.gca()
    if axis=='x':
        axis = ax.xaxis
    elif axis=='y':
        axis = ax.yaxis
    if locations is None:
        locations = np.arange(len(strings))
    axis.set_ticks(locations)
    axis.set_ticklabels(strings,rotation=rotation,**set_ticklabels_kwargs)

def set_tick_spacing(
        axis='x',               # 'x' or 'y'
        major_spacing=1,        # absolute
        minor_divisions=None,       # number of minor tick intervals per major,None for default
        ax=None):
    """Simple method for adjusting major/minor tick mark spacing."""
    if ax == None: ax=plt.gca()
    assert axis in ('x','y'),'axis must be x or y'
    if axis=='x':
        axis,(beg,end) = ax.xaxis,ax.get_xlim()
    elif axis=='y':
        axis,beg,end = ax.yaxis,ax.get_ylim()
    axis.set_ticks(np.arange(
        ((beg+major_spacing*0.9999)//major_spacing)*major_spacing,
        end, major_spacing))
    if minor_divisions is not None:
        minor_spacing = major_spacing/minor_divisions
        axis.set_ticks(np.arange(
            ((beg+minor_spacing*0.9999)//minor_spacing)*minor_spacing,
            end, minor_spacing),minor=True)

def show(show_in_ipython=False,block=True):
    """ Show current plot in a customised way."""
    ## do nothing if in an ipython shell
    for n in plt.get_fignums():
        extra_interaction(fig=plt.figure(n))
        # if (toolbar:=plt.get_current_fig_manager().toolbar) is not None:
            # toolbar.setHidden(True) # hide toolbar -- only works on qt?
        # set_figsize_fullscreen()
        pass
    try:
        __IPYTHON__
        if show_in_ipython:
            plt.show()
        return
    except NameError:
        qupdate()
        plt.show(block=block)

def qplot(*plot_args,fig=None,show=False,**plot_kwargs):
    """Issue a plot command and then output to file."""
    ax = qax(fig)
    ax.plot(*plot_args,**plot_kwargs)
    legend()
    ax.grid(True)
    if show:
        plt.show()

def ylogerrorbar(x,y,dy,ylolimScale=0.9,*args,**kwargs):
    """Ensures lower bound of error bars doesn't go negative messing
    up drawing with log scale y axis. 
    
    If necessary setes lower error lim to ylolimScale*y.
    
    All args and kwargs passed to
    regular errorbars."""
    ax = plt.gcf().gca()
    y = np.array(y)
    dy = np.array(dy)
    dylower = copy(dy)
    i = dylower>=y
    dylower[i] = ylolimScale*y[i]
    ax.errorbar(x,y,yerr=np.row_stack((dylower,dy)),*args,**kwargs)
    

def qplotfile(filenames,showLegend=True):
    """
    Try to guess a good way to 2D plot filename.

    Probably needs more work.
    """

    ## ensure a list of names, not just one
    if isinstance(filenames,str):
        filenames = [filenames]

    ## begin figure
    f = plt.figure()
    a = f.gca()

    ## loop through all filenames
    for filename in filenames:

        ## get data
        filename = expand_path(filename)
        x = np.loadtxt(filename)

        ## plot data
        if x.ndim==1:
            a.plot(x,label=filename)
        else:
            for j in range(1,x.shape[1]):
                a.plot(x[:,0],x[:,j],label=filename+' '+str(j-1))

    ## show figure
    if showLegend:
        legend()
    f.show()

def savefig(path,fig=None,**kwargs):
    """Like fig.savefig except saves first to a temporary file in
    order to achieve close to instant file creation."""
    path = tools.expand_path(path)
    name,ext = os.path.splitext(path)
    directory,filename = os.path.split(name)
    mkdir(directory)
    tmp = tempfile.NamedTemporaryFile(suffix=ext)
    kwargs.setdefault('dpi',300)
    if fig is None: 
        fig = plt.gcf()
    # kwargs.setdefault('transparent',True)
    fig.savefig(tmp.name,**kwargs)
    shutil.copyfile(tmp.name,path)

def annotate_spectrum(
        x,                      # list of x position to mark
        labels=[],              # labels for each x
        ylevel=0,               # absolute ydata
        length=0.02,            # fraction of yaxis
        hoffset=0.,
        labelpad=None,
        plotkwargs=None,        # affect drawn lines
        textkwargs=None,        # affect label and name text
        ax=None,
        name=None,              # printed name
        namepos='right',
        namepad=None,
        namesize=None,
        labelsize=None,
        clip= True,
        label_replacements={},  # substitute keys with values in labels
        plot_vline=False,       # plot a axvline as well as ticks showing line position
        color='black'           # colour of everything
):
    """Put simple rotational series labels at list of x. Length
    of ticks and horizontal/vertical offset of labels from ticks is
    specified. Optional text and plot arguments are passed on."""
    ## process args
    if plotkwargs==None: plotkwargs={}
    textkwargs = copy(textkwargs)
    plotkwargs = copy(plotkwargs)
    ## eliminate nan, x
    x = np.array(x)
    i = np.isnan(x)
    x = x[~i]
    if len(labels)>0:
        labels = np.array(labels)[~i]
    plotkwargs.setdefault('color',color)
    plotkwargs.setdefault('linewidth',1.)
    plotkwargs.setdefault('clip_on',clip)
    plotkwargs.setdefault('in_layout',False)
    if textkwargs==None: textkwargs={}
    textkwargs.setdefault('annotation_clip',clip)
    textkwargs.setdefault('color',color)
    textkwargs.setdefault('in_layout',False)
    if ax==None: ax = plt.gca()
    (xlim,ylim,) = (ax.get_xlim(),ax.get_ylim(),) # save limits so not changed by this function
    if namepad==None:
        # namepad = 0.5*np.abs(length)*(xlim[1]-xlim[0])
        namepad = np.abs(length)
    length = length*(ylim[1]-ylim[0]) # convert length fraction into absolute scale
    if labelpad==None:
        labelpad = -0.4*length
    ## dwim text alignment
    textkwargs.setdefault('horizontalalignment','center')
    if np.sign(length)<0:
        textkwargs.setdefault('verticalalignment','bottom')
    else:
        textkwargs.setdefault('verticalalignment','top')
    ## draw back bone
    ax.plot([min(x),max(x)],[ylevel,ylevel],**plotkwargs)
    ## draw lines in Jpp,energy lists and annotate labels
    labelkwargs = copy(textkwargs)
    if labelsize is not None:
        labelkwargs['fontsize'] = labelsize
    labelkwargs['horizontalalignment'] = 'center'
    if labelpad>0:
        labelkwargs['verticalalignment'] = 'bottom'
    else:
        labelkwargs['verticalalignment'] = 'top'
    for (i,e) in enumerate(x):
        ax.plot([e,e],[ylevel,ylevel+length],**plotkwargs)
        if plot_vline:
            ax.axvline(e,**(plotkwargs|{'alpha':0.2}))
        if len(labels) > i:
            # label = format(labels[i])
            label = tools.format_string_or_general_numeric(labels[i])
            if label in list(label_replacements.keys()):
                label = label_replacements[label]
            ax.annotate(label,[e+hoffset,ylevel+labelpad],**labelkwargs)
    ## if necessary annotate name
    if name != None:
        namekwargs = copy(textkwargs)
        if namesize is not None:
            namekwargs['fontsize'] = namesize # possibly override textkwargs
        name_pad_left,name_pad_right = '',''
        if namepos=='float':
            tline,tlabel = annotate_hline(
                str(name),
                ylevel,ax=ax,
                linewidth=0,
                va='center',
                color=namekwargs['color'],
                fontsize=namekwargs['fontsize'],
            )
            tlabel.set_bbox({'facecolor':'white', 'alpha':0.8, 'pad':0})
            tlabel.set_color(color)
            tlabel.set_in_layout(False)
        else:
            if tools.isiterable(namepos): # must be shift coordinates
                if np.abs(x[0]-x.min())<np.abs(x[0]-x.max()):
                    namekwargs['horizontalalignment'] = 'left'
                else:
                    namekwargs['horizontalalignment'] = 'right'
                if namepos[1]<0:
                    namekwargs['verticalalignment'] = 'top'
                elif namepos[1]==0:
                    namekwargs['verticalalignment'] = 'center'
                elif namepos[1]>0:
                    namekwargs['verticalalignment'] = 'bottom'
                name_xpos,name_ypos = (x[0]+namepos[0],ylevel+namepos[1])
            elif namepos=='right':
                namekwargs['horizontalalignment'] = 'left'
                namekwargs['verticalalignment'] = 'center'
                if np.max(x)<xlim[1]: # plot to right of rightmost energy
                    name_xpos,name_ypos = (np.max(x)+namepad,ylevel)
                    name_pad_left = '   '
                else: # plot inside right margin
                    name_xpos,name_ypos = (xlim[1]+namepad,ylevel)
                    name_pad_left = ' '
            elif namepos=='left':
                namekwargs['horizontalalignment'] = 'right'
                namekwargs['verticalalignment'] = 'center'
                if np.min(x)>xlim[0]:
                    name_xpos,name_ypos = (np.min(x)-namepad,ylevel)
                    name_pad_right = '   '
                else:
                    name_xpos,name_ypos = (xlim[0]-namepad,ylevel)
                    name_pad_right = ' '
            elif namepos in ('above','top'):
                namekwargs['horizontalalignment'] = 'center'
                namekwargs['verticalalignment'] = 'bottom'
                name_xpos,name_ypos = (0.5*(np.max(x)+np.min(x)),ylevel+namepad)
            ax.annotate(name_pad_left+str(name)+name_pad_right,(name_xpos,name_ypos),**namekwargs)
    ## revert limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def annotate_spectrum_by_branch(
        line,             # Line object
        ybeg = 1,               # y position of first coord
        ystep = 0.1, # separation between branch annotations in data coords
        zkeys = ('branch',), # how to divide up into separate annotations, 
        xkey = 'ν',          # 
        label_key='J_u', # what quantum number to give as a rotational label
        label_frequency=False,
        match_qn=None,        # only annotate matching qn
        qn_not_to_label=(), # e.g., [{'Σ':-1,'Jpp':[0,2],}] would not label these, requries label_translate_function is None
        name_function = None, # a function to modify automatically generated branch names
        label_function = None, # a function to modify automatically generated rotational level labels
        **kwargs_annotate_spectrum # pass directly to annotate_spectrum
):
    """Annotate spectrum with separate line and names found in a
    Line object."""
    zkeys = tools.ensure_iterable(zkeys)
    zkeys = [t for t in zkeys if line.is_known(t)]
    if label_key is not None and not line.is_known(label_key):
        label_key = None
    line.assert_known(xkey)
    if label_key is not None:
        line.assert_known(label_key)
    retval = []
    iz = 0
    for iz,(qn,zline) in enumerate(line.unique_dicts_matches(*zkeys)):
        zline.assert_known(xkey)
        if label_key is not None:
            zline.assert_known(label_key)
        if match_qn is not None:
            zline.limit_to_matches(**match_qn)
        if len(zline)==0: continue
        ## get annotation name
        if name_function is None:
            name = line.default_zlabel_format_function(qn)
        else:
            name = name_function(qn)
        ## update kwargs for this annotation
        kwargs = copy(kwargs_annotate_spectrum)
        kwargs.setdefault('name',name)
        kwargs.setdefault('color',newcolor(iz))
        if label_function is not None:
            labels = [label_function(t) for t in zline.rows(keys=label_key)]
        elif label_key is None:
            labels = ['' for t in zline.rows()]
        elif np.isscalar(label_key):
            if tools.isnumeric(line[label_key][0]):
                labels = [format(t[label_key],'g') for t in zline.rows()]
            else:
                labels = [format(t[label_key]) for t in zline.rows()]
        else:
            labels = [repr(t) for t in zline[label_key]]
        ## remove some labels
        for keys_vals in qn_not_to_label :
            for ii in tools.find(zline.match(**keys_vals)):
                labels[ii] = ''
        ## add frequency to labels
        if label_frequency:
            labels = [f'{label}({x:0.4f})' for label,x in zip(labels,ztransition[xkey])]
        ## make annotation
        retval.append(annotate_spectrum(zline[xkey],labels=labels,ylevel=ybeg+iz*ystep,**kwargs))
    return retval

def plot_stick_spectrum(
        x,y,
        fig=None, ax=None,
        **plot_kwargs):
    assert len(x)==len(y)
    fig,ax = plotting.qfigax(fig,ax)
    x = np.row_stack((x,x,x))
    t = np.zeros(y.shape)
    y = np.row_stack((t,y,t))
    x = np.reshape(x.transpose(),np.prod(x.shape))                                                                                                                                                              
    y = np.reshape(y.transpose(),np.prod(y.shape))
    ax.plot(x,y,**plot_kwargs)


def plot_hist_with_fitted_gaussian(y,bins=100,ax=None):
    if ax is None:
        ax = plt.gca()
    ax.hist(y,bins=bins,density=True)
    σ = np.std(y)
    μ = np.mean(y)
    x = np.linspace(*ax.get_xlim(),1000)
    yf = 1/(σ*np.sqrt(2*constants.pi))*np.exp(-1/2*((x-μ)/σ)**2)
    ax.plot(x,yf,label=f'μ={μ:0.5g}, σ={σ:0.5g}')
    legend(ax=ax)
