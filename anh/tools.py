import re
import os
import warnings
from copy import copy
from pprint import pprint
import itertools
import io
import inspect
import sys
import functools
from time import perf_counter as timestamp

import numpy as np
from numpy import array,arange,nan,inf

#######################################################
## decorators / decorator factories / function tools ##
#######################################################

cache = functools.lru_cache

def vectorise(vargs=None,vkeys=None,dtype=None,cache=False):
    """Vectorise a scalar-argument scalar-return value function.  If
    all arguments are scalar return a scalar result. If vargs is None
    vectorise all arguments, if it is a list of indices vectorise only
    those arguments. vkeys otherwise indicates keys to vectorise by
    name. If dtype is given return value is an array of this type, else
    a list is returned. If cache is True then cache indivdual scalar
    function calls."""
    import functools
    def actual_decorator(function):
        ## get a cached version fo the function if requested
        if cache:
            ## will not work with dill for some reason
            function_maybe_cached = functools.lru_cache(function)
        else:
            function_maybe_cached = function
        @functools.wraps(function)
        def vectorised_function(*args,**kwargs):
            ## this block subtitutes into kwargs with keys taken from
            ## the function signature.  get signature arguments -- skip
            ## first "self"
            signature_keys = list(inspect.signature(function).parameters.keys())
            for iarg,(arg,signature_key) in enumerate(zip(args,signature_keys)):
                if signature_key in kwargs:
                    raise Exception(f'Positional argument also appears as keyword argument {repr(signature_key)} in function {repr(function.__name__)}.')
                kwargs[signature_key] = arg
            ## get list of arg keys that should be vectorised and those not, and the common length of vector data
            vector_kwargs = {}
            scalar_kwargs = {}
            length = None
            for i,(key,val) in enumerate(kwargs.items()):
                if (
                        ## isiterable(val)
                        isinstance(val,(list,np.ndarray))
                        and ((vkeys is not None and key in vkeys)
                             or (vargs is not None and i in vargs)
                             or (vkeys is None and vargs is None))
                ):
                    ## vector data
                    if length is None:
                        length = len(val)
                    elif len(val) != length:
                        raise Exception(f'Nonconstant length of vector arguments in arg {repr(key)}')
                    vector_kwargs[key] = val
                else:
                    ## scalar data
                    scalar_kwargs[key] = val
            ## calculate scalar results and combine
            if len(vector_kwargs) == 0:
                ## all scalar, do scalar calc
                return function_maybe_cached(**scalar_kwargs)
            else:
                ## compute for each vectorised arg combination
                if dtype is None:
                    ## if no dtype return as list -- compute each separately
                    retval = []
                    for i in range(length):
                        vector_kwargs_i = {key:val[i] for key,val in vector_kwargs.items()}
                        iretval = function_maybe_cached(**scalar_kwargs,**vector_kwargs_i)
                        retval.append(iretval)
                else:
                    ## if dtype return as array, only compute for unique array combinations
                    retval = np.empty(length,dtype=dtype)
                    unique_combinations_masks(*vector_kwargs.values())
                    for values,index in unique_combinations_masks(*vector_kwargs.values()):
                        retval[index] = function_maybe_cached(
                            **scalar_kwargs,
                            **{key:val for key,val in zip(vector_kwargs,values)})
            return retval
        return vectorised_function
    return actual_decorator

def vectorise_arguments(function):
    """Convert all arguments to arrays of the same length.  If all
    original input arguments are scalar then convert the result back
    to scalar."""
    @functools.wraps(function)
    def function_with_vectorised_arguments(*args):
        arglen = 0
        for arg in args:
            if np.isscalar(arg):
                continue
            if arglen==0:
                arglen = len(arg)
            else:
                assert arglen == len(arg),'Mismatching lengths of vector arguments.'
        if arglen==0:
            ## all scalar -- make length 1 and compute, returning as
            ## scalar
            return function(*[np.array([arg]) for arg in args])[0]
        else:
            ## ensure all arguments vector and compute a vector
            ## result
            return function(*[np.full(arglen,arg) if np.isscalar(arg) else np.asarray(arg) for arg in args])
    return function_with_vectorised_arguments

#############################
## things for dictionaries ##
#############################

def dict_to_kwargs(d,keys=None):
    """Expand a dict into evaluable kwargs. Default to all keys."""
    if keys is None:
        keys = d.keys() # default to all keys
    return(','.join([key+'='+repr(d[key]) for key in keys]))

def format_dict(
        input_dict,
        indent='',
        newline_depth=inf,
        blank_depth=-1,
        max_line_length=inf,
        _depth=0,
        keys=None,
        enclose_in_braces=True,
):
    """pprint dict recursively but repr non-dict elements."""
    lines = []
    if enclose_in_braces:
        lines.append('{')
        indent = '    '
    ## add all values, either on new line, or as subdict
    if keys is None:
        keys = list(input_dict.keys())
    for i,key in enumerate(keys):
        val = input_dict[key]
        if blank_depth >= _depth:
            prefix = '\n'+indent
        else:
            prefix = indent
        if (
                not isinstance(val,dict) # not a dict
                or _depth >= newline_depth    # already too deep
                or len(val) == 0         # empty dict
                or (len(val) == 1 and not any([isinstance(t,dict) for t in val.values()])) # dict contains no other dicts
            ):
            ## put on one line
            formatted_value = repr(val)
            if len(formatted_value) > max_line_length:
                formatted_value = '...'
            lines.append(f'{prefix}{repr(key):20}: {formatted_value},')
        else:
            ## expand as subdict
            subdict = format_dict(
                val,
                indent+"    ",
                _depth=_depth+1,
                newline_depth=newline_depth,
                blank_depth=blank_depth,
                max_line_length=max_line_length,
            )
            lines.append(f'{prefix}{repr(key):10}: {subdict},')
    ## close dictionary with blank line first if needed
    if blank_depth >= _depth:
        lines.append('')
    if enclose_in_braces:
        lines.append('}')
    ## add indents to non blank lines
    lines = [(indent*_depth+t if len(t)>0 else t) for t in lines]
    ## combine into one
    retval = '\n'.join(lines)
    return retval

def load_dict(filename,attrs):
    """Import filename and return a dictionary of its named attributes. If
    name is a string, return the value of that attribute.  If it is a
    list of strings then return a dictionary of attributes."""
    ## import file as 'temporary_module'
    from importlib.machinery import SourceFileLoader
    module = SourceFileLoader('_import_data_temporary_module',filename).load_module()
    ## get requested attributes
    if isinstance(attrs,str):
        retval = getattr(module,attrs)
    else:
        retval = {t:getattr(module,t) for t in attrs}
    ## as far as possible unimport module so different data can be
    ## loaded/reloaded with this function
    sys.modules.pop('_import_data_temporary_module')
    del module
    return retval

def save_dict(
        filename,
        header='from numpy import nan,inf\nfrom spectr.optimise import P,Fixed\n',
        format_dict_kwargs={},
        **names_dicts,):
    """Write names_dicts as into filename preceeded by header.  Designed
    to be imported as valid python."""
    mkdir(os.path.split(filename)[0])
    format_dict_kwargs.setdefault('blank_depth',0)
    with open(filename,'w') as fid:
        ## add header
        if header is not None:
            fid.write(header)
            fid.write('\n')
        ## add data as dict_expanded_repr
        for name,val in names_dicts.items():
            fid.write(f'{name} = {format_dict(val,**format_dict_kwargs)}\n')


############################
## mathematical functions ##
############################

def compute_matrix_of_function(A,*args,**kwargs):
    """2D only"""
    retval = np.matrix([[Aij(*args,**kwargs) for Aij in Ai] for Ai in A])
    return retval

def kronecker_delta(x,y):
    """1 if x==y else 0."""
    if np.isscalar(x) and np.isscalar(y): return(1 if x==y else 0) # scalar case
    if np.isscalar(x) and not np.isscalar(y): x,y = y,x            # one vector, get in right order
    retval = np.zeros(x.shape)
    retval[x==y] = 1
    return(retval)              # vector case


def tanh_transition(x,xa,xb,center,width):
    """Creates a smooth match between extreme values xa and xb on grid x.
    Uses a hyperbolic tangent centred at center with the given transition
    width."""
    return (np.tanh((x-center)/width)+1)/2*(xb-xa)+xa

def tanh_hat(x,xa,xb,center,ramp_width,top_width):
    """Creates a smooth match between extreme values xa and xb on grid x.
    Uses a hyperbolic tangent centred at center with the given transition
    width."""
    return (
        tanh_transition(x,xa,xb,center-top_width/2,ramp_width)
        -tanh_transition(x,xa,xb,center+top_width/2,ramp_width)
    )
    # if np.isscalar(x):
        # if x<=center:
            # return (np.tanh((x-center+top_width)/ramp_width)+1)/2*(xb-xa)+xa 
        # else:
            # return (np.tanh((center+top_width-x)/ramp_width)+1)/2*(xb-xa)+xa
    # else:
        # i = x<center
        # retval = np.empty(x.shape,dtype=float)
        # retval[i] = (np.tanh((x[i]-center-top_width)/ramp_width)+1)/2*(xb-xa)+xa
        # retval[~i] = (np.tanh((center+top_width-x[~i])/ramp_width)+1)/2*(xb-xa)+xa
        # return retval
    

def leastsq(func,
            x0,
            dx,
            R=100.,
            print_error_mesg=True,
            error_only=False,
            xtol=1.49012e-8,
            rms_noise=None,     # for calculation of uncertaintes use this noise level rather than calculate from fit residual. This is useful in the case of an imperfect fit.
):
    """
    Rejig the inputs of scipy.optimize.leastsq so that they do what I
    want them to.
    \nInputs:\n
      func -- The same as for leastsq.
      x0 -- The same as for leastsq.
      dx -- A sequence of the same length as x0 containing the desired
      absolute stepsize to use when calculating the finite difference
      Jacobean.
      R -- The ratio of two step sizes: Dx/dx. Where Dx is the maximum
      stepsize taken at any time. Note that this is only valid for the
      first iteration, after which leastsq appears to approximately
      double the 'factor' parameter.
      print_error_mesg -- if True output error code and message if failure
    \nOutputs: (x,sigma)\n
    x -- array of fitted parameters
    sigma -- error of these
    The reason for doing this is that I found it difficult to tweak
    the epsfcn, diag and factor parametres of leastsq to do what I
    wanted, as far as I can determine these behave in the following
    way:
    dx = x*sqrt(epsfcn) ; x!=0,
    dx = 1*sqrt(epsfcn) ; x==0.
    Default epsfcn=2.2e-16 on scucomp2.
    Dx = abs(x*100)      ; x!=0, factor is not set,
    Dx = abs(x*factor)   ; x!=0, factor is set,
    Dx = abs(factor)     ; x==0, factor is set,
    Dx = 100             ; x==0, factor is not set, diag is not set,
    Dx = abs(100/diag)   ; x==0, factor is not set, diag is set,
    Dx = abs(factor/diag); x==0, factor is set, diag is set.
    Many confusing cases, particularly annoying when initial x==0 and
    it is not possible to control dx or Dx individually for each
    parameter.
    My solution was to add a large value to each parameter so that
    there is little or no chance it will change magnitude during the
    course of the optimisation. This value was calculated separately
    for each parameter giving individual control over dx. I did not
    think of a way to also control Dx individually, instead the ratio
    R=Dx/dx may be globally set.
    """
    from scipy import optimize
    ## limit the number of evaluation to a minimum number to compute
    ## the uncertainty from the second derivative - make R small to
    ## improve performance? - Doesn't work for very large number of
    ## parameters - errors are all nan, probably because of a bug in
    ## leastsq?
    if error_only:
        maxfev = len(x0)+1
        R = 1.
    else:
        maxfev = 0
    ## try and wangle actual inputs of numpy.leastsq to get the right
    ## step sizes
    x0=np.array(x0)
    if np.isscalar(dx): dx = np.full(x0.shape,dx)
    dx=np.array(dx)
    epsfcn = 1e-15              # required that sqrt(epsfcn)<<dp/p
    xshift = x0+dx/np.sqrt(epsfcn)    # required that xshift>>p
    factor = R*np.sqrt(epsfcn)
    x = x0-xshift
    ## perform optimisation. try block is for the case where failure
    ## to calculte error
    try:
        (x,cov_x,info,mesg,success)=optimize.leastsq(
            lambda x:func(x+xshift),
            x,
            epsfcn=epsfcn,
            factor=factor,
            full_output=True,
            maxfev=maxfev,
            xtol = xtol,
            )
    except ValueError as err:
        if str(err)=='array must not contain infs or NaNs':
            raise Exception('Bad covariance matrix in error calculation, residual independent of some variable?')
        else:
            raise
    ## check if any parameters have zero effect on the fit, and raise
    ## a warning if so. This will prevent the calculation of the
    ## uncertainty.
    if np.min(np.sum(np.abs(info['fjac']),1))==0: # a test for no effect
        ## calculate finite difference derivative
        reference_residual = np.array(func(x+xshift))
        diff = []
        for i,dxi in enumerate(dx):
            x1 = copy(x)
            x1[i] += dxi
            diff.append(np.max(np.abs(reference_residual-np.array(func(x1+xshift)))))
        diff = np.array(diff)
        ## warn about those that have no difference
        for j in find(diff==0):
            print(f'warning: Parameter has no effect: index={j}, value={float(x[j]+xshift[j])}')
    ## warn on error if requested
    if (not success) & print_error_mesg:
        warnings.warn("leastsq exit code: "+str(success)+mesg)
    ## sometimes this is not an array
    if not np.iterable(x): x=[x]
    ## attempt to calculate covariance of parameters
    if cov_x is None: 
        sigma_x = np.nan*np.ones(len(x))
    else:
        ## calculate noise rms from the resiudal if not explicitly provided
        if rms_noise is None:
            chisq=sum(info["fvec"]*info["fvec"])
            dof=len(info["fvec"])-len(x)+1        # degrees of freedom
            ## assumes unweighted data with experimental uncertainty
            ## deduced from fitted residual. ref gavin2011.
            std_y = np.sqrt(chisq/dof)
        else:
            std_y = rms_noise   # note that the degrees of freedom is not considered here
        sigma_x = np.sqrt(cov_x.diagonal())*std_y
    return(x+xshift,sigma_x)

def rms(x):
    """Calculate rms."""
    return np.sqrt(np.mean(np.array(x)**2))

def nanrms(x):
    """Calculate rms, ignoring NaN data."""
    return np.sqrt(np.nanmean(np.array(x)**2))

def randn(shape=None):
    """Return a unit standard deviation normally distributed random
    float, or array of given shape if provided."""
    if shape == None:
        return float(np.random.standard_normal((1)))
    else:
        return np.random.standard_normal(shape)


#####################
## array functions ##
#####################

def approximately_equal(x,y,abstol=None,fractol=1e-15):
    """The maximum difference between x and y is less than a given
    absolute tolerance or fractional tolerance."""
    if abstol is not None:
        d = np.max(np.abs(x-y))
        return d < abstol
    else:
        d = np.max(np.abs(x-y)/(x+y))*2
        return d < fractol

def convert_to_bool_vector_array(x):
    retval = array(x,ndmin=1)
    if retval.dtype.kind == 'b':
        return retval
    elif retval.dtype.kind == 'U':
        t = []
        for xi in retval:
            if xi=='True':
                t.append(True)
            elif xi=='False':
                t.append(False)
            elif xi=='Fixed':
                t.append(False)
            else:
                raise Exception("Valid boolean string values are 'True', 'False', and 'Fixed'")
        retval = array(t,ndmin=1,dtype=bool)
        return retval
    else:
        ## use numpy directly
        try:
            return np.asarray(x,dtype=bool,ndmin=1)
        except:
            return array([bool(t) for t in ensure_iterable(x)],dtype=bool)

    
###########################
## convenience functions ##
###########################

def uniquify_strings(strings):
    repeats = {}
    for s in strings:
        if s in repeats:
            repeats[s] +=1
        else:
            repeats[s] = 1
    retval = []
    counts = {}
    for s in strings:
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
        if repeats[s] == 1:
            retval.append(s)
        else:
            retval.append(s+'_'+str(counts[s]))
    return retval

def paginate_string(x,width=70):
    """Split string on words up to width."""
    x = x.replace('\n',' ')     # remove old line breaks
    xbreak = []
    while len(x)>width:
        i = x[:width].rfind(' ')
        assert i!=-1,'Not implemented no spaces within width'
        xbreak.append(x[:i])
        x = x[i+1:]
    xbreak.append(x)
    retval = '\n'.join(xbreak)
    return retval

def warnings_off():
    warnings.simplefilter("ignore")

def date_string():
    """Get string representing date in ISO format."""
    import datetime
    t = datetime.datetime.now()
    return('-'.join([str(t.year),format(t.month,'02d'),format(t.day,'02d')]))


def isiterable(x):
    """Test if x is iterable, False for strings."""
    if isinstance(x,str): 
        return False
    try:
        iter(x)
    except TypeError:
        return False
    return True

def indices(arr):
    """Generator return all combinations of indices for an array."""
    for i in itertools.product(
            *[range(n) for n in
              np.asarray(arr).shape]):
        yield i

########################
## file manipulations ##
########################

def expand_path(path):
    """Shortcut to os.path.expanduser(path). Returns a
    single file only, first in list of matching."""
    import os
    return os.path.expanduser(path)

def tmpfile():
    """Create a secure named temporary file which will be
    automatically deleted. Object is returned."""
    import tempfile
    return tempfile.NamedTemporaryFile()

def tmpdir():
    """Create a temporary directory which will not be
    automatically deleted. Pathname is returned."""
    import tempfile
    return tempfile.mkdtemp()

def trash(filename):
    """Put file in the trash can. Silence on error. No filename expansion."""
    import shlex
    os.system('trash-put '+shlex.quote(filename)+' > /dev/null 2>&1')

def mkdir(
        *directories,
        trash_existing=False,
        override_trash_existing_safety_check=False,):
    """Create directory tree (or multiple) if it doesn't exist."""
    if len(directories)>1:
        ## multiple directories then loop through them
        for directory in directories:
            mkdir(directory,
                  trash_existing=trash_existing,
                  override_trash_existing_safety_check=override_trash_existing_safety_check)
    else:
        ## expand file name
        directory = expand_path(directories[0])
        ## trash first if requested
        if trash_existing and os.path.isdir(directory):
            ## A very poor test to avoid destructive mistakes
            if directory in ('/','/home','.','./','../') and not override_trash_existing_safety_check:
                raise Exception(f'Will not trash directory {repr(directory)}. Set override_trash_existing_safety_check=True to proceed. ')
            trash(directory)
        ## do nothing if it already exists
        if os.path.isdir(directory):
            return
        ## walk parent directories making if necessary
        partial_directories = directory.split('/')
        for i in range(len(partial_directories)):
            partial_directory = '/'.join(partial_directories[0:i+1])
            if partial_directory=='' or os.path.isdir(partial_directory):
                continue
            else:
                if os.path.exists(partial_directory):
                    raise Exception("Exists and is not a directory: "+partial_directory)
                os.mkdir(partial_directory)

def trial_for_existing_path(
        paths,
        prefixes=(),
        suffixes=(),
        expanduser=False,
        raise_exception_on_fail=True, # else return None
):
    """Test which path in list of paths exists, return the first
    found. If prefixes and suffixes provided then trial all
    combinations. If expanduser=True also try version with expanded ~."""
    trialled_paths = []
    for prefix in prefixes:
        for suffix in suffixes:
            for path in paths:
                if expanduser:
                    trial_paths = [
                        prefix+path+suffix,
                        os.path.expanduser(prefix+path+suffix)]
                else:
                    trial_paths = [prefix+path+suffix]
                for trial_path in trial_paths:
                    trialled_paths.append(trial_path)
                    if os.path.exists(trial_path):
                        return trial_path
    if raise_exception_on_fail:
        paths = "\n".join(trialled_paths)
        raise Exception(f'Could not find existing path after trialling: {paths}')
    else:
        return None

#####################
## text formatting ##
#####################


def safe_eval_literal(string,string_on_error=False):
    """Evaluate a string into a python literal safely. If
    string_on_error=True then return the original string if it cannot be
    evaluated for any reason."""
    import ast
    try:
        retval = ast.literal_eval(string)
    except Exception as err:
        if string_on_error:
            raise Exception(f'Could not evaluate string: {repr(string):0.50s}')
        else:
            retval = string
    return retval

def regularise_symbol(
        x,
        permit_attr=False,       # permit attributes referenes, e.g., do not convert x.y to x_y
):
    """Turn an arbitrary string into a valid python symbol. INCOMPLETE
Check out
https://github.com/Ghostkeeper/Luna/blob/d69624cd0dd5648aec2139054fae4d45b634da7e/plugins/data/enumerated/enumerated_type.py#L91"""
    x = regularise_unicode(x)
    x = re.sub(r'[-<>(){}\[\]!^+|&/%]','_',x)
    if not permit_attr:
        x = re.sub(r'\.','_',x)
    if x[0] in '0123456789.':
        x = 'x'+x
    return x

def regularise_unicode(s):
    """Turn unicode symbols into something more ascii"""
    ## superscripts / subscripts 
    for x,y in ( ('⁰','0'), ('¹','1'), ('²','2'), ('³','3'),
                 ('⁴','4'), ('⁵','5'), ('⁶','6'), ('⁷','7'), ('⁸','8'),
                 ('⁹','9'), ('⁺','+'), ('⁻','-'), ('₀','0'), ('₁','1'),
                 ('₂','2'), ('₃','3'), ('₄','4'), ('₅','5'), ('₆','6'),
                 ('₇','7'), ('₈','8'), ('₉','9'), ):
        if x in s:
            s = s.replace(x,y)
    return s

def superscript_numerals(s):
    """Turn numerals and + and - into superscript versions."""
    if len(s)>1:
        s = ''.join([superscript_numerals(t) for t in s])
    ## superscripts / subscripts 
    for x,y in ( ('⁰','0'), ('¹','1'), ('²','2'), ('³','3'),
                 ('⁴','4'), ('⁵','5'), ('⁶','6'), ('⁷','7'), ('⁸','8'),
                 ('⁹','9'), ('⁺','+'), ('⁻','-'), ):
        if y in s:
            s = s.replace(y,x)
    return s

def subscript_numerals(s):
    """Turn numerals into subscript versions."""
    ## superscripts / subscripts 
    for x,y in (('₀','0'), ('₁','1'),
                 ('₂','2'), ('₃','3'), ('₄','4'), ('₅','5'), ('₆','6'),
                 ('₇','7'), ('₈','8'), ('₉','9'), ):
        if y in s:
            s = s.replace(y,x)
    return s



def align(string,input_delimiter_re=' +',output_delimiter=' '):
    """Realign string on input regexp with given output delimiter and
    padded with spaces. Blank lines removed."""
    ## decode if string given
    if isinstance(string,str):
        data = []
        for line in string.split('\n'):
            line = line.strip()
            if len(line) > 0:
                data.append(list(re.split(' *'+input_delimiter_re+' *',line)))
    ## get column widths
    column_widths = None
    for iline,line in enumerate(data):
        widths = [len(t) for t in line]
        if column_widths is not None and len(column_widths) != len(widths):
            raise Exception(f'Wrong number of columns on line {iline+1}.')
        if iline == 0:
            column_widths = widths
        else:
            column_widths = [max(t0,t1) for t0,t1 in zip(column_widths,widths)]
    ## format cells
    for line in data:
        for icell,(cell,width) in enumerate(zip(line,column_widths)):
            line[icell] = format(cell,f'<{width:d}')
    ## make aligned string
    string = '\n'.join([output_delimiter.join(line) for line in data])
    return string

# def decode_format_string(s):
    # """Get the different arts of a format string, return as dictionary."""
    # g = re.match(r'([<>+-]*)([0-9]*).?([0-9]*)([fgsed])',s).groups()
    # return(dict(prefix=g[0],length=g[1],precision=g[2],type=g[3]))


# def parentheses_style_errors_format(
        # x,s,
        # error_significant_figures=2,
        # tex=False,
        # treat_zero_as_nan=False,
        # default_sig_figs=3,     # if not error to indicate uncertainty use this many significant figures
        # max_leading_zeros=3,    # before use scientific notation
        # fmt='f',                # or 'e'
        # # nan_data_as_blank=False, # do not print nans, does something else instead
        # nan_substitute=None,
# ):
    # """
    # Convert a value and its error in to the form 1234.456(7) where the
    # parantheses digit is the error in the least significant figure
    # otherwise. If bigger than 1 gives 1200(400). If error_significant_figures>1
    # print more digits on the error.
    # \nIf tex=True make nicer still.
    # """
    # ## vectorise
    # if np.iterable(x):
        # return [
            # format_parentheses_style_errors(
                # xi,si,
                # error_significant_figures,
                # tex,
                # treat_zero_as_nan,
                # default_sig_figs,
                # max_leading_zeros,
                # fmt,
                # # nan_data_as_blank=nan_data_as_blank,
                # nan_substitute=nan_substitute,
            # ) for (xi,si) in zip(x,s)]
    # assert fmt in ('e','f'),'Only "e" and "f" formatting implemented.'
    # ## deal with zeros
    # if treat_zero_as_nan:
        # if x==0.: x=np.nan
        # if s==0.: s=np.nan
    # ## data is nan
    # if np.isnan(x):
        # if nan_substitute is None:
            # retval = 'nan(nan)'
        # else:
            # retval = nan_substitute
    # ## data exists but uncertainty is nan, just return with default_sig_figs
    # elif np.isnan(s):
        # if fmt=='f':
            # retval = format_float_with_sigfigs(x,default_sig_figs)
        # elif fmt=='e':
            # retval = format(x,f'0.{default_sig_figs-1}e')
    # ## data and uncertainty -- computed parenthetical error
    # else:
        # if 'f' in fmt:
            # ## format data string 'f'
            # ## format error string
            # t=format(s,'0.'+str(error_significant_figures-1)+'e') ## string rep in form +1.3e-11
            # precision = int(re.sub('.*e','',t))-error_significant_figures+1
            # s = t[0:1+error_significant_figures].replace('.','').replace('e','')
            # x=(round(x/10**precision)+0.1)*10**precision
            # x=format(x,('+0.30f' if fmt[0]=='+' else '0.30f'))
            # i=x.find('.')
            # if precision < 0:
                # x=x[:i-precision+1]
            # elif precision >0:
                # x=x[:i]
                # for j in range(precision): s=s+'0'
            # elif precision==0:
                # x=x[:i]
            # retval = x+'('+s+')'
        # elif 'e' in fmt:
            # ## format data string 'e'
            # r = re.match(r'^([+-]?[0-9]\.[0-9]+)e([+-][0-9]+)$',format(x,'0.50e'))
            # value_precision = int(r.group(2))
            # r = re.match(r'^([+-]?[0-9]\.[0-9]+)e([+-][0-9]+)$',format(s,'0.50e'))
            # error_precision = int(r.group(2))
            # error_string = r.group(1).replace('.','')[:error_significant_figures]
            # value_string = str(int(np.round(x*10**(-error_precision+error_significant_figures-1))))
            # ## add decimal dot
            # if value_string[0] in '+-':
                # if len(value_string)>2: value_string = value_string[0:2]+'.'+value_string[2:]
            # else:
                # if len(value_string)>1: value_string = value_string[0]+'.'+value_string[1:]
            # error_string = str(int(np.round(s*10**(-error_precision+error_significant_figures-1))))
            # retval = f'{value_string}({error_string})e{value_precision}'
        # else:
            # raise Exception('fmt must be "e" or "f"')
    # if tex:
        # ## separate thousands by \,, unless less than ten thousand
        # # if not re.match(r'[0-9]{4}[.(,]',retval): 
        # #     while re.match(r'[0-9]{4,}[.(,]',retval):
        # #         retval = re.sub(r'([0-9]+)([0-9]{3}[,.(].*)',r'\1,\2',retval)
        # #     retval = retval.replace(',','\,')
        # while True:
            # r = re.match(r'^([+-]?[0-9,]*[0-9])([0-9]{3}(?:$|[.(]).*)',retval)
            # if not r:
                # break
            # retval = r.group(1)+','+r.group(2)
        # retval = retval.replace(',','\\,')
        # ## separate decimal thousands by \,, unless less than ten thousandths
        # r = re.match(r'(.*\.)([0-9]{5,})(.*)',retval)
        # if r:
            # beg,decimal_digits,end = r.groups()
            # retval = beg+r'\,'.join([decimal_digits[i:i+3] for i in range(0,len(decimal_digits),3)])+end
        # ## replace nan with --
        # # retval = retval.replace('nan','--')
        # # retval = retval.replace('--(--)','--')
        # t = re.match(r'(.*)e([+-])([0-9]+)(.*)',retval)
        # if t:
            # beg,exp_sign,exp,end = t.groups()
            # exp = str(int(exp)) # get to single digit int
            # if exp==0:  exp_sign = '' # no e-0
            # if exp_sign=='+': exp_sign = '' # no e+10
            # retval = f'{beg}\\times10^{{{exp_sign}{exp}}}{end}'
        # # retval = re.sub(r'e([+-]?)0?([0-9]+)',r'\\times10^{\1\2}',retval)
        # # retval = re.sub(r'e([+-]?[0-9]+)',r'\\times10^{\1}',retval)
        # retval = '$'+retval.replace('--','-')+'$' # encompass
    # return(retval)
# format_parentheses_style_errors = parentheses_style_errors_format # deprecated
# parentheses_style_errors = parentheses_style_errors_format # deprecated name

# def parentheses_style_errors_decode(string):
    # """Convert string of format '##.##(##)' where parentheses contain
    # an uncertainty esitmate of the least significant digit of the
    # preceding value. Returns this value and an error as floats."""
    # ## if vector of strings given return an array of decoded values
    # if not np.isscalar(string):
        # return(np.array([parentheses_style_errors_decode(t) for t in string]))
    # ## 
    # m = re.match(r'([0-9.]+)\(([0-9]+)\)',str(string))
    # if m is None:
        # warnings.warn('Could not decode `'+str(string)+"'")
        # return np.nan,np.nan
    # valstr,errstr = m.groups()
    # tmpstr = list(re.sub('[0-9]','0',valstr,))
    # i = len(tmpstr)
    # for j in reversed(errstr):
        # i=i-1
        # while tmpstr[i]=='.': i=i-1
        # tmpstr[i] = j
    # return float(valstr),float(''.join(tmpstr))

# def format_tex_scientific(
        # x,
        # sigfigs=2,
        # include_math_environment=True,
        # nan_behaviour='--',     # None for error on NaN, else a replacement string
# ):
    # """Convert a string to scientific notation in tex code."""
    # if np.isnan(x):
        # if nan_behaviour is None:
            # raise Exception("NaN not handled.")
        # else:
            # return('--')
    # s = format(x,'0.'+str(sigfigs-1)+'e')
    # m = re.match(r'(-?[0-9.]+)[eEdD]\+?(-)?0*([0-9]+)',s)
    # digits,sign,exponent =  m.groups()
    # if sign is None:
        # sign=''
    # if include_math_environment:
        # delim = '$'
    # else:
        # delim = ''
    # return r'{delim}{digits}\times10^{{{sign}{exponent}}}{delim}'.format(
        # digits=digits,sign=sign,exponent=exponent,delim=delim)

# def format_numprint(x,fmt='0.5g',nan_behaviour='--'):
    # """Make an appropriate latex numprint formatting command. Fomra number
    # with fmt first. nan_behaviour # None for error on NaN, else a
    # replacement string """
    # if np.isnan(x):
        # if nan_behaviour is None:
            # raise Exception("NaN not handled.")
        # else:
            # return('--')
    # return(f'\\np{{{format(x,fmt)}}}')

# def format_float_with_sigfigs(
#         x,
#         sigfigs,
#         tex=False,
#         fmt='f',                # or 'e'
# ):
#     """Convert a float to a float format string with a certain number of
#     significant figures. This is different to numpy float formatting
#     which controls the number of decimal places."""
#     assert sigfigs>0
#     if tex:
#         thousands_separator = r'\,'
#     else:
#         thousands_separator = ''
#     ## get number of sigfigs rounded and printed into a string, special case for x=0
#     if x!=0:
#         x = float(x)
#         sign_x = np.sign(x)
#         x = np.abs(x)
#         x = float(x)
#         exponent = int(np.floor(np.log10(x)))
#         decimal_part = max(0,sigfigs-exponent-1)
#         s = format(sign_x*np.round(x/10**(exponent+1-sigfigs))*10**(exponent+1-sigfigs),'0.{0:d}f'.format(decimal_part))
#     else:
#         if sigfigs==1:
#             s = '0'
#         else:
#             s = '0.'+''.join(['0' for t in range(sigfigs-1)])
#     ## Split into >=1. <1 components
#     if s.count('.')==1:
#         greater,lesser = s.split('.')
#         sep = '.'
#     else:
#         greater,sep,lesser = s,'',''
#     ## add thousands separator, if number is bigger than 9999
#     if len(greater)>4:
#         indices_to_add_thousands_separator = list(range(len(greater)-3,0,-3))[-1::-1]
#         greater = thousands_separator.join([greater[a:b] for (a,b) in zip(
#             [0]+indices_to_add_thousands_separator,
#             indices_to_add_thousands_separator+[len(greater)])])
#     ## recomprise
#     s = greater+sep+lesser
#     return(s)

@vectorise()
def round_to_significant_figures(x,significant_figures):
    """Round x to a a limited number of significant decimal figures."""
    if x==0:
        retval = 0
    else:
        scale = 10**(significant_figures-np.floor(np.log10(np.abs(x)))-1)
        retval = np.round(x*scale)/scale
    return retval

# def format_fixed_width(x,sigfigs,width=None):
    # """Creates a exponential form floating point number with given
    # significant figures and total width. If this fails then return a
    # str form of the given width. Default width is possible."""
    # if width is None: 
        # width = sigfigs+6
    # try:
        # return format(x,'>+{}.{}e'.format(width,sigfigs-1))
    # except ValueError:
        # return format(x,'>'+str(width))
    
def format_string_or_general_numeric(x):
    """Return a string which is the input formatted 'g' if it is numeric or else is str representation."""
    try:
        return format(x,'g')
    except ValueError:
        return(str(x))

# def format_as_disjoint_ranges(x,separation=1,fmt='g'):
    # """Identify ranges and gaps in ranges and print as e.g, 1-5,10,12-13,20"""
    # x = np.sort(x)
    # i = [0]+list(find(np.diff(x)>separation)+1)+[len(x)] # range edges
    # ## Print each range as a-b if range is required or a if single
    # ## value is disjoin. Separate with commas.
    # return(','.join([
        # format(x[a],fmt)+'-'+format(x[b-1],fmt) if (b-a)>1 else format(x[a],fmt)
                    # for (a,b) in zip(
                            # [0]+list(find(np.diff(x)>separation)+1),
                            # list(find(np.diff(x)>separation)+1)+[len(x)])]))

def format_columns(
        data,                   # list or dict (for labels)
        fmt='>11.5g',
        labels=None,
        header=None,
        record_separator='\n',
        delimiter=' ',
        comment_string='# ',
):
    """Print args in with fixed column width. Labels are column
    titles.  NOT QUITE THERE YET"""
    ## if data is dict, reinterpret appropriately
    if hasattr(data,'keys'):
        labels = data.keys()
        data = [data[key] for key in data]
    ## make formats a list as long as data
    if isinstance(fmt,str):
        fmt = [fmt for t in data]
    ## get string formatting for labels and failed formatting
    fmt_functions = []
    for f in fmt:
        def fcn(val,f=f):
            if isinstance(val,str):
                ## default to a string of that correct length
                r = re.match(r'[^0-9]*([0-9]+)(\.[0-9]+)?[^0-9].*',f)
                return(format(val,'>'+r.groups()[0]+'s'))
            elif val is None:
                ## None -- print as None
                r = re.match(r'[^0-9]*([0-9]+)(\.[0-9]+)?[^0-9].*',f)
                return(format(repr(val),'>'+r.groups()[0]+'s'))
            else:
                ## return in given format if possible
                return(format(val,f)) 
        fmt_functions.append(fcn)
    ## begin output records
    records = []
    ## add header if required
    if header is not None:
        records.append(comment_string+header.strip().replace('\n','\n'+comment_string))
    ## labels if required
    if labels!=None:
        records.append(comment_string+delimiter.join([f(label) for (f,label) in zip(fmt_functions,labels)]))
    ## compose formatted data columns
    comment_pad = ''.join([' ' for t in comment_string])
    records.extend([comment_pad+delimiter.join([f(field) for (f,field) in zip(fmt_functions,record)]) for record in zip(*data)])
    t = record_separator.join(records)
    return(record_separator.join(records))

def printcols(*columns):
    """Print the data into readable columns heuristically."""
    print(format_columns(columns))


#####################################
## save / load /convert array data ##
#####################################

def find_inverse(index_array,total_shape):
    """Convert an integer index array into a boolean mask and also return
    sort_order to match index_array to boolean array."""
    bool_array = np.full(total_shape,False)
    sort_order = np.argsort(index_array)
    bool_array[index_array] = True
    return bool_array,sort_order

def cast_abs_float_array(x):
    """Return as 1D array of absolute floating point values."""
    return np.abs(np.asarray(x,dtype=float))

def repmat_vector(x,repeats=(),axis=-1):
    """x must be 1D. Expand to as many other dimension as length of
    repeats. Put x variability on axis. All other dimensions just
    repeat this one. """
    x = np.array(x)
    assert len(x.shape)==1,'1D only'
    retval = np.tile(x,list(repeats)+[1])
    if not (axis==-1 or axis==(retval.ndim-1)):
        return(np.moveaxis(np.tile(x,list(repeats)+[1]),-1,axis))
    return(retval)

def repmat(x,repeats):
    """Copy and expand matrix in various directions. Length of repeats
    must match dimension of x. If greater, then extra dimension are
    PREPENDED to x. Each (integer>0) element of repeat determines how
    many copies to make along that axis. """
    ## ensure types
    repeats = ensure_iterable(repeats)
    x = np.array(x)
    ## ensure all dimensions are included -- otherwise assume 1
    if len(repeats)<x.ndim:               #ensure long enough to match array ndim
        repeats.extend(np.ones(len(repeats-x.ndim)))
    ## ensure array has enough dimensions -- prepending empty dimensions
    if x.ndim<len(repeats):               
        x = np.reshape(x,[1 for t in range(len(repeats)-x.ndim)]+list(x.shape))
    ## for each non-unity repeat increase size of the array
    for axis,repeat in enumerate(repeats):
        if repeat==1: continue
        x = np.concatenate(tuple([x for t in range(repeat)]),axis=axis)
    return(x)

def file_to_array_unpack(*args,**kwargs):
    """Same as file_to_array but unpack data by default."""
    kwargs.setdefault('unpack',True)
    return file_to_array(*args,**kwargs)

def hdf5_to_array(filename,unpack=False,dataset=None,usecols=None):
    """Loads dataset dataset in hdf5 file into an array. If None then look
    for 'data', if its not there load first dataset. """
    import sys,h5py,os
    try:
        f = h5py.File(os.path.expanduser(filename),'r')
    except IOError:
        sys.stderr.write('Tried opening file: '+filename+'\n')
        raise
    if dataset is None:
        if 'data' in f:
            x = np.array(f['data']) # 'data'
        else:
            x = np.array(list(f.items())[0][1]) # first 
    else:
        x = np.array(f[dataset]) # dataset
    f.close()
    if usecols is not None:
        x = x[:,usecols]
    if unpack==True:
        x = x.transpose()
    return(x)

def hdf5_get_attributes(filename):
    """Get top level attributes."""
    import h5py
    with h5py.File(expand_path(filename_or_hdf5_object),'r') as fid:
        return {key:val for key,val in fid.attrs.items()}

def hdf5_to_numpy(value):
    if not np.isscalar(value):
        value = value[()]
    ## convert bytes string to unicode
    if np.isscalar(value):
        if isinstance(value,bytes):
            value = value.decode()
    else:
        ## this is a test for bytes string (kind='S') but for
        ## some reason sometimes (always?) loads as object
        ## type
        if value.dtype.kind in ('S','O'):
            # value = np.asarray(value,dtype=str)
            value = np.asarray([t.decode() for t in value],dtype=str)
    return value

def numpy_to_hdf5(value):
    ## deal with missing unicode type in hdft
    ## http://docs.h5py.org/en/stable/strings.html#what-about-numpy-s-u-type
    import h5py
    if not np.isscalar(value) and value.dtype.kind=='U':
        value = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))
    return value
    
def hdf5_to_dict(fid,load_attributes=True):
    """Load all elements in hdf5 into a dictionary. Groups define
    subdictionaries. Scalar data set as attributes."""
    import h5py
    ## open file if necessary
    if isinstance(fid,str):
        with h5py.File(expand_path(fid),'r') as fid2:
            return hdf5_to_dict(fid2,load_attributes=load_attributes)
    retval = {}            # the output data
    ## load attributes
    if load_attributes:
        for tkey,tval in fid.attrs.items():
            retval[str(tkey)] = hdf5_to_numpy(tval)
    ## load data and subdicts
    for key,val in fid.items():
        if isinstance(val,h5py.Dataset):
            retval[str(key)] = hdf5_to_numpy(val)
        else:
            retval[str(key)] = hdf5_to_dict(val,load_attributes=load_attributes)
    return retval

def dict_to_hdf5(fid,data,compress=False,verbose=True):
    """Save all elements of a dictionary as datasets, attributes, or
    subgropus in an hdf5 file."""
    import h5py
    if isinstance(fid,str):
        ## open file if necessary
        fid = expand_path(fid)
        mkdir(dirname(fid)) # make leading directories if not currently there
        with h5py.File(fid,mode='w') as new_fid:
            dict_to_hdf5(new_fid,data,compress,verbose)
            return
    ## add data
    for key,val in data.items():
        if isinstance(val,dict):
            ## recursively create groups
            group = fid.create_group(key)
            dict_to_hdf5(group,val,compress,verbose)
        else:
            if isinstance(val,np.ndarray):
                ## add arrays as a dataset
                if compress:
                    kwargs={'compression':"gzip",'compression_opts':9}
                else:
                    kwargs = {}
                fid.create_dataset(key,data=numpy_to_hdf5(val),**kwargs)
            else:
                ## add non-array data as attributes
                try:
                    fid.attrs.create(key,val)
                except TypeError as error:
                    if verbose:
                        raise error
                        print(error)
def dict_to_hdf5(fid,data,compress=False,verbose=True):
    """Save all elements of a dictionary as datasets, attributes, or
    subgropus in an hdf5 file."""
    import h5py
    if isinstance(fid,str):
        ## open file if necessary
        fid = expand_path(fid)
        mkdir(dirname(fid)) # make leading directories if not currently there
        with h5py.File(fid,mode='w') as new_fid:
            dict_to_hdf5(new_fid,data,compress,verbose)
            return
    ## add data
    for key,val in data.items():
        if isinstance(val,dict):
            ## recursively create groups
            group = fid.create_group(key)
            dict_to_hdf5(group,val,compress,verbose)
        else:
            if isinstance(val,np.ndarray):
                ## add arrays as a dataset
                if compress:
                    kwargs={'compression':"gzip",'compression_opts':9}
                else:
                    kwargs = {}
                fid.create_dataset(key,data=numpy_to_hdf5(val),**kwargs)
            else:
                ## add non-array data as attribute
                try:
                    fid.attrs.create(key,val)
                except TypeError as error:
                    if verbose:
                        raise error
                        print(error)

def append_to_hdf5(filename,**keys_vals):
    """Added key=val to hdf5 file."""
    import h5py
    with h5py.File(expand_path(filename),'a') as d:
        for key,val in keys_vals.items() :
            d[key] = val

def dict_to_recarray(d):
    """Convert a dictionary of identically sized arrays into a
    recarray. Names are dictionary keys. Add some dictionary-like
    methods to this particular instance of a recarray."""
    if len(d)==0:
        ra = np.recarray((0),float) # no data
    else:
        ra = np.rec.fromarrays([np.array(d[t]) for t in d], names=list(d.keys()),)
    return(ra)

def make_recarray(**kwargs):
    """kwargs are key=val pair defining arrays of equal length from
    which to make recarray."""
    ra = np.rec.fromarrays(kwargs.values(),names=list(kwargs.keys()))
    return(ra)

def dict_to_directory(
        directory,
        dictionary,
        array_format='npy',
        # remove_string_margin=True,
        trash_existing=True,
        override_trash_existing_safety_check=False,
        repr_strings=False,
):
    """Create a directory and save contents of dictionary into it."""
    ## make directory if necessary, possibly deleting old contents
    mkdir(
        directory,
        trash_existing=trash_existing,
        override_trash_existing_safety_check=override_trash_existing_safety_check)
    ## loop through all data
    for key,val in dictionary.items():
        str_key = str(key)
        if '/' in str_key:
            raise ImplementationError(f'Not implemented character for dict_to_directory [/] in key: {repr(key)}')
        if isinstance(val,dict):
            ## recursively save dictionaries as subdirectories
            dict_to_directory(
                f'{directory}/{str_key}',val,
                array_format,trash_existing=False,
                repr_strings=repr_strings)
        elif isinstance(val,str):
            ## save strings to text files
            if repr_strings:
                string_to_file(f'{directory}/{str_key}',repr(val))
            else:
                string_to_file(f'{directory}/{str_key}',val)
        elif isinstance(val,np.ndarray):
            ## save ndarray, defer formatting to numpy
            extension = get_extension(array_format)
            filename = f'{directory}/{str_key}{extension}'
            array_to_file(filename,np.asarray(val))
        else:
            ## save as repr of whatever it is
            string_to_file(f'{directory}/{str_key}',repr(val))

def directory_to_dict(
        directory,
        evaluate_strings=False,
):
    """Load all contents of a directory into a dictionary, recursive. If
    evaluate_strings=True then attempt to interpret these into python
    objects."""
    retval = {}
    directory = expand_path(directory)
    for filename in os.listdir(directory):
        full_filename = f'{directory}/{filename}'
        if os.path.isdir(full_filename):
            ## load subdirectories as subdictionaries
            retval[filename] = directory_to_dict(
                full_filename,evaluate_strings=evaluate_strings)
        else:
            root,extension = os.path.splitext(filename)
            if extension in ('.npz','.npy','.h5','.hdf5'):
                ## load binary array data
                retval[root] = file_to_array(full_filename)
            else:
                ## load as string
                retval[filename] = file_to_string(full_filename)
                if evaluate_strings:
                    retval[filename] = safe_eval_literal(retval[filename],string_on_error=False)
    return retval 

########################
## system interaction ##
########################

def get_memory_usage():
    import gc
    import os, psutil
    gc.collect()
    process = psutil.Process(os.getpid())
    before = 0
    memory_usage = process.memory_info().rss
    return memory_usage

def print_memory_usage():
    memory_usage = get_memory_usage()
    print(f'memory usage (B): {memory_usage:0.3e}')

_memory_usage_cache = {}
def memory_usage_start():
    usage = get_memory_usage()
    print(f'DEBUG: initial memory usage (B): {usage:0.3e}')
    _memory_usage_cache['initial'] = usage

def memory_usage_stop():
    usage = get_memory_usage()
    change = usage - _memory_usage_cache['initial']
    print(f'DEBUG: final memory usage (B): total={usage:0.3e} change={change:0.3e}')
    _memory_usage_cache['initial'] = usage
    
def pause(message="Press any key to continue..."):
    """Wait for use to press enter. Not usable outsdie linux."""
    input(message)

def get_clipboard():
    """Get a string from clipboard."""
    import subprocess
    status,output = subprocess.getstatusoutput("xsel --output --clipboard")
    assert status==0, 'error getting clipboard: '+output
    return output

def set_clipboard(string,target='clipboard'):
    """Send a string to clipboard."""
    if target == 'clipboard':
        pipe=os.popen(r'xsel --input --clipboard','w')
    elif target == 'primary':
        pipe=os.popen(r'xsel --input --primary','w')
    else:
        raise Exception(f"bad target: {target}")
    pipe.write(string)
    pipe.close()

def cl(x,fmt='0.15g'):
    """Take array or scalar x and convert to string and put on clipboard."""
    if np.isscalar(x):
        if isinstance(x,str):
            set_clipboard(x)
        else:
            set_clipboard(format(x,fmt))
    else:
        set_clipboard(array_to_string(x,fmt=fmt))

def pa():
    """Get string from clipboard. If possible convert to an array."""
    x = get_clipboard()
    try:
        return string_to_array(x)
    except:
        return x

def glob(path='*',regexp=None):
    """Shortcut to glob.glob(os.path.expanduser(path)). Returns a list
    of matching paths. Also sed alphabetically. If re is provided filter names accordingly."""
    import glob as glob_module
    retval = sorted(glob_module.glob(os.path.expanduser(path)))
    if regexp is not None:
        retval = [t for t in retval if re.match(regexp,t)]
    return(retval)

def glob_unique(path):
    """Match glob and return as one file. If zero or more than one file
    matched raise an error."""
    import glob as glob_module
    filenames = glob_module.glob(os.path.expanduser(path))
    if len(filenames)==1:
        return(filenames[0])
    elif len(filenames)==0:
        raise FileNotFoundError('No files matched path: '+repr(path))
    elif len(filenames)>1:
        raise Exception('Multiple files matched path: '+repr(path))

def basename(path):
    """Remove all leading directories. If the path is a directory strip
    final '/'."""
    if path[-1]=='/':
        return(basename(path[:-1]))
    else:
        return(os.path.basename(path))

def dirname(path):
    """Return full directory prefix."""
    try:
        i = path[-1::-1].index('/')
        return(path[0:len(path)-i])
    except ValueError:
        return('./')

def polyfit(
        x,y,
        dy=None,
        order=0,
        fixed=None,
        do_print=False,
        do_plot=False,
        error_on_missing_dy=True,
        # plot_kwargs=None,
):
    """
    Polyfit with weights calculated from provided standard
    deviation. Will ignore data with NaNs in any of x, y, or dy. If
    dy=None, or a dy is a constant, or dy is all 0., then a constant
    value (default 1) will be used. If some dy is 0, then these will
    be set to NaN and ignored.
    \nInputs:\n
    x - independent variables
    y - dependent variable
    dy - standard error of y
    order - order of polynomial to fit
    fixed - parameter to not vary, fixed values in dict
            indexed by order, e.g. {0:100,} fixes constant term
    extended_output - output more, default is False
    print_output - print some data, default is False
    plot_output - issue plotting commands, default is False
    return_style - 'list' or 'dict'.
    plotkwargs - a dictionary of kwargs passed to plot
    \nOutputs:\n
    p - the polynomial coefficients
    If extended_output=True then also returns:
    dp - standard error in p, will only be accurate if order is correct
    f - a function representing this polynomial
    residuals - of fit
    chisqprob - probability of arriving at these residuals 
                (or greater ones given the standard errors dy with the
                proposed polynomial model.
    """
    import scipy
    x,y = np.array(x),np.array(y) # ensure types
    if dy is None: dy = np.full(x.shape,1.,dtype=float) # default uncertainty if Noneprovided
    if type(dy) in [float,int,np.float64,np.int64]: dy = dy*np.ones(y.shape) # vectorise dy if constant given
    dy = np.array(dy)           # ensure array
    if error_on_missing_dy and (np.any(np.isnan(dy)) or np.any(dy==0)): raise Exception("Incomplete dy data, zero or nan.") # raise error bad dy 
    xin,yin,dyin = copy(x),copy(y),copy(dy) # save original data
    ## deal with nan or zero data by not fitting to them. If all zero or nan then set error to 1.
    dy[dy==0] = np.nan
    i = ~(np.isnan(x)|np.isnan(y)|np.isnan(dy))
    if np.any(i):
        x,y,dy = x[i],y[i],dy[i] # reduce data to known uncertianteis
    else:
        dy = np.full(dy.shape,1.) # set all uncertainties to 1 if none are known
        i = np.full(dy.shape,True)
    ## reduce polynomial order to match data length
    order = min(order,len(x)-1)
    ## solve linear least squares equation, do not include fixed parameters in matrix
    W = np.diag(1/dy)
    if fixed is None:
        At = np.array([x**n for n in range(order+1)])
        A = At.transpose()
        AtW = np.dot(At,W)
        AtWA = np.dot(AtW,A)
        invAtWA = scipy.linalg.inv(AtWA)
        invAtWAdotAtW  = np.dot(invAtWA,AtW)
        p = np.dot(invAtWAdotAtW,y)
    else:
        y_reduced = copy(y)
        At = []
        for n in range(order+1):
            if n in fixed:  y_reduced = y_reduced - fixed[n]*x**n
            else:           At.append(x**n)
        At = np.array([t for t in At])
        A = At.transpose()
        AtW = np.dot(At,W)
        AtWA = np.dot(AtW,A)
        invAtWA = scipy.linalg.inv(AtWA)
        invAtWAdotAtW  = np.dot(invAtWA,AtW)
        p_reduced = list(np.dot(invAtWAdotAtW,y_reduced))
        p = []
        for n in range(order+1):
            if n in fixed:
                p.append(fixed[n])
            else: 
                p.append(p_reduced.pop(0))
    p = np.flipud(p) # to conform to polyfit convention
    ## function
    f = lambda x: np.polyval(p,x)
    ## fitted values
    yf = f(xin)
    ## residuals
    residuals = yin-yf
    ## chi-square probability
    if dy is None:
        chisq = chisqnorm = chisqprob = None
    else:
        chisq = (residuals[i]**2/dy**2).sum()
        chisqnorm = chisq/(len(x)-order-1-1)
        # chisqprob = scipy.stats.chisqprob(chisq,len(x)-order-1-1)
        chisqprob = scipy.stats.distributions.chi2.sf(chisq,len(x)-order-1-1)
    ## stdandard error paramters (IGNORING CORRELATION!)
    if dy is None:
        dp = None
    else:
        dp = np.sqrt(np.dot(invAtWAdotAtW**2,dy**2))
        dp = np.flipud(dp) # to conform to polyfit convention
    ## a nice summary message
    if do_print:
        print(('\n'.join(
            ['             p             dp']
            +[format(a,'14.7e')+' '+format(b,'14.7e') for (a,b) in zip(p,dp)]
            +['chisq: '+str(chisq),'chisqprob: '+str(chisqprob),
              'rms: '+str(rms(residuals)),
              'max_abs_residual: '+str(abs(residuals).max())]
            )))
    ## a nice plot
    if do_plot:
        fig=plotting.plt.gcf()
        fig.clf()
        ax = subplot(0)
        ax.errorbar(x,y,dy,label='data') 
        ax.errorbar(x,yf,label='fit') 
        plotting.legend()
        ax = subplot(1)
        ax.errorbar(x,residuals,dy,label='residual error')
        plotting.legend()
    ## return 
    return dict(
        x=xin,y=yin,dy=dyin,
        p=p,dp=dp,
        yf=f(xin),f=f,
        residuals=residuals,
        chisqprob=chisqprob,chisq=chisq,chisqnorm=chisqnorm,
        fixed=fixed)

def ensure_iterable(x):
    """If input is not iterable enclose it as a list."""
    if np.isscalar(x): 
        return (x,)
    else: 
        return x

def spline(
        xs,ys,x,s=0,order=3,
        sort_data= True,
        remove_nan_data= True,
        out_of_bounds='extrapolate', # 'extrapolate','zero','error', 'constant'
):
    """Evaluate spline interpolation of (xs,ys) at x. Optional argument s
    is spline tension. Order is degree of spline. Silently defaults to 2 or 1
    if only 3 or 2 data points given.
    """
    ## prepare data
    xs,ys,x = np.array(xs,ndmin=1,dtype=float),np.array(ys,ndmin=1,dtype=float),np.array(x,ndmin=1,dtype=float)
    i = np.isnan(xs)|np.isnan(ys)
    if any(i):
        if remove_nan_data:
            xs,ys = xs[~i],ys[~i]
        else:
            raise Exception('NaN data present')
    if sort_data:
        xs,i = np.unique(xs,return_index=True)
        ys = ys[i]
    elif np.any(np.diff(xs)<=0):
        raise Exception('xspline is not monotonically increasing')
    ## interpolate
    order = min(order,len(xs)-1)
    if order == 0:
        ## piecewise constant interpolation
        y = np.empty(len(x),dtype=float)
        ## find indices of endpoints and  midpoints
        ixs = np.concatenate(([0], np.searchsorted(x, (xs[1:]+xs[:-1])/2), [len(x)],))
        for ia,ib,yi in zip(ixs[:-1],ixs[1:],ys):
            y[ia:ib] = yi
    else:
        ## actual spline
        from scipy import interpolate
        y = interpolate.UnivariateSpline(xs, ys, k=order, s=s)(x)
    ## out of bounds logic
    iout = (x<xs[0])|(x>xs[-1])
    if np.any(iout):
        if out_of_bounds == 'error':
            raise Exception('Data to spline is out of bounds')
        elif out_of_bounds == 'zero':
            y[iout] = 0.0
        elif out_of_bounds == 'constant':
            ## Regions above and below spline domain fixed to extreme
            ## spline points.
            raise NotImplementedError
        elif out_of_bounds == 'extrapolate':
            pass
        else:
            raise Exception(f'Invalid {out_of_bounds=}. Try "error", "zero", "constant", "extrapolate".')
    return y

def spline_from_list(spline_points,x,**spline_kwargs):
    """Rather than provide x and y spline points provide a list of pairs:
    [[x0,y0],[x1,y1],...]."""
    xspline,yspline = zip(*spline_points)
    return spline(xspline,yspline,x,**spline_kwargs)

def array_to_hdf5(
        filename,
        *columns,
        description=None,       # attribute
        **create_dataset_kwargs,
):
    """Column stack arrays in args and save in an hdf5 file. In a
    single data set named 'data'. Overwrites existing files."""
    import h5py
    filename = expand_path(filename)
    ## kwargs.setdefault('compression',"gzip") # slow
    ## kwargs.setdefault('compression_opts',9) # slow
    if os.path.exists(filename):
        assert not os.path.isdir(filename),'Will not overwrite directory: '+filename
        os.unlink(filename)
    with h5py.File(filename,'w') as fid:
        ## add data
        fid.create_dataset('data',data=np.column_stack(columns),**create_dataset_kwargs)
        ## add description as an attribute of root 
        if description is not None:
            fid.attrs.create('description',description)

def solve_linear_least_squares_symbolic_equations(
        system_of_equations,
        plot_residual=False,
):
    """Solve an overspecified system of linear equations. This is encoded pretry strictly e.g.,:
    1*x +  2*y =  4
    1*x +  3*y =  8
    0*x + -1*y = -3
    Important separators are: newline, =, + , *.
    """
    ## get system of equations
    equations = []
    for t in system_of_equations.split('\n'):
        t = t.split('#')[0]            # eliminate trailling comments
        if len(t.strip())==0: continue # blank line
        equations.append(t)
    ## decode into terms
    Aij,bi,variables = [],[],[]
    for i,equation in enumerate(equations):
        lhs,rhs = equation.split('=')
        for term in lhs.split('+'):
            coeff,var = term.split('*')
            coeff = float(coeff)
            var = var.strip()
            if var not in variables: variables.append(var)
            Aij.append((i,variables.index(var),coeff))
        bi.append(float(rhs))
    ## form matrices
    A = np.zeros((len(equations),len(variables)))
    for i,j,x in Aij: A[i,j] = x
    b = np.array(bi)
    ## solve. If homogeneous assume first variable==1
    homogeneous = True if np.all(b==0) else False
    if homogeneous:
        b = -A[:,0].squeeze()
        A = A[:,1:]
    x = np.dot( np.linalg.inv(np.dot(np.transpose(A),A)),   np.dot(np.transpose(A),b))
    if homogeneous:
        x = np.concatenate(([1],x))
        A = np.column_stack((-b,A))
        b = np.zeros(len(equations))
    if plot_residual:
        fig = plt.gcf()
        fig.clf()
        ax = fig.gca()
        ax.plot(b-np.dot(A,x),marker='o')
    ## return solution dictionary
    return({var:val for var,val in zip(variables,x)})

def weighted_mean(
        x,
        dx,
        error_on_nan_zero=True, # otherwise edit out affected values
):
    """Calculate weighted mean and its variance. If
    error_on_nan_zero=False then remove data with NaN x or dx, or 0
    dx."""
    # ## if ufloat, separate into nominal and error parts -- return as ufloat
    # if isinstance(x[0],uncertainties.AffineScalarFunc):
        # (mean,variance) = weighted_mean(*decompose_ufloat_array(x))
        # return(ufloat(mean,variance))
    x,dx = np.array(x,dtype=float),np.array(dx,dtype=float) 
    ## trim data to non nan if these are to be neglected, or raise an error
    i = np.isnan(x)
    if np.any(i):
        if error_on_nan_zero:
            raise Exception('NaN values present.') 
        else:
            x,dx = x[~i],dx[~i]
    i = np.isnan(dx)
    if np.any(i):
        if error_on_nan_zero:
            raise Exception('NaN errors present.') 
        else:
            x,dx = x[~i],dx[~i]
    i = (dx==0)
    if np.any(i):
        if error_on_nan_zero:
            raise Exception('NaN errors present.') 
        else:
            x,dx = x[~i],dx[~i]
    ## make weighed mean
    weights = dx**-2           # assuming dx is variance of normal pdf
    weights = weights/sum(weights) # normalise
    mean = np.sum(x*weights)
    variance = np.sqrt(np.sum((dx*weights)**2))
    return (mean,variance)

def cumtrapz(y,
             x=None,               # if None assume unit xstep
             reverse=False,        # or backwards
):
    """Cumulative integral, with first point equal to zero, same length as
    input."""
    if reverse:
        y = y[::-1]
        if x is not None: 
            x = x[::-1]
    from scipy import integrate
    yintegrated = np.concatenate(([0],integrate.cumtrapz(y,x)))
    if reverse:
        yintegrated = -yintegrated[::-1] # minus sign to account for change in size of dx when going backwards, which is probably not intended
    return yintegrated 

def power_spectrum(
        x,y,
        make_plot=False,
        fit_radius=1,
        return_peaks=False,
        **find_peaks_kwargs
):
    """Return (frequency,power) after spectral analysis of y. Must be on a
    uniform x grid."""
    dx = np.diff(x)
    assert np.abs(dx.max()/dx.min()-1)<1e-5,'Uniform grid required.'
    dx = dx[0]
    F = np.fft.fft(y)          # Fourier transform
    F = np.real(F*np.conj(F))         # power spectrum
    F = F[:int((len(F-1))/2+1)] # keep up to Nyquist frequency
    f = np.linspace(0,1/dx/2.,len(F)+1)[:-1] # frequency scale
    if make_plot:
        fig = plotting.gcf()
        fig.clf()
        ax = fig.gca()
        ax.cla()
        ax.plot(f,F,color=newcolor(0))
        ax.set_xlabel('f')
        ax.set_ylabel('F')
        ax.set_yscale('log')
    if return_peaks:
        from . import dataset
        from . import lineshapes
        resonances = dataset.Dataset()
        find_peaks_kwargs.setdefault('min_peak',0.9)
        find_peaks_kwargs.setdefault('x_minimimum_separation',fit_radius)
        ipeaks = find_peaks(F,f,**find_peaks_kwargs)
        if make_plot:
            ax.plot(f[ipeaks],F[ipeaks],marker='o',ls='',color=newcolor(1))
        return f,F,ipeaks
        # print(f'{len(ipeaks)} peaks found')
        # for ipeak in ipeaks:
            # i = slice(max(0,ipeak-int(fit_radius/dx)),min(ipeak+int(fit_radius/dx)+1,len(f)))
            # fit = lineshapes.fit_lorentzian(f[i],F[i],x0=f[ipeak],S=F[ipeak],Γ=dx)
            # pprint(fit)
            # resonances.append(f0=fit['p']['x0'], λ0=1/fit['p']['x0'], S=fit['p']['S'],Γ=fit['p']['Γ'])
            # print( resonances)
            # if make_plot:
                # ax.plot(fit['x'],fit['yf'],color=newcolor(2))
        # return f,F,resonances
    else:
        return(f,F)

def argmedian(x):
    """Return index of median point."""
    isorted = np.argsort(x)
    imedian = isorted[int(len(isorted)/2)]
    return imedian

def unique(x,preserve_ordering=False):
    """Returns unique elements. preserve_ordering is likely slower"""
    if preserve_ordering:
        x = list(x)
        for t in copy(x):
            while x.count(t)>1:
                x.remove(t)
        return(x)
    else:
        return(list(set(x)))

def unique_combinations(*args):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    return(set(zip(*args)))

def unique_combinations_masks(*arrs):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    ## pack into recarray, find unique hashes (cannot figure out how
    ## to hash recarray rows).
    ra = np.rec.fromarrays(arrs)
    hashes = np.array([hash(t) for t in zip(*arrs)],dtype=int)
    retval = []
    for unique_hash,ifirst in zip(*np.unique(hashes,return_index=True)):
        i = hashes == unique_hash
        retval.append((ra[ifirst],i))
    return retval

def unique_combinations_first_index(*arrs):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    ra = np.rec.fromarrays(arrs)
    unique_values,first_index = np.unique(ra,return_index=True)
    return unique_values,first_index

def sortall(x,*others,reverse=False):
    """Sort x and return sorted. Also return others sorted according
    to x."""
    x = np.asarray(x)
    i = np.argsort(x)
    retval = [x[i]]
    for y in others:
        retval.append(np.asarray(y)[i])
    if reverse:
        retval = [t[::-1] for t in retval]
    return retval

def common(x,y,use_hash=False):
    """Return indices of common elements in x and y listed in the order
    they appear in x. Raises exception if repeating multiple matches
    found."""
    if not use_hash:
        ix,iy = [],[]
        for ixi,xi in enumerate(x):
            iyi = find([xi==t for t in y])
            if len(iyi)==1: 
                ix.append(ixi)
                iy.append(iyi[0])
            elif len(iyi)==0:
                continue
            else:
                raise Exception('Repeated value in y for: '+repr(xi))
        if len(np.unique(iy))!=len(iy):
            raise Exception('Repeated value in x for something.')
        return(np.array(ix),np.array(iy))
    else:
        xhash = np.array([hash(t) for t in x])
        yhash = np.array([hash(t) for t in y])
        ## get sorted hashes, checking for uniqueness
        xhash,ixhash = np.unique(xhash,return_index=True)
        assert len(xhash)==len(x),f'Non-unique values in x.'
        yhash,iyhash = np.unique(yhash,return_index=True)
        assert len(yhash)==len(y),f'Non-unique values in y.'
        ## use np.searchsorted to find one set of hashes in the other
        iy = np.arange(len(yhash))
        ix = np.searchsorted(xhash,yhash)
        ## remove y beyond max of x
        i = ix<len(xhash)
        ix,iy = ix[i],iy[i]
        ## requires removing hashes that have no search sorted partner
        i = yhash[iy]==xhash[ix]
        ix,iy = ix[i],iy[i]
        ## undo the effect of the sorting above
        ix,iy = ixhash[ix],iyhash[iy]
        ## sort by index of first array -- otherwise sorting seems to be arbitrary
        i = np.argsort(ix)
        ix,iy = ix[i],iy[i]
        return(ix,iy)

def isin(x,y):
    """Return arrays of booleans same size as x, True for all those
    elements that exist in y."""
    return np.array([i in y for i in x],dtype=bool,ndmin=1)

def find(x):
    """Convert boolean array to array of True indices."""
    return(np.where(x)[0])

def find_unique(x):
    """Convert boolean array to array of True indices."""
    i = find(x)
    assert len(i)>0,'No match found'
    assert len(i)<2,'Multiple matches found'
    return i[0]

def findin_unique(x,y):
    """Find one only match of x in y. Else raise an error."""
    i = findin(x,y)
    assert len(i)!=1,'No match found'
    assert len(i)<2,'Multiple matches found'
    return i[0]

def findin(x,y):
    """Find x in y and return a list of the matching y indices. If an
    element of x cannot be found in y, or if multiple found, an error
    is raised."""
    x = ensure_iterable(x)
    y = ensure_iterable(y)
    i = np.zeros(len(x),dtype='int')
    for j,xj in enumerate(x):
        ii = find(y==xj)
        if len(ii) != 1:
            if len(ii) == 0:
                raise Exception(f'Element not found in y: {repr(xj)}')
            if len(ii) > 1:
                raise Exception(f'Element non-unique in y: {repr(xj)}')
        i[j] = ii[0]
    return i

def find_nearest(x,y):
    """Find nearest match of x in y and return a list of the nearest-match
    y indices. If multiple x match the same y an error is raised."""
    y = np.asarray(y)
    i = np.array([np.argmin(abs(y-xi)) for xi in ensure_iterable(x)])
    assert len(i) == len(np.unique(i)),'Multiple values in x nearest-match the same value in y.'
    return i

def findin_numeric(x,y,tolerance=1e-10):
    """Use compiled code to findin with numeric data only."""
    ix,iy = np.argsort(x),np.argsort(y) # sort data
    tx,ty = np.asarray(x,dtype=float)[ix],np.asarray(y,dtype=float)[iy]
    i = np.full(tx.shape,-1,dtype=int)
    myf.findin_sorted(tx,ty,tolerance,i)
    if i[0]==-1: raise Exception('Some value of x not found in y within tolerance.') # hack of an error code
    i = i[np.argsort(ix)]                 # undo x sort
    i = np.argsort(iy)[i]                 # undo y sort
    return(i)

def integrate(x,y,method='trapz'):
    """Integrate x vs. y"""
    import scipy
    if method == 'trapz':
        retval = scipy.integrate.trapz(y,x)
    elif method == 'simps':
        retval = scipy.integrate.simps(y,x)
    elif method == 'bin':
        dx = np.diff(x)/2
        width = np.full(len(x),0.0)
        width[1:] += dx
        width[:-1] += dx
        retval = np.sum(y*width)
    else:
        raise Exception(f'Unknown integration method: {repr(method)}')
    return retval

def find_blocks(b,error_on_empty_block=True):
    """Find boolean index arrays that divide boolean array b into
    independent True blocks."""
    i = np.full(len(b),True)    # not yet in a block
    blocks = []                 # final list of blocks
    while np.any(i):            # until all rows in a block
        ## start new block with first remaining row
        block = b[find(i)[0],:]
        if np.sum(block) == 0:
            ## a block is all zero -- connected to nothing
            if error_on_empty_block:
                ## raise an error
                raise Exception('empty block found')
            else:
                ## return as one connected block
                blocks.append(block|True)
                break
        ## add coupled elements to this block until no new ones found
        while np.any((t:=np.any(b[block,:],0)) & ~block):
            block |= t
            t = np.any(b[block,:],0)
        ## record found block and blocked rows
        blocks.append(block)
        i &= ~block
    return blocks

def full_range(x):
    return np.max(x)-np.min(x)

def inrange(
        x,
        xbeg,xend,
        include_adjacent=False,
        return_as='bool',       # 'bool','slice','find'
):
    """Return slice of sortd vector x between xbeg and xend. If
    include_adjacent then neighbouring pionts included in slice. Array
    x must be sorted."""
    i = np.searchsorted(x,xbeg,side='left')
    j = np.searchsorted(x,xend,side='right')
    if include_adjacent:
        if i>0:
            i -= 1
        if j<len(x):
            j += 1
    if return_as == 'slice':
        retval = slice(i,j)
    elif return_as == 'bool':
        retval = np.full(len(x),False)
        retval[i:j] = True
    elif return_as == 'find':
        retval = np.arange(i,j)
    else:
        raise Exception("Invalid return_as, shoule be one of 'bool', 'slice', or 'find'.")
    return retval

def limit_to_range(beg,end,x,*other_arrays):
    """Limit x to range between beg and end (using np.searchsorted, must
    be sorted.  Also index other_arrays and return all arrays."""
    i = np.searchsorted(x,(beg,end))
    return tuple([t[i[0]:i[1]] for t in [x]+list(other_arrays)])

def total_fractional_range(x):
    """Compute  total fractional range of x."""
    xmax,xmin = np.max(x),np.min(x)
    if xmax == 0 and xmin == 0:
        total_fractional_range = 0
    else:
        total_fractional_range = abs((xmax-xmin)/((xmax+xmin)/2))
    return total_fractional_range

def match_regexp(regexp,x):
    """Returns boolean array of elements of x whether or not they match
    regexp."""
    return np.array([bool(re.match(regexp,t)) for t in x])

def find_regexp(regexp,x):
    return find(match_regexp(regexp,x))

def match_lines(string,regexp):
    """"""
    r = re.compile(regexp)
    retval = []
    for line in string.split('\n'):
        if re.match(r,line):
            retval.append(line)
    retval = '\n'.join(retval)
    return retval

def meshgrid(*args):
    """ meshgrid(arr1,arr2,arr3,...)
    Expand 1D arrays arr1,... into multiple multidimensional arrays
    that loop over the values of arr1, ....  Similar to matlab/octave
    meshgrid. Sorry about the poor documentation, an example:
    meshgrid(np.array([1,2]),np.array([3,4]),)
    returns
    (array([[1, 1],[2, 2]]), array([[3, 4],[3, 4]]))
    """
    ## a sufficiently confusing bit of code its probably easier to
    ## rewrite than figure out how it works
    n = len(args)
    assert n>=2, 'requires at least two arrays'
    l = [len(arg) for arg in args]
    ret = []
    for i in range(n):
        x = np.array(args[i])
        for j in range(n):
            if i==j: continue
            x = np.expand_dims(x,j).repeat(l[j],j)
        ret.append(x)
    return tuple(ret)


def normal_distribution(x,μ=0.,σ=1):
    """Normal distribution."""
    from scipy import constants
    return(1/np.sqrt(constants.pi*σ**2)*np.exp(-(x-μ)**2/(2*σ**2)))

def fit_normal_distribution(x,bins=None,figure=None):
    """Fit a normal distribution in log space to a polynomial."""
    if bins is None:
        ## estimate good amount of binning
        bins = max(10,int(len(x)/200))
    count,edges = np.histogram(x,bins)
    centres = (edges[:-1]+edges[1:])/2
    logcount = np.log(count)
    i = ~np.isinf(logcount)
    p = np.polyfit(centres[i],logcount[i],2)
    σ = np.sqrt(-1/p[0]/2)
    μ = p[1]*σ**2
    if figure is not None:
        ## make a plot
        figure.clf()
        ax = plotting.subplot(0,fig=figure)
        ax.plot(centres,logcount)
        ax.plot(centres,np.polyval(p,centres))
        ax = plotting.subplot(1,fig=figure)
        ax.set_title("Fit in log space")
        ax.plot(centres,count)
        nd = normal_distribution(centres,μ,σ)
        ax.plot(centres,nd/np.mean(nd)*np.mean(count))
        ax.set_title("Fit in linear space")
    return μ,σ

def gaussian(x,fwhm=1.,mean=0.,norm='area'):
    """
    y = gaussian(x[,fwhm,mean]). 
    Produces a gaussian with area normalised to one.
    If norm='peak' peak is equal to 1.
    If norm='sum' sums to 1.
    Default fwhm = 1. Default mean = 0.
    """
    fwhm,mean = float(fwhm),float(mean)
    if norm=='area':
        ## return 1/fwhm*np.sqrt(4*np.log(2)/constants.pi)*np.exp(-(x-mean)**2*4*np.log(2)/fwhm**2);
        return 1/fwhm*0.9394372786996513*np.exp(-(x-mean)**2*2.772588722239781/fwhm**2);
    elif norm=='peak':
        return np.exp(-(x-mean)**2*4*np.log(2)/fwhm**2)
    elif norm=='sum':
        t = np.exp(-(x-mean)**2*4*np.log(2)/fwhm**2)
        return t/t.sum()
    else:
        raise Exception('normalisation method '+norm+' not known')

def convolve_with_padding(x,y,xconv,yconv):
    """Convolve function (x,y) with (xconv,yconv) returning length of (x,y)."""
    import scipy
    if len(xconv) == 0:
        raise Exception('Length of convolving array is zero.')
    elif len(xconv) == 1:
        xpad = x
        ypad = y
        npadding = 0
    else:
        dxorig = (x[-1]-x[0])/(len(x)-1)
        width = xconv[-1]-xconv[0]
        dx = width/(len(xconv)-1)
        if abs(dx-dxorig)/dx > 1e-5:
            raise Exception("Grid steps do not match")
        padding = arange(dx,width+dx,dx)
        npadding = len(padding)
        xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
        ypad = np.concatenate((np.full(padding.shape,y[0]),y,np.full(padding.shape,y[-1])))
    yout = scipy.signal.oaconvolve(ypad,yconv,mode='same')[npadding:npadding+len(x)]
    return yout

def convolve_with_gaussian(
        x,y,
        fwhm,
        fwhms_to_include=10,
        regrid_if_necessary=False,
):
    """Convolve function y(x) with a gaussian of FWHM fwhm. Truncate
    convolution after a certain number of fwhms. x must be on a
    regular grid."""
    dx = (x[-1]-x[0])/(len(x)-1)
    ## check on regular grid, if not then spline to a new one
    t = np.diff(x)
    regridded = False
    if (t.max()-t.min())>dx/100.:
        if regrid_if_necessary:
            regridded = True
            x_original = x
            xstep = t.min()
            x = np.linspace(x[0],x[-1],(x[-1]-x[0])/xstep)
            y = spline(x_original,y,x)
        else:
            raise Exception("Data not on a regular x grid")
    ## add padding to data
    xpad = np.arange(dx,fwhms_to_include*fwhm,dx)
    x = np.concatenate((x[0]-xpad[-1::-1],x,x[-1]+xpad))
    y = np.concatenate((np.full(xpad.shape,y[0]),y,np.full(xpad.shape,y[-1])))
    ## convolve
    gx = np.arange(-fwhms_to_include*fwhm,fwhms_to_include*fwhm,dx)
    if (len(gx)%2) == 0:        # is even
        gx = gx[0:-1]
    gx = gx-gx.mean()
    gy = gaussian(gx,fwhm=fwhm,mean=0.,norm='sum')
    assert len(y)>len(gy), 'Data vector is shorter than convolving function.'
    y = np.convolve(y,gy,mode='same')
    ## remove padding
    y = y[len(xpad):-len(xpad)]
    x = x[len(xpad):-len(xpad)]
    ## return to original grid if regridded
    if regridded:
        y = spline(x,y,x_original)
    return y

def convolve_with_spline_signum_regular_grid(
        x,y,                    # input data
        spline_point_list,      # signum amplitude spline points
        xmax,                   # how far to perormm the signum convolution
        **spline_kwargs,        # for computing signum spline
):
    from .fortran_tools import fortran_tools
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=float)
    yconv = np.full(x.shape,0,dtype=float)
    s = spline(*zip(*spline_point_list),x,**spline_kwargs)
    fortran_tools.convolve_with_variable_signum_regular_grid(x,y,s,xmax,yconv)
    return yconv

def convolve_with_spline_signum_regular_grid_2(
        x,y,                    # input data
        spline_point_list,      # signum amplitude spline points
        xmax,                   # how far to perormm the signum convolution
        **spline_kwargs,        # for computing signum spline
):
    dx = (x[-1]-x[0])/(len(x)-1) # grid step -- x must be regular
    ## get hyperbola to convolve -- Δx=0 is zero
    xconv = arange(dx,xmax,dx)
    yconv = 1/xconv
    xconv = np.concatenate((-xconv[::-1],[0],xconv))
    yconv = np.concatenate((-yconv[::-1],[0],yconv))
    ## scale y but signum magnitude
    yscaled =y*spline_from_list(spline_point_list,x,**spline_kwargs)*dx
    ## get convolved asymmetric y to add to self 
    yadd = convolve_with_padding(x,yscaled,xconv,yconv,)
    ## full signum added spectrum
    ynew = y + yadd
    return ynew


def autocorrelate(x,nmax=None):
    if nmax is None:
        retval = np.correlate(x, x, mode='full')
        retval  = retval[int(len(retval-1)/2):]
    else:
        retval = np.empty(nmax,dtype=float)
        for i in range(nmax):
            retval[i] = np.sum(x[i:]*x[:len(x)-i])/np.sum(x[i:]**2)
    return retval

def isnumeric(a):
    """Test if constant numeric value."""
    return type(a) in [int,np.int64,float,np.float64]


def rootname(path,recurse=False):
    """Returns path stripped of leading directories and final
    extension. Set recurse=True to remove all extensions."""
    path = os.path.splitext(os.path.basename(path))[0]
    if not recurse or path.count('.')+path.count('/') == 0:
        return path
    else:
        return rootname(path,recurse=recurse)

def array_to_file(filename,*args,make_leading_directories=True,**kwargs):
    """Use filename to decide whether to attempt to save as an hdf5 file
    or ascii data.\n\nKwargs:\n\n mkdir -- If True, create all leading
    directories if they don't exist. """
    filename = expand_path(filename)
    extension = os.path.splitext(filename)[1]
    if mkdir:
        mkdir(dirname(filename))
    if extension in ('.hdf5','.h5'):
        array_to_hdf5(filename,*args,**kwargs)
    elif extension=='.npy':
        np.save(filename,args,**kwargs)
    elif extension=='.npz':
        np.savez_compressed(filename,args,**kwargs)
    else:
        if not any(isin(('fmt','header',),kwargs)):     # can't use via hdf5 for formatting
            try:
                return Array_to_txt_via_hdf5(filename,*args,**kwargs)
            except:
                pass
        np.savetxt(filename, np.column_stack(args),**kwargs) # fall back

def loadxy(
        filename,
        xkey=None,ykey=None,
        xcol=None,ycol=None,
        **kwargs
):
    """Load x and y data from a file."""
    if xkey is not None:
        ## load by key
        from .dataset import Dataset
        d = Dataset()
        d.load(filename,**kwargs)
        x,y = d['x'],d['y']
    else:
        if xcol is None:
            xcol = 0
        if ycol is None:
            ycol = 1
        d = file_to_array(filename,**kwargs)
        x,y = d[:,xcol],d[:,ycol]
    return x,y

def file_to_array(
        filename,
        xmin=None,xmax=None,    # only load this range
        sort=False,             #
        check_uniform=False,
        awkscript=None,
        unpack=False,
        filetype=None,
        **kwargs,               # passed to function depending on filetype
):
    """Use filename to decide whether to attempt to load as an hdf5
    file or ascii data. xmin/xmax data ranges to load."""
    ## dealt with filename and type
    filename = expand_path(filename)
    if filetype is None:
        filetype = infer_filetype(filename)
    ## default kwargs
    kwargs.setdefault('comments','#')
    kwargs.setdefault('encoding','utf8')
    kwargs.setdefault('delimiter',' ')
    ## load according to filetype
    if filetype=='hdf5':
        hdf5_kwargs = copy(kwargs)
        for key,key_hdf5 in (
                ('comments',None),
                ('delimiter',None),
                ('encoding',None),
                ('skip_header',None),
        ):
            if key_hdf5 is None:
                if key in hdf5_kwargs: hdf5_kwargs.pop(key)
            else:
                hdf5_kwargs[key_hdf5] = hdf5_kwargs.pop(key)
        d = hdf5_to_array(filename,**hdf5_kwargs)
    elif filetype=='npy':
        d = np.load(filename)
    elif filetype in ('opus', 'opus_spectrum', 'opus_background'):
        from . import bruker
        data = bruker.OpusData(filename)
        if filetype in ('opus','opus_spectrum') and data.has_spectrum():
            d = np.column_stack(data.get_spectrum())
        elif data.has_background():
            d = np.column_stack(data.get_background())
        else:
            raise Exception(f"Could not load opus data {filename=}")
    else:
        ## fallback try text
        np_kwargs = copy(kwargs)
        if len(filename)>4 and filename[-4:] in ('.csv','.CSV'):
            np_kwargs['delimiter'] = ','
        if 'delimiter' in np_kwargs and np_kwargs['delimiter']==' ':
            np_kwargs.pop('delimiter')
        d = np.genfromtxt(filename,**np_kwargs)
    ## post processing
    d = np.squeeze(d)
    if xmin is not None:
        d = d[d[:,0]>=xmin]
    if xmax is not None:
        d = d[d[:,0]<=xmax]
    if sort:
        d = d[np.argsort(d[:,0])]
    if check_uniform:
        Δd = np.diff(d[:,0])
        fractional_tolerance = 1e-5
        Δdmax,Δdmin = np.max(Δd),np.min(Δd)
        assert (Δdmax-Δdmin)/Δdmax<fractional_tolerance,f'{check_uniform=} and first column of data is not uniform within {fractional_tolerance=}'
    if unpack:
        d = d.transpose()
    return(d)

def string_to_array(s,**array_kwargs):
    """Convert string of numbers separated by spaces, tabs, commas, bars,
    and newlines into an array. Empty elements are replaced with NaN
    if tabs are used as separators. If spaces are used then the excess
    is removed, including all leading and trailing spaces."""
    ## remove all data after an '#'
    s,count = re.subn(r' *#[^\n]*\n','\n',s)
    ## replace commas and bars with spaces
    s,count = re.subn(r'[|,]',' ',s)
    ## remove leading and trailing and excess spaces, and leading and
    ## trailing newlines
    s,count = re.subn(' {2,}',' ',s)
    s,count = re.subn(r'^[ \n]+|[ \n]+$','',s)
    # s,count = re.subn(r'^\n+|\n+$','',s)
    s,count = re.subn(r' *\n *','\n',s)
    ## spaces to tabs - split on tabs
    s = s.replace(' ','\t')
    ## split on \n
    s = s.splitlines()
    ## strip whitespace
    s = [t.strip() for t in s]
    ## remove blank lines
    s = [t for t in s if len(t)>0]
    ## split each line on tab, stripping each
    s = [[t0.strip() for t0 in t1.split('\t')] for t1 in s]
    ## convert missing values to NaNs
    for i in range(len(s)):
        if s[i] == []: s[i] = ['NaN']
        for j in range(len(s[i])):
            if s[i][j] == '': s[i][j] = 'NaN'
    ## convert to array of numbers, failing that array of strings
    try: 
        s = np.array(s,dtype=np.number,**array_kwargs)
    except ValueError:
        s = np.array(s,dtype=str,**array_kwargs)
        ## also try to convert into complex array -- else leave as string
        try:
            s = np.array(s,dtype=complex,**array_kwargs)
        except ValueError:
            pass
            # # warnings.warn('Nonnumeric value, return string array')
    ## if 2 dimensional transpose so that columns in text file are first index
    # if s.ndim==2: s=s.transpose()
    ## squeeze to smallest possible ndim
    # return s.squeeze()
    return(s.squeeze())

def string_to_array_unpack(s):
    '''return(string_to_array(s).transpose())'''
    return(string_to_array(s).transpose())

def array_to_string(*arrays,fmt='g',field_sep=' ',record_sep='\n'):
    """Convert array to a string format. Input arrays are concatenatd
    column wise.Nicer output than numpy.array2string, only works on 0,
    1 or 2D arrays. Format fmt can be a single string or a list of
    strings corresponding to each column."""
    a = np.column_stack(arrays)
    ## 0D
    if a.ndim==0: return(format(a[0],fmt))
    ## make 1D array 2D
    if a.ndim==1: a = a.reshape((-1,1))
    ## if fmt is a fmt string expand to list with same length as 2nd D
    ## of array
    if isinstance(fmt,str):
        fmt = fmt.strip()
        fmt = fmt.split()
        ## same fmt for all columsn else must be the same length as
        ## columns
        if len(fmt)==1: fmt = [fmt[0] for t in range(a.shape[1])]
    ## build string and return
    return(record_sep.join(
        [field_sep.join(
            [format(t0,t1) for (t0,t1) in zip(record,fmt)]
        ) for record in a]))

def string_to_file(
        filename,
        string,
        mode='w',
        encoding='utf8',
):
    """Write string to file_name."""
    filename = expand_path(filename)
    mkdir(dirname(filename))
    with open(filename,mode=mode,encoding=encoding) as f: 
        f.write(string)

# def str2range(string):
    # """Convert string of integers like '1,2,5:7' to an array of
    # values."""
    # x = string.split(',')
    # r = []
    # for y in x:
        # try:
            # r.append(int(y))
        # except ValueError:
            # y = y.split(':')
            # r.extend(list(range(int(y[0]),int(y[1])+1)))
    # return r

# def derivative(x,y=None,n=1):
    # """Calculate d^ny/dx^n using central difference - end points are
    # extrapolated. Endpoints could use a better formula."""
    # if y is None:
        # x,y = np.arange(len(x),dtype=float),x
    # if n==0:
        # return(y)
    # if n>1:
        # y = derivative(x,y,n-1)
    # d = np.zeros(y.shape)
    # d[1:-1] = (y[2::]-y[0:-2:])/(x[2::]-x[0:-2:])
    # d[0] = (y[1]-y[0])/(x[1]-x[0])
    # d[-1] = (y[-2]-y[-1])/(x[-2]-x[-1])
    # return d

# def curvature(x,y):
    # """Calculate curvature of function."""
    # d=derivative(x,y);  # 1st diff
    # dd=derivative(x,d); # 2nd diff
    # return dd/((1.+d**2.)**(3./2.)) 



def file_to_dict(filename,*args,filetype=None,**kwargs):
    """Convert text file to dictionary.
    \nKeys are taken from the first uncommented record, or the last
    commented record if labels_commented=True. Leading/trailing
    whitespace and leading commentStarts are stripped from keys.\n
    This requires that all elements be the same length. Header in hdf5
    files is removed."""
    filename = expand_path(filename)
    if filetype is None:
        filetype = infer_filetype(filename)
    if filetype == 'text':
        d = txt_to_dict(filename,*args,**kwargs)
    elif filetype=='npz':
        d = dict(**np.load(filename))
        ## avoid some problems later whereby 0D  arrays are not scalars
        for key,val in d.items():
            if val.ndim==0:
                d[key] = np.asscalar(val)
    elif filetype == 'hdf5':
        if 'load_attributes' in kwargs:
            load_attributes = kwargs['load_attributes']
        else:
            load_attributes = False
        d = hdf5_to_dict(filename,load_attributes=load_attributes)
        if 'header' in d:
            d.pop('header') # special case header, not data 
        if 'README' in d:
            d.pop('README') # special case header, not data 
    elif filetype in ('csv','ods'):
        ## load as spreadsheet, set # as comment char
        kwargs.setdefault('comment','#')
        d = sheet_to_dict(filename,*args,**kwargs)
    elif filetype == 'rs':
        ## my convention -- a ␞ separated file
        kwargs.setdefault('comment_regexp','#')
        kwargs.setdefault('delimiter','␞')
        d = txt_to_dict(filename,*args,**kwargs)
    elif filetype == 'org':
        ## load as table in org-mode file
        d = org_table_to_dict(filename,*args,**kwargs)
    elif filetype == 'opus':
        raise ImplementationError()
        # ## load as table in org-mode file
        # d = org_table_to_dict(filename,*args,**kwargs)
    elif filetype == 'directory':
        ## load as data directory
        d = Data_Directory(filename)
    else:
        ## fall back try text
        d = txt_to_dict(filename,*args,**kwargs)
    return d

def infer_filetype(filename):
    """Determine type of datafile from the name or possibly its
    contents."""
    filename = expand_path(filename)
    extension = os.path.splitext(filename)[1]
    if extension=='.npy':
        return 'npy'
    elif extension=='.npz':
        return 'npz'
    elif extension in ('.hdf5','.h5'): # load as hdf5
        return 'hdf5'
    elif extension == '.ods':
        return 'ods'
    elif extension in ('.csv','.CSV'):
        return 'csv'
    elif extension == '.rs':
        return 'rs'
    elif basename(filename) == 'README' or extension == '.org':
        return 'org'
    elif extension in ('.txt','.dat'):
        return 'text'
    elif extension == '.psv':
        return 'psv'
    elif extension == '.TXT':
        return 'desirs_fts'
    elif re.match(r'.*\.[0-9]+$',basename(filename)):
        return 'opus'
    elif extension in ('spectrum','model','experiment',):
        return 'dataset'
    elif os.path.exists(filename) and os.path.isdir(filename):
        return 'directory'
    elif extension == 'dataset':
        return 'directory'
    else:
        return None
    return(d)

_get_extension_data = {
    'npy':'.npy',
    'npz':'.npz',
    'hdf5':'.h5',
    'ods':'.ods',
    'csv':'.csv',
    'rs':'.rs',
    'org':'.org',
    'text':'',
    'opus':'',
    'directory':'',
    'dataset':'.dataset',
    }
def get_extension(filetype):
    return _get_extension_data[filetype]

def file_to_string(filename):
    with open(expand_path(filename),mode='r',errors='replace') as fid:
        string = fid.read(-1)
    return(string)

def file_to_lines(filename,**open_kwargs):
    """Split file data on newlines and return as a list."""
    fid = open(expand_path(filename),'r',**open_kwargs)
    string = fid.read(-1)
    fid.close()
    return(string.split('\n'))

def file_to_tokens(filename,**open_kwargs):
    """Split file on newlines and whitespace, then return as a list of
    lists."""
    return([line.split() for line in file_to_string(filename).split('\n')])

def org_table_to_dict(filename,table_name=None):
    """Load a table into a dicationary of arrays. table_name is used to
    find a #+NAME: tag."""
    with open(filename,'r') as fid:
        if table_name is not None:
            ## scan to beginning of table
            for line in fid:
                if re.match(r'^ *#\+NAME: *'+re.escape(table_name)+' *$',line):
                    break
            else:
                raise Exception("Could not find table_name "+repr(table_name)+" in file "+repr(filename))
        ## skip other metadata
        for line in fid:
            if not re.match(r'^ *#',line): break
        ## load lines of table
        table_lines = []
        for line in fid:
            ## skip horizontal lines
            if re.match(r'^ *\|-',line): continue
            ## end of table
            if not re.match(r'^ *\|',line): break
            ## remove new line
            line = line[:-1]
            ## remove leading and pipe character
            line = re.sub(r'^ *\| *',r'',line)
            ## remove empty following fields
            line = re.sub(r'^[ |]*',r'',line[-1::-1])[-1::-1]
            table_lines.append(line.split('|'))
        ## turn into an array of dicts
        return(stream_to_dict(iter(table_lines)))

def string_to_dict(string,**kwargs_txt_to_dict):
    """Convert a table in string from into a dict. Keys taken from
    first row. """
    ## turn string into an IO object and pass to txt_to_dict to decode the lines. 
    # string = re.sub(r'^[ \n]*','',string) # remove leading blank lines
    import io
    kwargs_txt_to_dict.setdefault('labels_commented',False)
    return(txt_to_dict(io.StringIO(string),**kwargs_txt_to_dict))

def file_to_recarray(filename,*args,**kwargs):
    """Convert text file to record array, converts from dictionary
    returned by file_to_dict."""
    return(dict_to_recarray(file_to_dict(filename,*args,**kwargs)))

def file_to_dataset(*args,**kwargs):
    from . import dataset
    return dataset.Dataset(**file_to_dict(*args,**kwargs))

def string_to_number_if_possible(s):
    """ Attempt to convert string to either int or float. If fail use
    default, or raise error if None."""
    ## if container, then recursively operate on elements instead
    if np.iterable(s) and not isinstance(s,str):
        return([string_to_number_if_possible(ss) for ss in s])
    elif not isinstance(s,str):
        raise Exception(repr(s)+' is not a string.')
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s 

def txt_to_dict(
        filename,              # path or open data stream (will be closed)
        labels=None,
        delimiter=None,
        skiprows=0,
        comment_regexp='#',
        labels_commented=True,
        awkfilter=None,
        filter_function=None,
        filter_regexp=None,
        ignore_blank_lines=True,
        replacement_for_blank_elements='nan',
        try_cast_numeric=True,
):
    """Convert text file to dictionary. Keys are taken from the first
    uncommented record, or the last commented record if
    labels_commented=True. Leading/trailing whitespace and leading
    comment_starts are stripped from keys.
    filter_function: if not None run lines through this function before
    parsing.
    filter_regexp: if not None then must be (pattern,repl) and line
    run through re.sub(pattern,repl,line) before parsing.
    """
    ## If filename is a path name, open it, else assumed is already an open file.
    if type(filename)==str:
        filename = expand_path(filename)
        if awkfilter is not None:
            filename = pipe_through_awk(filename,awkfilter)
        else:
            filename=open(filename,'r', encoding='utf8')
    ## Reads all data, filter, and split
    lines = []
    last_line_in_first_block_of_commented_lines = None
    first_block_commented_lines_passed = False
    number_of_columns = None
    for i,line in enumerate(filename.readlines()):
        if i<skiprows:
            continue
        ## remove leading/trailing whitespace
        line = line.strip() 
        if ignore_blank_lines and len(line)==0:
            continue
        if filter_function is not None:
            line = filter_function(line)
        if filter_regexp is not None:
            line = re.sub(filter_regexp[0],filter_regexp[1],line)
        line = (line.split() if delimiter in (None,' ') else line.split(delimiter)) # split line
        if comment_regexp is not None and re.match(comment_regexp,line[0]): # commented line found
            if not first_block_commented_lines_passed:
                line[0] = re.sub(comment_regexp,'',line[0]) # remove comment start
                if len(line[0])==0:
                    line = line[1:] # first element was comment only,
                last_line_in_first_block_of_commented_lines = line
            continue
        first_block_commented_lines_passed = True # look for no more key labels
        if number_of_columns is None:
            number_of_columns = len(line) 
        else:
            assert len(line)==number_of_columns,f'Wrong number of column on line {i}'
        lines.append(line)      # store this line, it contains data
    filename.close()
    if number_of_columns is None:
        ## no data
        return({}) 
    ## get data labels
    if labels is None:          # look for labels if not given
        if labels_commented:    # expect last line of initial unbroken comment block
            if last_line_in_first_block_of_commented_lines is None:
                labels = ['column'+str(i) for i in range(number_of_columns)] # get labels as column indices
            else:
                labels = [t.strip() for t in last_line_in_first_block_of_commented_lines] # get labels from commented line
        else:
            labels = [t.strip() for t in lines.pop(0)] # get from first line of data
    if len(set(labels))!=len(labels):
        raise Exception(f'Non-unique data labels: {repr(labels)}')
    if len(labels)!=number_of_columns:
        raise Exception(f'Number of labels ({len(labels)}) does not match number of columns ({number_of_columns})')
    if len(lines)==0:
        ## no data
        return({t:[] for t in key}) 
    ## get data from rest of file, and convert to arrays
    data = dict()
    for key,column in zip(labels,zip(*lines)):
        data[key] = [(t.strip() if len(t.strip())>0 else replacement_for_blank_elements) for t in column]
        if try_cast_numeric:
            data[key] = try_cast_to_numeric_array(data[key])
    return data 

def try_cast_to_numeric(string):
    """Try to cast into a numerical type, or else return as a string."""
    try:                 return(int(string))
    except ValueError:   pass
    try:                 return(float(string))
    except ValueError:   pass
    try:                 return(complex(string))
    except ValueError:   pass
    return(str(string))

def try_cast_to_numeric_array(x):
    """Try to cast an interator into an array of ints. On failure try
    floats. On failure return as array of strings."""
    try:
        return np.array(x,dtype=int)
    except ValueError:
        try:
            return np.array(x,dtype=float)
        except ValueError:
            return np.array(x,dtype=str)

def ods_reader(fileName,tableIndex=0):
    """
    Opens an odf spreadsheet, and returns a generator that will
    iterate through its rows. Optional argument table indicates which
    table within the spreadsheet. Note that retures all data as
    strings, and not all rows are the same length.
    """
    import odf.opendocument,odf.table,odf.text
    ## common path expansions
    fileName = expand_path(fileName)
    ## loads sheet
    sheet = odf.opendocument.load(fileName).spreadsheet
    ## Get correct table. If 'table' specified as an integer, then get
    ## from numeric ordering of tables. If specified as a string then
    ## search for correct table name.
    if isinstance(tableIndex,int):
        ## get by index
        table = sheet.getElementsByType(odf.table.Table)[tableIndex]
    elif isinstance(tableIndex,str):
        ## search for table by name, if not found return error
        for table in sheet.getElementsByType(odf.table.Table):
            # if table.attributes[(u'urn:oasis:names:tc:opendocument:xmlns:table:1.0', u'name')]==tableIndex:
            if table.getAttribute('name')==tableIndex:
                break
        else:
            raise Exception('Table `'+str(tableIndex)+'\' not found in `'+str(fileName)+'\'')
    else:
        raise Exception('Table name/index`'+table+'\' not understood.')
    ## divide into rows
    rows = table.getElementsByType(odf.table.TableRow)
    ## For each row divide into cells and then insert new cells for
    ## those that are repeated (multiple copies are not stored in ods
    ## format). The number of multiple copies is stored as a string of
    ## an int.
    for row in rows:
        cellStrs = []
        for cell in row.getElementsByType(odf.table.TableCell):
            cellStrs.append(str(cell))
            if cell.getAttribute('numbercolumnsrepeated')!=None:
                for j in range(int(cell.getAttribute('numbercolumnsrepeated'))-1):
                    cellStrs.append(str(cell))
        ## yield each list of cells to make a generator
        yield cellStrs


def sheet_to_dict(path,return_all_tables=False,skip_header=None,**kwargs):
    """Converts csv or ods file, or list of lists to a dictionary.\n\nFor
    csv files, path can be open file object or path. For ods it must
    be a path\n\nKeys read from first row unless skiprows
    specified.\n\nIf tableName is supplied string then keys and data
    are read betweenen first column flags <tableName> and
    <\\tableName>. Other wise reads to end of file.\n\nConversions
    specify a dictionary of (key,function) pairs where function is
    used to convert the string from of each element of key, rather
    than str2num.\n\nFurther kwargs are passed to csv.reader if a csv
    file is used, or for ods files ignored.\n\nIf there is missing
    data the line might get ignored.  \nLeading/trailing white space
    and leading commentChar.\n\nSpecify ods/xls sheet with
    sheet_name=name.\n\nIf return_all_tables, return a dict of dicts,
    with keys given by all table names found in sheet. """
    import csv
    ## deprecated kwargs
    if 'tableName' in kwargs:
        kwargs['table_name'] = kwargs.pop('tableName')
    if 'commentChar' in kwargs:
        kwargs['comment'] = kwargs.pop('commentChar')
    ## open generator reader according to file extension
    fid = None
    if isinstance(path,list):
        reader = (line for line in path)
    ## some common path expansions
    elif isinstance(path,str) and (path[-4:]=='.csv' or path[-4:]=='.CSV'):
        fid=open(expand_path(path),'r')
        reader=csv.reader(
            fid,
            skipinitialspace=True,
            quotechar=(kwargs.pop('quotechar') if 'quotechar' in kwargs else '"'),)
    elif isinstance(path,str) and path[-4:] =='.ods':
        kwargs.setdefault('sheet_name',0)
        reader=ods_reader(expand_path(path),tableIndex=kwargs.pop('sheet_name'))
    elif isinstance(path,str) and (
            path[-5:] =='.xlsx'
            or path[-4:] =='.xls'):
        assert 'sheet_name' not in kwargs,'Not implemented'
        import openpyxl
        data = openpyxl.open(path,read_only=True,data_only=True,keep_links=False)
        print( data)
        reader=ods_reader(expand_path(path),tableIndex=kwargs.pop('sheet_name'))
    elif isinstance(path,io.IOBase):
        reader=csv.reader(expand_path(path),)
    else:
        raise Exception("Failed to open "+repr(path))
    ## if skip_header is set this is the place to pop the first few recrods of the reader objects
    if skip_header is not None:
        for t in range(skip_header): next(reader)
    ## if requested return all tables. Fine all names and then call
    ## sheet2dict separately for all found tables.
    if return_all_tables:
        return_dict = dict()
        for line in reader:
            if len(line)==0: continue
            r = re.match(r'<([^\\][^>]*)>',line[0],)
            if r:
                table_name = r.groups()[0]
                return_dict[table_name] = sheet_to_dict(path,table_name=table_name,**kwargs)
        return return_dict
    ## load the data into a dict
    data = stream_to_dict(reader,**kwargs)
    ## close file if necessary
    if fid!=None: fid.close()
    ## return
    return data

def stream_to_dict(
        stream,
        split=None,             # a string to split rows on
        comment=None,           # a string to remove from beginnig of rows (regexp is comment+)
        table_name=None,
        conversions={},
        skip_rows=0,
        error_on_missing_data=False,
        types = None,           # a dictionary of data keys will cast them as this type
        cast_types=True,        # attempt to convert strings to numbers
):
    r"""Read a stream (line-by-line iterator) into a dictionary. First
    line contains keys for columns. An attempt is made to cast as
    numeric data.\n\nsplit -- if not None, split line on this character
    comment -- remove from keys if not None
    table_name -- stop reading at end of <\table_name> or <\>
    conversions -- convert data belonging to keys of conversions by a function"""
    ## get keys first line must contain keys, split if requested, else already
    ## iterable, remove trailing new line if necessary
    def get_line():
        line = next(stream)
        ## split if split string given
        if split!=None:
            if line[-1]=='\n':
                line = line[:-1]
            line = line.split(split)
        ## else check if already split (i.e., in a list) or make a
        ## list with one element
        else:
            if np.isscalar(line):
                line = [line]
        ## blank lines -- skip (recurse)
        if len(line)==0:
            line = get_line()
        ## if a comment string is defined then skip this line
        ## (recurse) if it begins with a comment
        if comment!=None:
            if re.match(r'^ *'+re.escape(comment),line[0]): line = get_line()
        return line
    ## skip rows if requested
    for i in range(skip_rows): 
        next(reader)
    ## if requested, scan through file until table found
    if table_name!=None:
        while True:
            try:
                line = get_line()
                if line==[]: continue   # blank line continue
            except StopIteration:
                raise Exception('table_name not found: '+repr(table_name))
            ## if table specified stop reading at the end of it and dont'
            ## store data before it
            except:
                raise               # an actual error
            if len(line)>0 and str(line[0]).strip()=='<'+table_name+'>':
                break # table found
    ## this line contains dict keys
    keys = get_line()           
    ## eliminate blank keys and those with leading/trailing space and
    ## comment char from keys
    if comment != None:
        keys = [re.sub(r'^ *'+comment+r' *',r'',key) for key in keys]
    keys = [key.strip() for key in keys] # remove trailing/leading white space around keys
    nonBlankKeys=[]
    nonBlankKeys = [i for i in range(len(keys)) if keys[i] not in ['','None']]
    keys = [keys[i] for i in nonBlankKeys]
    ## check no repeated keys
    assert len(keys)==len(np.unique(keys)),'repeated keys'
    ## initialise dictionary of lists
    data = dict()
    for key in keys: data[key] = []
    ## read line-by-line, collecting data
    while True:
        try:
            line = get_line()
            if line==[]: continue   # blank line conitinue
        except StopIteration: break # read until end of file
        except: raise               # an actual error
        ## if table specified stop reading at the end of it and dont'
        ## store data before it
        if table_name!=None:
            if '<\\'+table_name+'>'==str(line[0]): break
            if str(line[0]).strip() in ('<\\>','<\\'+table_name+'>',): break
        if len(line)==0 or (len(line)==1 and line[0]==''): continue # skip empty data
        ## if partially missing data pad with blanks or raise an error
        if len(line)<len(keys):
            if error_on_missing_data:
                raise Exception('Length data less than length keys: '+str(line))
            else:
                line.extend(['' for t in range(len(keys)-len(line))])
        line = np.take(line,nonBlankKeys)
        ## add data to lists - loop through each cell and try to cast
        ## it appropriately, if a conversions is explicitly given for
        ## each key, then use that instead
        for (key,cell) in zip(keys,line):
            if key in conversions:
                data[key].append(conversions[key](cell))
            else:
                cell = cell.strip() # remove end blanks
                ## data[key].append(str2num(cell,default_to_nan=False,blank_to_nan=True))
                if cell=="": cell = "nan" # replace empty string with "nan" to facilitate possible numerical convervsion
                data[key].append(cell)
    ## Convert lists to arrays of numbers or whatever. If type given
    ## in types use that, esle try to cast as int, on failure try
    ## float, on failure revert to str.
    if cast_types:
        for key in keys:
            if types is not None and key in types:
                data[key] = np.array(data[key],dtype=types[key])
            else:
                data[key] = try_cast_to_numeric_array(data[key])
    return data

###################
## miscellaneous ##
###################

def digitise_postscript_figure(
        filename,
        xydpi_xyvalue0 = None,  # ((xdpi,ydpi),(xvalue,yvalue)) for fixing axes
        xydpi_xyvalue1 = None,  # 2nd point for fixing axes

):
    """Get all segments in a postscript file. That is, an 'm' command
    followed by an 'l' command. Could find points if 'm' without an 'l' or
    extend to look for 'moveto' and 'lineto' commands."""
    data = file_to_string(filename).split() # load as list split on all whitespace
    retval = []                  # line segments
    ## loop through looking for line segments
    i = 0
    while (i+3)<len(data):
        if data[i]=='m' and data[i+3]=='l': # at least one line segment
            x,y = [float(data[i-1])],[-float(data[i-2])]
            while (i+3)<len(data) and data[i+3]=='l':
                x.append(float(data[i+2]))
                y.append(-float(data[i+1]))
                i += 3
            retval.append([x,y])
        i += 1
    ## make into arrays
    for t in retval:
        t[0],t[1] = np.array(t[0],ndmin=1),np.array(t[1],ndmin=1)
    ## transform to match axes if possible
    if xydpi_xyvalue0 is not None:
        a0,b0,a1,b1 = xydpi_xyvalue0[0][0],xydpi_xyvalue0[1][0],xydpi_xyvalue1[0][0],xydpi_xyvalue1[1][0]
        m = (b1-b0)/(a1-a0)
        c = b0-a0*m
        xtransform = lambda t,c=c,m=m:c+m*t
        a0,b0,a1,b1 = xydpi_xyvalue0[0][1],xydpi_xyvalue0[1][1],xydpi_xyvalue1[0][1],xydpi_xyvalue1[1][1]
        m = (b1-b0)/(a1-a0)
        c = b0-a0*m
        ytransform = lambda t,c=c,m=m:c+m*t
        for t in retval:
            t[0],t[1] = xtransform(t[0]),ytransform(t[1])
    return(retval)

def bibtex_file_to_dict(filename):
    """Returns a dictionary with a pybtex Fielddict, indexed by bibtex
    file keys."""
    from pybtex.database import parse_file
    database  = parse_file(filename)
    entries = database.entries
    retval_dict = dict()
    for key in entries:
        fields = entries[key].rich_fields
        retval_dict[key] = {key:str(fields[key]) for key in fields}
    return(retval_dict)

##############################################################
## functions for time series / spectra / cross sections etc ##
##############################################################

def fit_noise_level(x,y,order=3,plot=False,fig=None):
    """Fit a polynomial through some noisy data and calculate statistics on
    the residual.  Set plot=True to see how well this works."""
    p = np.polyfit(x,y,order)
    yf = np.polyval(p,x)
    r = y-yf
    nrms = rms(r)
    if plot:
        if fig is None:
            fig = plotting.qfig()
        else:
            fig.clf()
        ax = fig.gca()
        ax.plot(x,y)
        ax.plot(x,yf)
        ax = plotting.subplot()
        ax.plot(x,r)
        plotting.hist_with_normal_distribution(r, ax=plotting.subplot())
    return nrms

def fit_background(
        x,                      # x data, probably should be ordered
        y,                      # y data, same length as x
        fit_min_or_max='max', # 'max' to fit absorption data, 'min' for emission
        x1=3, # spline x-points (or interval) for initial fit of the local maximum/minimum y-values
        auto_trim_half=True,    # refit keeping half of the data to hopefully recover the mean
        x2=None, # spline x-points for a following fit of all data encompassed by trim
        trim=(0,1), # fractional interval of data to keep ordered by initial fit residual. (0,1) = all points, (0.9,1) points with the highest 10% y-value
        make_plot=True,
        order=3,         # spline order
):
    """Initially fit background of a noisy spectrum to a spline fixed
    to maximum (minimum) values in an interval around points x1.  The,
    optionally cut off half of the data and refit the min (max) values
    of the remainder, perhaps finding the centre of scatter. Then
    optionally (if x2 is set) least-squares spline-fit data with
    y-data in the fractional interval "trim"."""
    ## fit a spline to max or min points near x1
    xspline,yspline,yfit = fit_spline_to_extrema(x,y,fit_min_or_max,x1,1/4,order=order)
    yresidual = y-yfit
    ## if auto_trim_half then find the upper 50% of data and fit hte
    ## bottom edge (hopefully given a line through the mean of the
    ## local noise
    if auto_trim_half:
        if fit_min_or_max == 'min':
            t_fit_min_or_max = 'max'
            t_trim = (0,0.5)
        else:
            t_fit_min_or_max = 'min'
            t_trim = (0.5,1)
        itrim = np.full(x.shape,False)
        itrim[int(len(x)*t_trim[0]):int(len(x)*t_trim[1])] = True
        itrim = itrim[np.argsort(np.argsort(yresidual))] # put in x-order
        itrim[0] = itrim[-1] = True                      # do not trim endpoints
        xtrim,ytrim = x[itrim],y[itrim]
        xspline,yspline,ytrimfit = fit_spline_to_extrema(
            xtrim,ytrim,t_fit_min_or_max,x1,1/4,order=order)
        yfit = spline(xspline,yspline,x)
        yresidual = y-yfit
    ## get fitted statistics
    ## μ,σ = fit_normal_distribution(yresidual[itrim])
    ## refit trimmed data
    if x2 is not None:
        ## another trim of residual maxima and minima 
        itrim = np.full(x.shape,False)
        itrim[int(len(x)*trim[0]):int(len(x)*trim[1])] = True
        itrim = itrim[np.argsort(np.argsort(yresidual))] # put in x-order
        itrim[0] = itrim[-1] = True                      # do not trim endpoints
        ## fit splien to full trimmed data
        xspline,yspline = fit_least_squares_spline(x[itrim],y[itrim],x2)
        yfit = spline(xspline,yspline,x,order=order)
    # ## adjust for missing noise due to trimming
    # if not np.isnan(μ):
        # yfit += μ
    if make_plot:
        ## plot some stuff
        ax = plotting.gca()
        ax.cla()
        ax.plot(x,y,label='data')
        ax.plot(x[itrim],y[itrim],label='trimmed data')
        ax.plot(xtrim,ytrimfit,label='refit trimmed data')
        ax.plot(x,yfit,lw=3,label='yfit')
        ax.plot(x,yresidual,lw=3,label='yresidual')
        legend(ax=ax)
        # ax1 = subplot(1,fig=figure)
        # n,b,p = ax1.hist(yresidual[itrim],max(10,int(len(y)/200)),density=True)
        # b = (b[1:]+b[:-1])/2
        # t = normal_distribution(b,μ,σ)
        # ax1.plot(b,t/np.mean(t)*np.mean(n))
        # ax1.set_title('Fitted normal distribution')
    return xspline,yspline,yfit

def fit_least_squares_spline(
        x,                      # x data in spetrum -- sorted
        y,                      # y data in spectra
        xspline=10,             # spline points or an interval
        order=3,
):
    """Fit least squares spline coefficients at xspline to (x,y)."""
    ## get x spline points
    xbeg,xend = x[0],x[-1]
    if np.isscalar(xspline):
        xspline = np.linspace(xbeg,xend,max(2,int((xend-xbeg)/xspline)))
    xspline = np.asarray(xspline,dtype=float)
    ## get initial y spline points
    yspline = np.array([y[np.argmin(np.abs(x-xsplinei))] for xsplinei in xspline])
    print( f'optimising {len(yspline)} spline points onto {len(y)} data points')
    yspline,dyspline = leastsq(lambda yspline:y-spline(xspline,yspline,x), yspline, yspline*1e-5,)
    return xspline,yspline

def fit_spline_to_extrema(
        x,                      # x data in spetrum - must be sorted
        y,                      # y data in spectra
        fit_min_or_max='max',
        xi=10, # x values to fit spline points, or interval of evenly spaced points
        interval_fraction=1/4,  # select a value in this interval around xi
        order=3,s=0,            # spline parameters
):
    """Fit a spline to data defined at points near xi. Exact points
    are selected as the maximum (minimum) in an interval around xi
    defined as the fraction bounded by neighbouring xi."""
    assert fit_min_or_max in ('min','max')
    ## get xi spline points
    xbeg,xend = x[0],x[-1]
    if np.isscalar(xi):
        xi = np.linspace(xbeg,xend,max(2,int((xend-xbeg)/xi)))
    xi = np.asarray(xi,dtype=float)
    assert np.all(np.sort(xi)==xi),'Spline points not monotonically increasing'
    ## get y spline points
    interval_beg = np.concatenate((xi[0:1], xi[1:]-(xi[1:]-xi[:-1])*interval_fraction))
    interval_end = np.concatenate(((xi[:-1]+(xi[1:]-xi[:-1])*interval_fraction,x[-1:])))
    xspline,yspline = [],[]
    for begi,endi in zip(interval_beg,interval_end):
        if begi>x[-1] or endi<x[0]:
            ## out of bounds of data
            continue
        iinterval = (x>=begi)&(x<=endi)
        if fit_min_or_max == 'min':
            ispline = find(iinterval)[np.argmin(y[iinterval])]
        elif fit_min_or_max == 'max':
            ispline = find(iinterval)[np.argmax(y[iinterval])]
        xspline.append(x[ispline])
        yspline.append(y[ispline])
    # xspline,yspline = np.array(xspline),np.array(yspline)
    xspline[0],xspline[-1] = x[0],x[-1] # ensure endpoints are included
    yf = spline(xspline,yspline,x,order=order,s=s)
    return xspline,yspline,yf

def fit_spline_to_extrema_or_median(
        x,                      # x data (sorted)
        y,                      # y data 
        fit='median',           # 'median', 'min', or 'max'
        xi=5, # x values to fit spline points, or interval for evenly spaced points
        interval_fraction=0.4,  # select a value from this half-interval around xi
        refit_median =  True,
        order=3,            # spline order
        make_plot=False,     # show what was done
):
    """Fit a spline to (x,y) with knots in intervals around points
    xi. Knots are at either the maximum or minimum y value in each
    interval, or the median point. If refit_median then find the median of
    residual error of first fit and fit to that."""
    ## get xi spline points
    if np.isscalar(xi):
        xi = np.linspace(x[0],x[-1],max(2,int((x[-1]-x[0])/xi)))
    xi = np.asarray(xi,dtype=float)
    assert np.all(np.sort(xi)==xi),'Spline points not monotonically increasing'
    ## find intervals to fit get spline points in
    ## add extra outer boundary points
    if len(xi) == 1:
        ## if only one point given then use equal width interval on both sides
        xi = array([x[0],x[-1]])
    ## get allowed interval fraction around each xi 
    xi = np.concatenate((xi[:1],xi,xi[-1:]))
    xbeg = xi[1:-1] - (xi[1:-1] - xi[:-2] ) * min(interval_fraction,0.5)
    xend = xi[1:-1] + (xi[2:]   - xi[1:-1]) * min(interval_fraction,0.5)
    ## and as inidices
    ibeg =  np.searchsorted(x,xbeg)
    iend =  np.searchsorted(x,xend)
    iend = np.max([iend,ibeg+1],0) # ensure at least one point
    ## join overlapping ibeg/iend into single interval
    while np.any(i:=(ibeg[1:]<iend[:-1])):
        assert False
    ## find min or max in intervals
    xspline,yspline = [],[]     
    for ibegi,iendi in zip(ibeg,iend):
        xi = x[ibegi:iendi]
        yi = y[ibegi:iendi]
        if fit == 'min':
            j = np.argmin(yi)
        elif fit == 'max':
            j = np.argmax(yi)
        elif fit == 'median':
            j = argmedian(yi)
        else:
            raise Exception(f'Invalid value {fit=}, expecint "min", "max", or "median"')
        xspline.append(xi[j])
        yspline.append(yi[j])
    yf = spline(xspline,yspline,x,order=order)
    if refit_median:
        ## refit data to the median of each 
        xspline,yspline = [],[]
        for ibegi,iendi in zip(ibeg,iend):
            ## data for this interval
            xi = x[ibegi:iendi]
            yi = y[ibegi:iendi]
            yresiduali = y[ibegi:iendi]-yf[ibegi:iendi]
            ## find median data point
            j = argmedian(yresiduali)
            xspline.append(xi[j])
            yspline.append(yi[j])
        yf = spline(xspline,yspline,x,order=order)
    if make_plot:
        ## summarise the results
        gca().plot(x,y,label='data')
        gca().plot(xspline,yspline,marker='o',ls='',label='spline points')
        gca().plot(x,yf,label='fit')
        ## plot intervals
        ## for ibegi,iendi in zip(ibeg,iend):
        ##     gca().plot(x[ibegi:iendi],y[ibegi:iendi],color='black')
        legend()
    ## returen as a list of (xspline,yspline) pairs where yspline
    ## points are Parameter object, or return three lists
    ## (xspline,yspline,yfit).
    # if return_as_parameters:
        # from .optimise import Parameter
        # return [[xi,Parameter(yi)] for xi,yi in zip(xspline,yspline)]
    # else:
        # return xspline,yspline,yf
    return xspline,yspline,yf

def piecewise_sinusoid(x,regions,Aorder=3,forder=3):
    """"""
    from scipy import constants
    xmid = []
    As = []
    fs = []
    p = np.full(x.shape,0.0)
    xs = np.full(x.shape,0.0)
    for iregion,(xbeg,xend,amplitude,frequency,phase) in enumerate(regions):
        i = slice(*np.searchsorted(x,[xbeg,xend]))
        p[i] = float(phase)
        xs[i] = x[i]-xbeg
        xmid.append((xbeg+xend)/2)
        fs.append(float(frequency))
        As.append(float(amplitude))
    ## compute sinusoid -- zero outside defined regions
    i0,i1  = np.searchsorted(x,(regions[0][0],regions[-1][1]))
    i = slice(i0,i1+1)
    retval = np.full(x.shape,0.0)
    retval[i] = (
        spline(xmid,As,x[i],order=Aorder)
        *np.cos(
            2*constants.pi*xs[i]*spline(xmid,fs,x[i],order=forder)
            +p[i]))
    return retval

def fit_piecewise_sinusoid(
        x,y,
        xi=10,
        plot=True,
        optimise=False,
):
    """Define piecewise sinusoids for regions joined by points xi (or on
    grid with interval xi)."""
    import scipy
    ## get join points between regions and begining and ending points
    if np.isscalar(xi):
        xi = np.linspace(x[0],x[-1],max(2,int(np.ceil((x[-1]-x[0])/xi))))
    ## loop over all regions, gettin dominant frequency and phase
    ## from the residual error power spectrum
    regions = []
    shift = np.full(x.shape,0.0)
    for xbegi,xendi in zip(xi[:-1],xi[1:]):
        i = slice(*np.searchsorted(x,[xbegi,xendi]))
        shift[i] = np.mean(y[i])
        xti = x[i]
        yi = y[i] - shift[i]
        FT = scipy.fft.fft(yi)
        imax = np.argmax(np.abs(FT)[1:])+1 # exclude 0
        phase = np.arctan(FT.imag[imax]/FT.real[imax])
        if FT.real[imax]<0:
            phase += scipy.constants.pi
        dx = (xti[-1]-xti[0])/(len(xti)-1)
        frequency = 1/dx*imax/len(FT)
        amplitude = rms(yi)
        regions.append((xbegi,xendi,amplitude,frequency,phase))
    ## nonlinear optimisation of  sinusoid parameters 
    if optimise:
        from .optimise import Optimiser,P,_collect_parameters_and_optimisers
        optimiser = Optimiser()
        region_parameters = [[xbeg,xend,P(amplitude,True),P(frequency,True),P(phase,True,2*scipy.constants.pi*1e-5),]
                             for (xbeg,xend,amplitude,frequency,phase) in regions]
        for parameter in _collect_parameters_and_optimisers(region_parameters)[0]:
            optimiser.add_parameter(parameter)
        def f():
            return y - (piecewise_sinusoid(x,region_parameters) + shift)
        optimiser.add_construct_function(f)
        optimiser.optimise()
        regions = [[xbeg,xend,float(amplitude),float(frequency),float(phase)]
                   for (xbeg,xend,amplitude,frequency,phase) in region_parameters]
    ## plot result
    if plot:
        from . import tools
        from . import plotting
        yf = piecewise_sinusoid(x,regions)# + shift
        ax = plotting.gca()
        ax.cla()
        ax.plot(x,y,label=f'data (rms={tools.rms(y):0.2e})')
        ax.plot(x,yf,label=f'fit (rms={tools.rms(yf):0.2e})')
        ax.plot(x,y-yf,label=f'residual (rms={tools.rms(y-yf):0.2e})')
        plotting.legend()
        plotting.show()
    return regions

def bin_data(x,n,mean=False):
    """Reduce the number of points in y by factor n, either summing or computing the mean of 
    n neighbouring points. Any data remaining after the last complete bin is discarded."""
    if n == 1:
        return x
    x = np.asarray(x,dtype=float)
    n = np.asarray(n,dtype=int)
    
    retval = np.empty(len(x)//n)
    fortran_tools.bin_data(x,retval,n,len(x),len(retval))
    if mean:
        retval /= n
    return retval

def resample(xin,yin,xout):
    """One particular way to spline or bin (as appropriate) (x,y) data to
    a given xout grid. Trapezoidally-integrated value is preserved."""
    from scipy import integrate
    assert np.all(xin==np.unique(xin)),'Input x-data not monotonically increasing.'
    assert all(yin>=0),'Negative cross section in input data'
    assert not np.any(np.isnan(yin)),'NaN cross section in input data'
    assert xout[0]>=xin[0],'Output x minimum less than input.'
    assert xout[-1]<=xin[-1],'Output x maximum greater than input.'
    ## integration region boundary points -- edge points and mid
    ## points of xout
    xbnd = np.concatenate((xout[0:1],(xout[1:]+xout[:-1])/2,xout[-1:]))
    ## linear spline data to original and boundary points
    xfull = np.unique(np.concatenate((xin,xbnd)))
    yfull = spline(xin,yin,xfull,order=1)
    ## indentify boundary pointsin full 
    ibnd = np.searchsorted(xfull,xbnd)
    ## compute trapezoidal cumulative integral 
    ycum = np.concatenate(([0],integrate.cumtrapz(yfull,xfull)))
    ## output cross section points are integrated values between
    ## bounds
    yout = (ycum[ibnd[1:]]-ycum[ibnd[:-1]])/(xfull[ibnd[1:]]-xfull[ibnd[:-1]])
    return yout

def resample_out_of_bounds_to_zero(xin,yin,xout):
    """Like resample but can handle out of bounds by setting this to
    zero."""
    yout = np.zeros(xout.shape,dtype=float)
    i = (xout>=xin[0])&(xout<=xin[-1])
    if sum(i)>0:
        yout[i] = resample(xin,yin,xout[i])
    return yout

def find_peaks(
        y,
        x=None,                    # if x is None will use index
        peak_type='maxima',     # can be 'maxima', 'minima', 'both'
        peak_min=None, # minimum height of trough between adjacent peaks as a fraction of the lowest neighbouring peak height. I.e., 0.9 would be a very shallow trough.
        ybeg = None,       # peaks below this will be ignored
        yend = None,      # peaks below this will be ignored
        xbeg = None,       # peaks below this will be ignored
        xend = None,      # peaks below this will be ignored
        x_minimimum_separation = None, # two peaks closer than this will be reduced to the taller
        return_coords=True,
):
    """A reworked version of locate_peaks with difference features. Does
not attempt to fit the background with a tensioned spline, isntead
this should already be reduced to zero for the fractional_trough_depth
part of the algorithm to work. """
    ## find both maxima and minima
    assert peak_type in ('maxima', 'minima', 'both')
    if peak_type=='both':
        maxima = find_peaks(y,x,'maxima',fractional_trough_depth,ybeg,yend,xbeg,xend,x_minimimum_separation)
        minima = find_peaks(y,x,'minima',fractional_trough_depth,ybeg,yend,xbeg,xend,x_minimimum_separation)
        return np.concatenate(np.sort(maxima,minima))
    if ybeg is None:
        ybeg = -np.inf
    if yend is None:
        yend = np.inf
    ## get data in correct array format
    y = np.array(y,ndmin=1)             # ensure y is an array
    if x is None:
        x = np.arange(len(y)) # default x to index
    assert all(np.diff(x)>0), 'Data not sorted or unique with respect to x.'
    ## in case of minima search
    if peak_type=='minima':
        y *= -1
        ybeg,yend = -1*yend,-1*ybeg
    ## shift for conveniences
    shift = np.min(y)
    y -= shift
    ybeg -= shift
    yend -= shift
    ## find all peaks
    ipeak = find((y[:-2]<=y[1:-1])&(y[2:]<=y[1:-1]))+1
    ## limit to minima/maxima
    if ybeg is not None:
        ipeak = ipeak[y[ipeak]>=ybeg]
    if yend is not None:
        ipeak = ipeak[y[ipeak]<=yend]
    if xbeg is not None:
        ipeak = ipeak[x[ipeak]>=xbeg]
    if xend is not None:
        ipeak = ipeak[x[ipeak]<=xend]
    ## compare with next point to see if trough is deep enough to
    ## count as tow peaks. If not keep the tallest peak.
    if peak_min is not None:
        ipeak = list(ipeak)
        i = 0
        while i < (len(ipeak)-1):
            ## index of minimum between maxima
            j = ipeak[i] + np.argmin(y[ipeak[i]:ipeak[i+1]+1])
            
            if (min(y[ipeak[i]],y[ipeak[i+1]])-y[j]) > peak_min:
                ## no problem, proceed to next maxima
                i += 1
            else:
                ## not happy, delete the lowest height maxima
                if y[ipeak[i]]>y[ipeak[i+1]]:
                    ipeak.pop(i+1)
                else:
                    ipeak.pop(i)
        ipeak = np.array(ipeak)
    ## if any peaks are closer than x_minimimum_separation then keep the taller.
    if x_minimimum_separation is not None:
        while True:
            jj = find(np.diff(x[ipeak]) < x_minimimum_separation)
            if len(jj)==0: break
            for j in jj:
                if y[ipeak[j]]>y[ipeak[j+1]]:
                    ipeak[j+1] = -1
                else:
                    ipeak[j] = -1
            ipeak = [ii for ii in ipeak if ii!=-1]
    ipeak = np.array(ipeak)
    return ipeak



#################
## sympy stuff ##
#################

# @functools.lru_cache
# def cached_pycode(*args,**kwargs):
    # return(pycode(*args,**kwargs))

@functools.lru_cache
def lambdify_sympy_expression(
        sympy_expression,
        *args,                  # strings denoting input arguments of lambda function
        **kwargs,               # key=val, set to kwarg arguments of lambda function
): 
    """An alternative to sympy lambdify.  ."""
    import sympy
    ## make into a python string
    # t =  cached_pycode(sympy_expression,fully_qualified_modules=False)
    t =  sympy.printing.pycode(sympy_expression,fully_qualified_modules=False)
    ## replace math functions
    for t0,t1 in (('sqrt','np.sqrt'),):
        t = t.replace(t0,t1)
    ## build argument list into expression
    arglist = list(args) + [f'{key}={repr(val)}' for key,val in kwargs.items()] 
    eval_expression = f'lambda {",".join(arglist)},**kwargs: np.asarray({t},dtype=float)' # NOTE: includes **kwargs
    return eval(eval_expression)

