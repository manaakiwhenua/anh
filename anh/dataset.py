import re
from copy import copy,deepcopy
from pprint import pprint,pformat
import importlib
import warnings
import itertools

import numpy as np
from numpy import nan,arange,linspace,array

from . import tools
from .tools import timestamp,cache
from .exceptions import InferException,NonUniqueValueException
from . import convert
from . import optimise
from .optimise import optimise_method,Parameter,Fixed,format_input_method,P
from . import __version__



##########################################
## load and convert data_dict functions ##
##########################################

## Dataset.load_from_dict loads data from a dictionary mirroring the
## structure of the dataset class, with an example below.  The
## ['data'][key] subdicts must contain a 'value' and optionally any
## other valid Dataset subkey. The various _load_* functions below
## have the task of converting a data source into a valid data_dict.

_example_data_dict = {
    'description': 'string',
    'attributes': {'attr0':'string','attr1':1,'attr2':['a',3,None]},
    'data': {
        'key0': {
            'value': [1,2,3],
            'description': 'string',
            'fmt': 'g',},
        'key1': {
            'value': ['a','b','c'],},}}



def _convert_flat_data_dict(flat_data_dict):
    """Convert a data dict without nesting into a nested dictionary in the
    correct format for load_from_dict. With subkeys specified as
    'key:subkey'.

    An example flat dict:
    
    flat_data_dict = {
        'key0':['a','b','c'],
        'key1':[1,2,3],
        'key1:fmt':'d',
        'key1:unc':[0.1,0.1,0.2],}

    is converted to

    data_dict = {
        'data': {
            'key0': {'value':['a','b','c'],},
            'key1':{'value':[1,2,3],'fmt':'d','unc':[0.1,0.1,0.2],},},}

    """
    ## nested dict
    data_dict = {}
    ## initialise nested 'data' and 'attibute' dicts. Actually if
    ## they already exist as dicitonaries in flat_data_dict then
    ## just copy them
    for key in ('data','attributes'):
        if key in flat_data_dict and isinstance(flat_data_dict[key],dict):
            data_dict[key] = flat_data_dict.pop(key)
        else:
            data_dict[key] = {}
    ## flat_data_dict key description is assumned to be the
    ## description of this class
    if 'description' in flat_data_dict:
        data_dict['description'] = str(flat_data_dict.pop('description'))
    ## add remaining keys in flat_data_dict to the data subdict
    for key,val in flat_data_dict.items():
        if r:=re.match(r'([^:]+)[:]([^:]+)',key): 
            key,subkey = r.groups()
        else:
            subkey = 'value'
        data_dict['data'].setdefault(key,{})
        data_dict['data'][key][subkey] = val
    return data_dict

def _load_from_directory(filename):
    """Load data stored in a structured directory tree."""
    data_dict = tools.directory_to_dict(filename,evaluate_strings=True)
    ## determine file formatting and version of spectr used to
    ## save this directory
    if ('format' in data_dict
        and isinstance(data_dict['format'],dict)
        and 'version' in data_dict['format']):
        version = data_dict['format']['version']
    else:
        version = None
    ## make any modificatiosn to make this compatible with the
    ## current version
    if version is None:
        ## no version data
        if 'data' not in data_dict:
            ## no 'data' assume this is pre version 1.0.0 --
            ## DELETE THIS BLOCK SOME DAY
            ## 
            ## add all keys to a data dict
            data = {}
            for key in list(data_dict):
                if key not in ('classname','description','attributes','default_step'):
                    data[key] = data_dict.pop(key)
            data_dict['data'] = data
            ## move data out of old 'assoc' subdict
            for key in data:
                if 'assoc' in list(data[key]):
                    for subkey in data[key]['assoc']:
                        data[key][subkey] = data[key]['assoc'][subkey]
                    data[key].pop('assoc')
        else:
            ## assume most recent version
            pass
    else:
        ## assume most recent version
        pass
    return data_dict

def _load_from_hdf5(filename,load_attributes=True):
    """Load data stored in a structured or unstructured hdf5 file."""
    data = tools.hdf5_to_dict(filename,load_attributes=load_attributes)
    ## hack to get flat data or not
    for val in data.values():
        if isinstance(val,dict):
            flat = False
            break
    else:
        flat = True
    ## if data consists of a single 2D array then replace it with
    ## with enumerated columns
    if len(data) == 1:
        key = list(data)[0]
        val = data[key]
        if isinstance(val,np.ndarray) and val.ndim == 2:
            for i,column in enumerate(data.pop(key).T):
                data[f'{key}{i}'] = column
    ## HACK: make compatible with old data format
    if 'data' not in data:
        data['data'] = {}
        for key in copy(data):
            if (isinstance(data[key],dict)
                and 'value' in data[key]):
                data['data'][key] = data.pop(key)
    ## end of HACK
    if flat:
        data = _convert_flat_data_dict(data)
    return data

def _load_from_npz(filename):
    """numpy npz archive.  get as scalar rather than
    zero-dimensional numpy array"""
    data = {}
    for key,val in np.load(filename).items():
        if val.ndim == 0:
            val = val.item()
        data[key] = val
    data = _convert_flat_data_dict(data)
    return data

def _load_from_sqlite(filename,table_name=None,keys=None):
    """Load from a sqlite database using standard library module
    sqlite3. table_name defaults to a strictly unique value."""
    ## open database
    import sqlite3
    with sqlite3.connect(filename) as database_connection:
        database_cursor = database_connection.cursor()
        ## get unique table name if not given in input arguments
        if table_name is None:
            table_names = database_cursor.execute(
                "SELECT name FROM sqlite_master"
            ).fetchall()
            if len(table_names)!=1:
                raise Exception("Can only load sqlite database for one and only one data table.")
            table_name = table_names[0][0]
        ## get keys if not given in input arguments
        if keys is None:
            keys = database_cursor.execute(f"SELECT name FROM pragma_table_info('{table_name}');").fetchall()
            keys = [t[0] for t in keys]
        ## load data into dict. SLOW USES LISTS AND NOT ARRAYS?!?
        data = {}
        for key in keys:
            tdata = database_cursor.execute(
                f"SELECT {key} FROM {table_name}").fetchall()
            data[key] = np.array(tdata).flatten()
    ## make structure dict and return
    data = _convert_flat_data_dict(data)
    return data
 
def _load_from_org(filename,table_name=None):
    """Load form org table"""
    data = tools.org_table_to_dict(filename,table_name)
    data = _convert_flat_data_dict(data)
    return data

def _load_from_simple_text(filename,**txt_to_dict_kwargs):
    """Load data from a simple text table without any formatted header information."""
    filename = tools.expand_path(filename)
    data = tools.txt_to_dict(filename,**txt_to_dict_kwargs)
    data = _convert_flat_data_dict(data)
    return data

def _load_from_text(
        filename,
        comment='#',
        labels_commented=False, # data labels are commented
        delimiter=' ',
        txt_to_dict_kwargs=None,
        header_commented=False, # header preceding data labels is commented
):
    """Load data from a text-formatted Dataset file."""
    ## text table to dict with header
    if txt_to_dict_kwargs is None:
        txt_to_dict_kwargs = {}
    txt_to_dict_kwargs |= {'delimiter':delimiter,'labels_commented':labels_commented}
    filename = tools.expand_path(filename)
    data = {}
    metadata = {}
    ## load header
    escaped_comment = re.escape(comment)
    blank_line_re = re.compile(r'^ *$')
    commented_line_re = re.compile(f'^ *{escaped_comment} *(.*)$')
    beginning_of_section_re = re.compile(f'^ *\\[([^]]+)\\] *$') 
    key_line_without_value_re = re.compile(f'^ *([^# ]+) *# *(.+) *') # no value in line
    key_line_with_value_re = re.compile(f'^ *([^= ]+) *= *([^#]*[^ #])') # may also contain description
    current_section = 'data'
    valid_sections = ('classname','description','keys','data','metadata','attributes')
    section_iline = 0       # how many lines read in this section
    classname = None
    description = None
    attributes = []
    with open(filename,'r') as fid:
        for iline,line in enumerate(fid):
            ## remove newline
            line = line[:-1]
            ## check for bad section title
            if current_section not in valid_sections:
                raise Exception(f'Invalid data section: {repr(current_section)}. Valid sections: {repr(valid_sections)}')
            ## remove comment character unless in data section —
            ## then skip the line, or description then keep it in
            ## place
            if r:=re.match(commented_line_re,line):
                if header_commented:
                    line = re.sub(f'^ *{escaped_comment} *','',line)
                elif current_section == 'description':
                    pass
                else:
                    continue
            ## skip blank lines unless in the description
            if re.match(blank_line_re,line) and current_section != 'description':
                continue
            ## moving forward in this section
            section_iline += 1
            ## process data from this line
            if r:=re.match(beginning_of_section_re,line):
                ## new section header line
                current_section = r.group(1)
                section_iline = 0
                if current_section == 'description':
                    description = ''
                continue
            elif current_section == 'classname':
                ## save classname 
                if section_iline > 1:
                    raise Exception("Invalid classname section")
                classname = line.strip()
            elif current_section == 'description':
                ## add to description
                description += '\n'+line
            elif current_section == 'attributes':
                ## add global attribute
                attributes.append(line)
            elif current_section == 'keys':
                ## decode key line getting key, value, and any metadata
                r = re.match(
                    f'^(?:{escaped_comment}| )*([^#= ]+) *(?:= *([^ #]+))? *(?:# *(.* *))?',
                    line)
                key = None
                value = None
                info = None
                if r:
                    if r.group(1) is not None:
                        key = r.group(1)
                    if r.group(2) is not None:
                        value = tools.safe_eval_literal(r.group(2))
                    if r.group(3) is not None:
                        try:
                            info = tools.safe_eval_literal(r.group(3))
                            if not isinstance(info,dict):
                                info = {'description':r.group(3)}
                        except:
                            info = {'description':r.group(3)}
                    if value is not None:
                        data[key] = value
                    if info is not None:
                        metadata[key] = info
            elif current_section == 'metadata': 
                ## decode key line getting key, value, and any
                ## metadata from an python-encoded dictionary e.g.,
                ## key={'description':"abd",kind='f',value=5.0,...}.
                ## Or key=description_string.
                r = re.match(f'^(?:{escaped_comment}| )*([^= ]+)(?: *= *(.+))?',line) 
                if r:
                    key = r.group(1)
                    if r.group(2) is None:
                        key_metadata = None
                    else:
                        try:
                            key_metadata = tools.safe_eval_literal(r.group(2))
                        except:
                            raise Exception(f"Invalid metadata encoding for {repr(key)}: {repr(r.group(2))}")
                        if isinstance(key_metadata,dict):
                            pass
                        elif isinstance(key_metadata,str):
                            key_metadata = {'description':key_metadata}
                        else:
                            raise Exception(f'Could not decode key metadata for {key}: {repr(key_metadata)}')
                        if 'default' in key_metadata:
                            data[key] = key_metadata['default']
                        metadata[key] = key_metadata
            elif current_section == 'data':
                ## remainder of data is data, no more header to
                ## process
                break
    ## load array data
    if header_commented:
        ## iline will have skipped over labels otherwise
        iline -= 1
    data.update(
        tools.txt_to_dict(
            filename,
            skiprows=iline,
            try_cast_numeric=False,
            **txt_to_dict_kwargs))
    ## a blank key indicates a leading or trailing delimiter -- delete it
    if '' in data:
        data.pop('')
    ## if there is no kind for this key then try and cast to numeric data
    for key in data:
        if key not in metadata or 'kind' not in metadata[key]:
            data[key] = tools.try_cast_to_numeric_array(data[key])
    ## return as non flat with metadata inserted 
    data = _convert_flat_data_dict(data)
    for key,metadata_val in metadata.items():
        for subkey,val in metadata_val.items():
            data['data'][key][subkey] = val
    ## add other header data
    if classname is not None:
        data['classname'] = classname
    if description is not None:
        data['description'] = description
    if len(attributes) > 0:
        tdict = '{'+','.join(attributes)+'}'
        data['attributes'] = tools.safe_eval_literal(tdict)
    return data


#######################
## The Dataset class ##
#######################

class Dataset(optimise.Optimiser):

    """Stores a table of data vectors of common length indexed by key.

    For the types of data that can be bstored see "data_kinds".  For
    the associated data that can be store alongside its value
    "vector_subkinds" and "scalar_subkinds".

    For indexing see __getitem__."""

    ## The kinds of data that may be stored as a 'value' and their
    ## default prototype data
    data_kinds = {
        'f': {'description': 'float'         , 'cast': lambda x: np.array(x,dtype=float) , 'fmt': '+12.8e', },
        'a': {'description': 'positive float', 'cast': lambda x: np.array(x,dtype=float) , 'fmt': '+12.8e', },
        'i': {'description': 'int'           , 'cast': lambda x: np.array(x,dtype=int)   , 'fmt': 'd'     , },
        'b': {'description': 'bool'          , 'cast': tools.convert_to_bool_vector_array  , 'fmt': ''      , },
        'U': {'description': 'str'           , 'cast': lambda x: np.array(x,dtype=str)   , 'fmt': 's'     , },
        'O': {'description': 'object'        , 'cast': lambda x: np.array(x,dtype=object), 'fmt': ''      , },
        'h': {'description': 'SHA1 hash'     , 'cast': lambda x: np.array(x,dtype='S20') , 'fmt': ''      , },
    }

    ## Kinds of subdata associatd with a key that are vectors of the
    ## same length as 'value'. All subkinds other than value must have
    ## a default.
    vector_subkinds = {
        'value':    { 'description': 'Value of this data' }, 
        'unc':      { 'description': 'Uncertainty',
                      'default': 0.0,
                      'kind': 'f',
                      'valid_kinds': ('f', 'a'),
                      'cast': tools.cast_abs_float_array,
                      'fmt': '0.1e', }, 
        'step':     { 'description': 'Default numerical differentiation step size', 
                      'default': 1e-8, 
                      'kind': 'f', 
                      'valid_kinds': ('f', 'a'), 
                      'cast': tools.cast_abs_float_array,
                      'fmt': '0.1e', }, 
        'vary':     { 'description': 'Whether to vary during optimisation',
                      'default': False,
                      'kind': 'b',
                      'valid_kinds': ('f', 'a'),
                      'cast': tools.convert_to_bool_vector_array,
                      'fmt': '', }, 
        'ref':      { 'description': 'Source reference',
                      'default': nan,
                      'kind': 'U',
                      'cast': lambda x: np.array(x, dtype='U20'),
                      'fmt': 's', }, 
        'error':    { 'description': 'Residual error',
                      'default': nan,
                      'kind': 'f',
                      'valid_kinds': ('f', 'a'),
                      'cast': lambda x: np.array(x, dtype=float),
                      'fmt': '+0.12e',}, 
    }
    assert np.all(['default' in val or key == 'value' for key,val in vector_subkinds.items()])

    ## Kinds of subdata associatd with a key that are single valued
    ## but objects.
    scalar_subkinds = {
        'infer'          : {'description':'List of infer functions',},
        'kind'           : {'description':'Kind of data in value corresponding to a key in data_kinds',},
        'cast'           : {'description':'Vectorised function to cast data',},
        'fmt'            : {'description':'Format string for printing',},
        'description'    : {'description':'Description of data',},
        'units'          : {'description':'Units of data'},
        'default'        : {'description':'Default value',},
        'default_step'   : {'description':'Default differentiation step size','valid_kinds':('f',)},
        '_inferred_to'   : {'description':'List of keys inferred from this data',},
        '_inferred_from' : {'description':'List of keys and the function used to infer this data',},
        '_modify_time'   : {'description':'When this data was last modified',},
    }

    ## all subdata kinds in one convenient dictionary
    all_subkinds = vector_subkinds | scalar_subkinds

    ## prototypes that will automatically set on instantiation
    default_prototypes = {}
    default_permit_nonprototyped_data = False

    ## used for plotting and sorting the data
    default_zkeys = ()
    default_zlabel_format_function = tools.dict_to_kwargs


    ## functions that load specific filetypes, used by
    ## dataset.load. The first input is a filename and any other
    ## kwargs passed to dataset.load (except thoseused by load itself
    ## or load_data_dict) are passed on.
    default_load_functions = {
        'hdf5'         : _load_from_hdf5,
        'directory'    : _load_from_directory,
        'npz'          : _load_from_npz,
        'org'          : _load_from_org,
        'text'         : lambda *args,**kwargs : _load_from_text(*args,**({'delimiter' : ' '}|kwargs)),
        'rs'           : lambda *args,**kwargs : _load_from_text(*args,**({'delimiter' : '␞'}|kwargs)),
        'psv'          : lambda *args,**kwargs : _load_from_text(*args,**({'delimiter' : '|'}|kwargs)),
        'csv'          : lambda *args,**kwargs : _load_from_text(*args,**({'delimiter' : ','}|kwargs)),
        'sqlite'       : _load_from_sqlite,
        'simple_text'  : _load_from_simple_text,
    }


    def __init__(
            self,
            name=None,          # name of this Dataset
            permit_nonprototyped_data = True, # if False then raise an error on the setting of any data that has no prototype
            permit_indexing = True, # if False then raise an error on any operation that unsets or indexes any explicitly set data
            permit_dereferencing = True, # if False raise an error on any operation that reallocates an array -- causes inferred data to be recomputed in place where it would otherwise be deleted
            prototypes = None,      # a dictionary of prototypes
            load_from_file = None,  # load from a file -- guess type
            load_from_string = None, # load from formatted string
            load_from_parameters_dict = None, # load from a dictionary of dictionaries
            copy_from = None,        # copy form another dataset
            limit_to_match=None,     # reduce by this match dictionary after any data is loaded
            description='',          # description of this Dataset
            data=None,               # load keys_vals into self, or set with set_value if they are Parameters
            attributes=None,  # keys and values of this dictionary are stored as attributes
            **data_kwargs,           # data to add from keyword arguments
    ):
        ## generate an encoded classname to identify type of Dataset,
        ## default name from this
        self.classname = re.sub(r"<class 'spectr.(.+)'>", r'\1', str(self.__class__))
        if name is None:
            # name = self.classname.lower()
            name = self.classname
        ## init as optimiser, make a custom form_input_function, save
        optimise.Optimiser.__init__(self,name=name)
        ## basic internal variables
        self._data = {} # table data and its properties stored here
        self._length = 0    # length of data
        self._over_allocate_factor = 2 # to speed up appending to data
        self.description = description
        self._row_modify_time = np.array([],dtype=float,ndmin=1) # record modification time of each explicitly set row
        self._global_modify_time = timestamp() # record modification time of any explicit change
        self.default_ykeys = None             # used by plot
        self.default_xkeys = None             # used by plot
        self.load_functions = copy(self.default_load_functions)
        ## whether to allow the addition of keys not found in
        ## prototypes
        if permit_nonprototyped_data is None:
            self.permit_nonprototyped_data = self.default_permit_nonprototyped_data
        else:
            self.permit_nonprototyped_data = permit_nonprototyped_data
        ## If permit_indexing=False then data can be added to the end
        ## of arrays and new keys added but explicitly set keys can
        ## not be removed and the Dataset can not be internally
        ## indexed.
        self.permit_indexing = permit_indexing 
        ## If permit_dereferencing=False then it is not allowed to
        ## unset a key or replace internal arrays with new ones. This
        ## will raise various errors.  It also changes the behaviour
        ## when dependent data is changed, rather than deleting
        ## inferred data (which can be recalculated later if needed)
        ## this is instead recomputed immediately and stored in place.
        self.permit_dereferencing = permit_dereferencing
        ## print extra information at various places
        self.verbose = False 
        ## get prototypes from defaults and then input argument
        self.prototypes = copy(self.default_prototypes)
        if prototypes is not None:
            self.prototypes |= prototypes
        ## dictionary to store global attributes
        self.attributes = {}
        if attributes is not None:
            self.attributes |= attributes
        ## new format input function -- INCOMPLETE
        def format_input_function():
            retval = f'{self.name} = {self.classname}({repr(self.name)},'
            if load_from_file is not None:
                retval += f'load_from_file={repr(load_from_file)},'
            if len(data_kwargs)>0:
                retval += '\n'
            for key,val in data_kwargs.items():
                retval += f'    {key}={repr(val)},\n'
            retval += ')'
            return retval
        self.add_format_input_function(format_input_function)
        ## save self to directory
        # def save_to_directory_function(self,directory):
            # """Save data in directory as a directory, also save as psv
            # file if data is not too much."""
            # self.save(f'{directory}/data',filetype='directory')
            # if len(self)*len(self.keys()) < 10000:
                # self.save(f'{directory}/data.psv')
        self.add_save_to_directory_function(
            lambda directory: self.save(f'{directory}/data',filetype='directory'))
        ## copy data from another dataset provided as argument
        if copy_from is not None:
            self.copy_from(copy_from)
        ## load data from a filename provide as an argument
        if load_from_file is not None:
            self.load(load_from_file)
        ## load data from an encoded string provided as an argument
        if load_from_string is not None:
            self.load_from_string(load_from_string)
        ## A dictionary of dictionaires. Keys become "key" and each
        ## dictionaries contents are flattened and concatenated
        ## together
        if load_from_parameters_dict is not None:
            self.load_from_parameters_dict(load_from_parameters_dict)
        ## load input data from data dictionary or keyword arguments
        if data is None:
            data = {}
        data |= data_kwargs
        for key,val in data.items():
            if isinstance(val,optimise.Parameter):
                ## an optimisable parameter (input function already
                ## handled)
                self.set_value(key,val)
                self.pop_format_input_function()
            else:
                ## set data
                self[key] = val
        ## limit to matching data
        if limit_to_match is not None:
            self.limit_to_match(**limit_to_match)


    def __len__(self):
        return self._length

    def set(
            self,
            key,                # "key" or ("key","subkey")
            subkey,
            value,
            index=None,         # set these indices only
            match=None,         # set these matches only
            set_changed_only=False, # only set data if it differs from value
            kind=None,
            **match_kwargs
    ):
        """Set value of key or (key,subkey)"""
        ## check for invalid key
        forbidden_character_regexp = r'.*([\'"=#,:]).*' 
        if r:=re.match(forbidden_character_regexp,key):
            raise Exception(f"Forbidden character {repr(r.group(1))} in key {repr(key)}. Forbidden regexp: {repr(forbidden_character_regexp)}")
        ## make array copies
        value = copy(value)
        ## scalar subkind — set and return, not cast
        if subkey in self.scalar_subkinds:
            self._data[key][subkey] = value
        elif subkey in self.vector_subkinds:
            ## combine indices -- might need to sort value if an index array is given
            combined_index = self.get_combined_index(index,match,**match_kwargs)
            ## reduce index and value to changed data only. If nothing
            ## has changed return without setting anything.
            if set_changed_only and self.is_set(key,subkey):
                index_changed = self[key,subkey,combined_index] != value
                if combined_index is None:
                    combined_index = tools.find(index_changed)
                else:
                    combined_index = combined_index[index_changed]
                if len(combined_index) == 0:
                    return
                if tools.isiterable(value):
                    value = np.array(value)[index_changed]
            ## set value or other subdata
            if subkey == 'value':
                self._set_value(key,value,combined_index,kind=kind)
            else:
                # if not self.is_explicit(key):
                    # raise Exception(f'Setting {subkey=} for non-explicitly set {key=}, to avoid bugs this should resul in this key becoming explicitly set, but this is not implemented')
                self._set_subdata(key,subkey,value,combined_index)
        else:
            raise Exception(f'Invalid subkey: {repr(subkey)}')
    
    def _guess_kind_from_value(self,value):
        """Guess what kind of data this is from a provided scalar or
        vector value."""
        dtype = np.asarray(value).dtype
        kind = dtype.kind
        return kind

    def set_new(self,key,value,kind=None,**other_metadata):
        """Set key to value along with other kinds of
        metadata. Creates a prototype first."""
        if key in self:
            raise Exception(f"set_new but key already exists: {repr(key)}")
        if key in self.prototypes:
            raise Exception(f"set_new but key already in prototypes: {repr(key)}")
        if kind is None:
            kind = self._guess_kind_from_value(value)
        self.set_prototype(key,kind=kind,**other_metadata)
        self.set(key,'value',value)

    def set_default(self,key,val,kind=None):
        """Set a default value if not already set."""
        if key not in self:
            self.set_new(key,val,kind)
        self[key,'default'] = val

    def set_prototype(self,key,kind,**kwargs):
        """Set a prototype."""
        if kind is str:
            kind = 'U'
        elif kind is float:
            kind = 'f'
        elif kind is bool:
            kind = 'b'
        elif kind is object:
            kind = 'O'
        self.prototypes[key] = dict(kind=kind,**kwargs)
        for tkey,tval in self.data_kinds[kind].items():
            self.prototypes[key].setdefault(tkey,tval)

    @optimise_method(format_multi_line=99)
    def set_value(
            self,
            key,
            value,
            default=None,
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Set a value and it will be updated every construction and may be a
        Parameter for optimisation."""
        ## cache matching indices
        if self._clean_construct:
            ## set a default value if key is not currently known
            if not self.is_known(key) and default is not None:
                self[key] = default
            self.permit_indexing = False
            _cache['index'] = self.get_combined_index(**get_combined_index_kwargs)
        index = _cache['index']
        ## set the data if it has changed
        if isinstance(value,Parameter):
            ## set from a Parameter, update if Parameter or self has changed
            if (
                    self._clean_construct
                    or value._last_modify_value_time > self[key,'_modify_time']
                    or self[key,'_modify_time'] > self._last_construct_time
            ):
                self.set(key,'value',value,index=index,set_changed_only=True)
                self.set(key,'unc' ,value.unc ,index=index,set_changed_only=True)
        else:
            ## constant value, set if self has changed
            if (
                    self._clean_construct
                    or self[key,'_modify_time'] > self._last_construct_time
                ): 
                self.set(key,'value',value,index=index,set_changed_only=True)
                if self.is_set(key,'unc'):
                    self.set(key,'unc',nan,index=index,set_changed_only=True)

    def auto_set_matching_values(
            self,               
            parameters,         # keys to vary
            combinations,       # keys to find unique combinations of
            limit_to_match_kwargs=None, # if specified then only set matching data
    ):
        """Set values to groups of data with unique combinations of
        combinations_parameters. Values are Parameters or will be cast
        into parameters. If they are an empty list then their initial
        value is taking from the first datum in each combination.
        Note that limit_to_match_kwargs is used to find combinations,
        but if any elements are found then all data (including outside
        limit_to_match_kwargs) will then be set."""
        ## get unique combinations of unique_combination_keys for data
        ## conformign to limit_to_match_kwargs
        matches = self.matches(
            **limit_to_match_kwargs).unique_dicts(
                *tools.ensure_iterable(combinations))
        ## initialise each parameter
        combinations_parameters = [
            match | {key:Parameter(
                        value=self[key,self.match(**match)][0],
                        vary=True,
                        step=(self[key,'default_step'] if self.is_set(key,'default_step') else None),)
                      for key in parameters}
                for match in matches]
        ## pass to set_matching_values
        self.set_matching_values(*combinations_parameters)
        
    def set_matching_values(self,*combinations_parameters):
        """Find lines matching a combination of keys=vals and set the
        values of multiple keys=Parameter values."""
        ## find indices matching the input combination dictionaries
        combinations,parameters = [],[]
        for t in combinations_parameters:
            t_parameters,t_combinations = {},{}
            for key,val in t.items():
                if isinstance(val,Parameter):
                    t_parameters[key] = val
                else:
                    t_combinations[key] = val
            combinations.append(t_combinations)
            parameters.append(t_parameters)
        i = [self.match(**combination) for combination in combinations]
        ## add all Parameters in the input parameters dictionaries
        for parameter in parameters:
            for p in parameter.values():
                self.add_parameter(p)
        ## add a construct function that insers the parameters into
        ## the data in self
        def construct_function():
            for ii,parameter in zip(i,parameters):
                for key,p in parameter.items():
                    self[key,ii] = p
                    self[key,'unc',ii] = p.unc
        self.add_construct_function(construct_function)
        ## a new input line including optimised Parameters
        self.add_format_input_function(
            lambda: '\n'.join([f'{self.name}.set_matching_values(']
                              +['   dict('+','.join(
                                  [f'{key}={val!r}'
                                   for key,val in (tc|tp).items()])+',),'
                                for tc,tp in zip(combinations,parameters)]
                              +[')']))
            
    @optimise_method(format_multi_line=3)
    def set_spline(
            self,
            xkey,
            ykey,
            knots,
            order=3,
            default=None,
            _cache=None,
            optimise=True,
            **get_combined_index_kwargs
    ):
        """Set ykey to spline function of xkey defined by knots at
        [(x0,y0),(x1,y1),(x2,y2),...]. If index or a match dictionary
        given, then only set these."""
        ## To do: cache values or match results so only update if
        ## knots or match values have changed
        if len(_cache) == 0: 
            ## set data
            if not self.is_known(ykey):
                if default is None:
                    raise Exception(f'Setting {repr(ykey)} to spline but it is not known and no default value is provided')
                else:
                    self[ykey] = default
            xspline,yspline = zip(*knots)
            ## get index limit to defined xkey range
            self.permit_indexing = False
            index = self.get_combined_index(**get_combined_index_kwargs)
            if index is None:
                index = slice(0,len(self))
            _cache['index'] = index
            _cache['xspline'],_cache['yspline'] = xspline,yspline
        ## get cached data 
        index,xspline,yspline = _cache['index'],_cache['xspline'],_cache['yspline']
        ## set value to a newly-calculated spline — could include
        ## code to only update value if necessary
        value = tools.spline(xspline,yspline,self[xkey,index],order=order)
        self.set(ykey,'value',value=value,index=index)
        ## set previously-set uncertainties to NaN
        if self.is_set(ykey,'unc'):
            self.set(ykey,'unc',nan,index=index)

    @optimise_method(format_multi_line=3)
    def add_spline(
            self,
            xkey,
            ykey,
            knots,
            order=3,
            default=None,
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Compute a spline function of xkey defined by knots at
        [(x0,y0),(x1,y1),(x2,y2),...] and add to current value of
        ykey. If index or a match dictionary given, then only set
        these."""
        ## To do: cache values or match results so only update if
        ## knots or match values have changed
        if len(_cache) == 0:
            self.assert_known(xkey)
            self.assert_known(ykey)
            xspline,yspline = zip(*knots)
            ## get index limit to defined xkey range
            get_combined_index_kwargs |= {f'min_{xkey}':np.min(xspline),f'max_{xkey}':np.max(xspline)}
            _cache['index'] = self.get_combined_index(**get_combined_index_kwargs)
            _cache['xspline'],_cache['yspline'] = xspline,yspline
            self.permit_indexing = False
        ## get cached data
        index,xspline,yspline = _cache['index'],_cache['xspline'],_cache['yspline']
        ## add to ykey
        ynew = self.get(ykey,'value',index=index)
        yspline = tools.spline(xspline,yspline,self.get(xkey,index=index),order=order)
        self.set(ykey,'value',value=ynew+yspline,index=index)
        ## set previously-set uncertainties to NaN
        if self.is_set(ykey,'unc'):
            self.set(ykey,'unc',nan,index=index)
        ## set vary to False if set, but only on the first execution
        ## (for some reason?!?) 
        if 'not_first_execution' not in _cache:
            if 'vary' in self._data[ykey]:
                self.set(ykey,'vary',False,index=index)
            _cache['not_first_execution'] = True

    @optimise_method(format_multi_line=3)
    def multiply(
            self,
            key,                # key to multiply
            factor,             # factor to multiply by (optimisable)
            from_original_value=False,        # if true then multiply original value on method call during optimisation, else multiply whatever is currenlty there
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Scale key by optimisable factor."""
        ## get index of values to adjsut
        if self._clean_construct:
            index = self.get_combined_index(**get_combined_index_kwargs)
            if index is None:
                index = slice(0,len(self))
            _cache['index'] = index
            if from_original_value:
                original_value = self[key,index]
                _cache['original_value'] = original_value
            self.permit_indexing = False
        ## multiply value
        index = _cache['index']
        if from_original_value:
            value = _cache['original_value']*factor
        else:
            value = self.get(key,'value',index=index)*factor
        self.set(key,'value',value=value,index=index)
        ## not sure how to handle uncertainty -- unset it for now
        self.unset(key,'unc')

    def _increase_char_length_if_necessary(self,key,subkey,new_data):
        """reallocate with increased unicode dtype length if new
        strings are longer than the current array dtype"""
        ## test if (key,subkey is set actually a string data
        if (self.is_set(key,subkey)
            and ((subkey == 'value' and self[key,'kind'] == 'U')
                 or (key in self.vector_subkinds and self.vector_subkinds(subkey,'kind')=='U'))):
            old_data = self[key,subkey]
            ## this is a really hacky way to get the length of string in a numpy array!!!
            old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(old_data.dtype)))
            new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(new_data).dtype)))
            ## reallocate array with new dtype with overallocation if necessary
            if new_str_len > old_str_len:
                if not self.permit_dereferencing:
                    raise Exception(f'Cannot increase U-array string length: {self.permit_dereferencing=}')
                t = np.empty(len(self)*self._over_allocate_factor,
                             dtype=f'<U{new_str_len*self._over_allocate_factor}')
                t[:len(self)] = old_data
                self._data[key][subkey] = t

    def _set_value(self,key,value,index=None,inferred_from=None,kind=None):
        """Set a value"""
        ## turn Parameter into its floating point value
        if isinstance(value,Parameter):
            value = float(value)
        ## If an index is provided then data must already exist, set
        ## new indexed data.  If the key is set and len(self)==0 then
        ## that that implies only a default value is set, so skip this
        ## block and instaed create an entirely new array below - but
        ## erasing default!
        if (
                index is not None
                or ( self.is_set(key) and len(self) > 0)
        ):
            ## set index to all data if not provided
            if index is None and self.is_set(key):
                index = slice(0,len(self)) 
            if not self.is_known(key):
                raise Exception(f'Could not set unknown key {key!r} with index')
                raise Exception(f'Cannot set data by index for unknown key: {key}')
            ## reallocate string arrays if needed
            self._increase_char_length_if_necessary(key,'value',value)
            ## scalar to array so that casdt does not fail
            if np.isscalar(value):
                value = array([value])
            ## set indexed data
            self._data[key]['value'][:self._length][index] = self[key,'cast'](value)
        else:
            ## create entire data dict
            if not self.permit_nonprototyped_data and key not in self.prototypes:
                raise Exception(f'New data is not in prototypes: {repr(key)}')
            ## new data
            data = {'infer':[],'_inferred_to':[]}
            ## get any prototype data
            if key in self.prototypes:
                for tkey,tval in self.prototypes[key].items():
                    data[tkey] = tval
            ## object kind not implemented
            if 'kind' in data and data['kind'] == 'O':
                raise NotImplementedError()
            ## use data to infer kind if necessary
            if kind is not None:
                data['kind'] = kind
            if 'kind' not in data:
                value = np.array(value)
                data['kind'] = self._guess_kind_from_value(value)
            ## convert bytes string to unicode
            if data['kind'] == 'S':
                data['kind'] = 'U'
            ## some other prototype data based on kind
            data = self.data_kinds[data['kind']] | data
            ## If this is a scalar value then expand it to the length
            ## of self and also set as default value.  But if there is
            ## no data at all in self then raise an error, otherwise a
            ## zero-length self with be created.
            if not tools.isiterable(value):
                #  if len(self.keys()) == 0:
                    #  raise Exception(f'It is not implemented to set a scalar value ({key=}, {value=}) in a Dataset containing new data because this will create an awkward zero-length Dataset.')
                data['default'] = value
                value = np.full(len(self),value)
            ## if this is the first data then allocate an initial
            ## length to match
            if len(self) == 0:
                self._reallocate(len(value))
                ## If this is the first nonzero-length data set then increase
                ## length of self and set any other keys with defaults to
                ## their default values
                if len(value) > 0:
                    for tkey,tdata in self._data.items():
                        if tkey == key:
                            continue
                        if 'default' in tdata:
                            tdata['value'] = tdata['cast'](np.full(len(self),tdata['default']))
                        else:
                            raise Exception(f'Need default for key {tkey}')
            if len(value) != len(self):
                raise Exception(f'Length of new data {repr(key)} is {len(value)} and does not match the length of existing data: {len(self)}.')
            ## cast and set data
            data['value'] = data['cast'](value)
            ## add to self
            self._data[key] = data
        ## If key is already set then delete or recompute anything
        ## previously inferred from it, and if it is inferred from
        ## something else then delete the record of it bcause it is
        ## now explicitly set.
        if key in self:
            self.unlink_inferences(
                key,
                recompute_inferences=(not self.permit_dereferencing))
        ## If the new data is inferred data then record dependencies
        if inferred_from is not None:
            self[key,'_inferred_from'] = inferred_from
            for dependency in inferred_from[0]:
                self[dependency,'_inferred_to'].append(key)
        ## Record key, global, row modify times if this is an explicit
        ## change.
        tstamp = timestamp()
        if inferred_from is None or len(inferred_from[0]) == 0:
            ## not inferred, or inferred from thin air, gets its own
            ## up-to-date timestamp
            self._global_modify_time = tstamp
            if index is None:
                self._row_modify_time[:self._length] = tstamp
            else:
                self._row_modify_time[:self._length][index] = tstamp
            self[key,'_modify_time'] = tstamp
        else:
            ## If inferred data record modification time of the most
            ## recently modified dependency
            self[key,'_modify_time'] = max(
                [self[tkey,'_modify_time'] for tkey in inferred_from[0]])

    def _set_subdata(self,key,subkey,value,index=None):
        """Set vector subdata."""
        subkind = self.vector_subkinds[subkey]
        if not self.is_known(key):
            raise Exception(f"Value of key {repr(key)} must be set before setting subkey {repr(subkey)}")
        data = self._data[key]
        if ('valid_kinds' in subkind and self.get_kind(key) not in subkind['valid_kinds']):
            raise Exception(f"The value kind of {repr(key)} is {repr(data['kind'])} and is invalid for setting {repr(subkey)}")
        if subkind['kind'] == 'O':
            raise ImplementationError()
        ## set data
        if index is None and not self.is_set(key,subkey):
            ## set entire array
            if not tools.isiterable(value):
                ## expand scalar input
                value = np.full(len(self),value)
            elif len(value) != len(self):
                raise Exception(f'Length of new subdata {repr(subkey)} for key {repr(key)} ({len(value)} does not match existing data length ({len(self)})')
            ## set data
            data[subkey] = subkind['cast'](value)
        else:
            if index is None:
                self.assert_known(key,subkey)
                index = slice(0,len(self))
            ## set part of array by index
            if subkey not in data:
                ## set missing data outside indexed range to a default
                ## value using the get method
                self.get(key,subkey)
            ## reallocate string arrays if needed
            self._increase_char_length_if_necessary(key,subkey,value)
            ## set indexed data
            data[subkey][:len(self)][index] = subkind['cast'](value)

    row_modify_time = property(lambda self:self._row_modify_time[:self._length])
    global_modify_time = property(lambda self:self._global_modify_time)

    def get(
            self,
            key,
            subkey='value',
            index=None,
            units=None,
            match=None,
            **match_kwargs
    ):
        """Get value for key or (key,subkey). This is the data in place, not a
        copy."""
        index = self.get_combined_index(index,match,**match_kwargs)
        ## ensure data is known
        if key not in self._data:
            try:
                ## attempt to infer
                self._infer(key)
            except InferException as err:
                if key in self.prototypes and 'default' in self.prototypes[key]:
                    self[key] = self.prototypes[key]['default']
                else:
                    raise err
        ## get relevant data
        data = self._data[key]
        ## check subdata exists
        subkind = self.all_subkinds[subkey] 
        ## test that this subkind is valid
        if 'valid_kinds' in subkind and data['kind'] not in subkind['valid_kinds']:
            raise Exception(f"Key {repr(key)} of kind {data['kind']} is not a valid kind for subdata {repr(subkey)}")
        ## if data is not set then set default if possible
        if not self.is_set(key,subkey):
            assert subkey != 'value','should be inferred above'
            if subkey == 'step' and 'default_step' in data:
                ## special shortcut case for specied default step
                self.set(key,subkey,data['default_step'])
            elif 'default' in subkind:
                self.set(key,subkey,subkind['default'])
            else:
                raise InferException(f'Could not determine default value for subkey {repr(subkey)})')
        ## return data
        if subkey in self.scalar_subkinds:
            ## scalar subdata
            return data[subkey]
        elif subkey in self.vector_subkinds:
            ## return indexed data
            retval = data[subkey][:len(self)]
            if index is not None:
                retval = retval[index]
            if units is not None:
                retval = convert.units(retval,data['units'],units)
            return retval
        else:
            raise Exception(f'Invalid subkey: {repr(subkey)}')

    def get_kind(self,key):
        return self._data[key]['kind']

    def get_default(self,key,subkey='value'):
        if subkey == 'value' and self.is_set(key,'default'):
            return self[key,'default']
        elif 'default' in self.all_subkinds[subkey]:
            return self.all_subkinds[subkey]['default']
        else:
            raise Exception(f'Could not determine default value for {key=} {subkey=}')
        
    def modify_key(self,key,rename=None,**new_metadata):
        """Modify metadata of a key, change its units, or rename it."""
        if not self.permit_dereferencing:
            raise Exception(f'Cannot modify {key!r}: {self.permit_dereferencing=} (could loosen this)')
        self.assert_known(key)
        ## rename key by adding a new one and unsetting the original --
        ## breaking inferences
        if rename is not None:
            for subkey in self.all_subkinds:
                if self.is_set(key,subkey):
                    self[rename,subkey] = self[key,subkey]
            self.unset(key)
            key = rename
        for subkey,val in new_metadata.items():
            if subkey == 'description':
                self[key,subkey] = val
            elif subkey == 'kind':
                if self[key,subkey] != val:
                    new_data = {subkey:self[key,subkey] for subkey in all_subkinds if self.is_set(key,subkey)}
                    new_data['kind'] = val
                    self.unset(key)
                    self.set_new(key,**new_data)
            elif subkey == 'units':
                if self.is_set(key,'units'):
                    ## convert units of selected subkeys, convert all
                    ## first before setting
                    new_data = {}
                    for tsubkey in ('value','unc','step',):
                        if self.is_set(key,tsubkey):
                            new_data[tsubkey] = convert.units(self[key,tsubkey],self[key,'units'],val)
                    for tsubkey,tval in new_data.items():
                            self[key,tsubkey] = tval
                self[key,'units'] = val
            elif subkey == 'fmt':
                self[key,'fmt'] = val
            else:
                raise NotImplementedError(f'Modify {subkey}')

    def _has_subkey_attribute_property(self,key,subkey,attribute):
        """Test if key,subkey has a certain attribute."""
        self.assert_known(key,subkey)
        if subkey == 'value':
            return (attribute in self._data[key])
        else:
            return (attribute in self.all_subkinds[subkey])

    def _get_subkey_attribute_property(self,key,subkey,attribute):
        """Get data from data_kinds or all_subkinds"""
        self.assert_known(key,subkey)
        if subkey == 'value':
            return self[key,attribute]
        else:
            return self.all_subkinds[subkey][attribute]
            
    def get_combined_index(
            self,
            index=None,
            match=None,
            return_as='int or None', # 'int or None', 'bool', 'int'
            **match_kwargs):
        """Combined specified index with match arguments as integer
        array. If no data given the return None"""
        ## combine match dictionaries
        if match is None:
            match = {}
        if match_kwargs is not None:
            match |= match_kwargs
        if index is None and len(match) == 0:
            ## no indices at all
            retval = None
        elif np.isscalar(index):
            ## single index
            retval = index
            if len(match) > 0:
                raise Exception("Single index cannot be addtionally matched.")
        else:
            ## get index array
            if index is None:
               retval = np.arange(len(self))
            elif isinstance(index,slice):
                ## slice
                retval = np.arange(len(self))
                retval = retval[index]
            else:
                retval = np.array(index,ndmin=1)
                if retval.dtype == bool:
                    retval = tools.find(retval)
            ## reduce by matches if given
            if len(match) > 0:
                imatch = self.match(match)
                retval = retval[tools.find(imatch[retval])]
        ## convert to a boolean array
        if return_as == 'int or None':
            ## return as None if no input restrictions or an array of
            ## integers of matching data
            pass
        elif return_as == 'bool':
            ## return as an array of boolean values
            index = retval
            if index is None:
                retval = np.full(len(self),True)
            else:
                retval = np.full(len(self),False)
                retval[index] = True
        elif return_as == 'int':
            ## return as an array of integers
            if retval is None:
                retval = np.arange(len(self))
        return retval
            
    # def get_combined_index_bool(self,*get_combined_index_args,**get_combined_index_kwargs):
        # """Combined specified index with match arguments as integer array. If
        # no data given the return None"""
        # raise DeprecationWarning('use get_combined_index(return_as="bool") instead')
        # index = self.get_combined_index(*get_combined_index_args,**get_combined_index_kwargs)
        # if index is None:
            # raise Exception('Cannot return bool array combined index if None.')
        # if np.isscalar(index):
            # raise Exception("Cannot return bool array for Single index.")
        # ## convert to boolean array
        # retval = np.full(len(self),False)
        # retval[index] = True
        # return retval

    def keys(self):
        return list(self._data.keys())

    def sort_keys(self,keys):
        """Keys are sorted into the given. If any keys are not listed in the
        input argument `keys' they are sorted automatically."""
        ## transfer keys form internal data dict to temporary dict in
        ## sorted order
        sorted_data = {}
        for key in keys:
            self.assert_known(key)
            sorted_data[key] = self._data.pop(key)
        other_keys = sorted(self._data.keys())
        for key in other_keys:
            sorted_data[key] = self._data.pop(key)
        ## transfer back to internal data dict
        self._data |= sorted_data

    def limit_to_keys(self,keys):
        """Unset all keys except these."""
        keys = tools.ensure_iterable(keys)
        for key in keys:
            self.assert_known(key)
        self.unlink_inferences(keys)
        for key in list(self):
            if key not in keys:
                self.unset(key)

    def optimised_keys(self):
        """Return a list of keys that will be considered for
        optimisation."""
        return [key for key in self.keys() if self.is_set(key,'vary')]

    def explicitly_set_keys(self):
        """Return a list of keys that were not inferred."""
        return [key for key in self if not self.is_inferred(key)]

    def known_keys(self):
        """Return a list of all known keys."""
        retval = [key for key in {*self.keys()}|{*self.prototypes.keys()}
                  if self.is_known(key)]
        return retval 

    def match_keys(self,regexp=None,beg=None,end=None,):
        """Return a list of keys matching any of regex or
        beginning/ending string beg/end."""
        keys = []
        if regexp is not None:
            keys += [key for key in self if re.match(regexp,key)]
        if beg is not None:
            keys += [key for key in self if len(key)>=len(beg) and key[:len(beg)] == beg]
        if end is not None:
            keys += [key for key in self if len(key)>=len(end) and key[-len(end):] == end]
        return keys
            
    def match_keys_matches(self,regexp):
        """Return a list of keys matching a regexp along with any
        matched groups. NEED THIS?"""
        retval = []
        for key in self:
            if r:=re.match(regexp,key):
                retval.append((key,*r.groups()))
        return retval 

    def __iter__(self):
        """Iterate through keys."""
        for key in self._data:
            yield key

    def items(self):
        """Iterate over keys and their values."""
        for key in self:
            yield key,self[key]


    def is_set(self,key,subkey='value'):
        """Test if (key,subkey) is currently defined."""
        if key in self._data and subkey in self._data[key]:
            return True
        else:
            return False

    def assert_known(self,key,subkey='value'):
        """Raise error if (key,subkey) is not set and cannot be
        inferred."""
        self[key,subkey]

    def is_known(self,key,subkey='value'):
        """Test if (key,subkey) is currently defined, try and infer it
        if necessary."""
        try:
            self.assert_known(key,subkey)
            return True 
        except InferException:
            return False
            
    def __getitem__(self,arg):
        """Possible forms for arg:
            key                -- return value of this key
            integer            -- return this row of data as a dict
            slice              -- return copy of self indexed by this
            int/bool array     -- return copy of self indexed by this
            [key0,key1...]     -- return copy of self restricted to these keys
            (key,subkey)       -- return subkey of this key
            (key,index)        -- return value of this key for this index
            (key,subkey,index) -- return subkey of this key for this index
        """
        if isinstance(arg,str):
            ## a non indexed key
            return self.get(key=arg)
        elif isinstance(arg,(int,np.int64)):
            ## single indexed row
            return self.row(index=arg)
        elif isinstance(arg,slice):
            ## index by slice
            return self.copy(index=arg)
        elif isinstance(arg,np.ndarray):
            ## an index array
            return self.copy(index=arg)
        elif not tools.isiterable(arg):
            ## no more single valued arguments defined
            raise Exception(f"Cannot interpret getitem argument: {repr(arg)}")
        elif isinstance(arg[0],(int,np.int64,bool)):
            ## index
            return self.copy(index=arg)
        elif len(arg) == 1:
            if isinstance(arg[0],str):
                ## copy with keys
                return self.copy(keys=arg)
            else:
                ## copy with index
                return self.copy(index=arg[0])
        elif len(arg) == 2:
            if isinstance(arg[0],str):
                if isinstance(arg[1],str):
                    if arg[1] in self.all_subkinds:
                        ## return key,subkey
                        return self.get(key=arg[0],subkey=arg[1])
                    else:
                        ## copy keys
                        return self.copy(keys=arg)
                else:
                    ## return key,index
                    return self.get(key=arg[0],index=arg[1])
            else:
                ## copy with keys and index
                return self.copy(keys=arg[0],index=arg[1])
        elif len(arg) == 3:
            if arg[1] in self.all_subkinds:
                ## get key,subkey,index
                return self.get(key=arg[0],subkey=arg[1],index=arg[2])
            else:
                ## copy keys
                return self.copy(keys=arg)
        else:
            ## all args are keys for copying
            return self.copy(keys=arg)

    def __setitem__(self,key,value):
        """Set key, (key,subkey), (key,index), (key,subkey,index) to
        value."""
        if isinstance(key,str):
            tkey,tsubkey,tindex = key,'value',None
        elif len(key) == 1:
            tkey,tsubkey,tindex = key[0],'value',None
        elif len(key) == 2:
            if isinstance(key[1],str):
                tkey,tsubkey,tindex = key[0],key[1],None
            else:
                tkey,tsubkey,tindex = key[0],'value',key[1]
        elif len(key) == 3:
                tkey,tsubkey,tindex = key[0],key[1],key[2]
        self.set(tkey,tsubkey,value,tindex)
       
    def clear(self):
        """Clear all data"""
        if not self.permit_indexing:
            raise Exception(f'Cannot clear: {self.permit_indexing=}')
        elif not self.permit_dereferencing:
            raise Exception(f'Cannot clear: {self.permit_dereferencing=}')
        self._last_modify_value_time = timestamp()
        self._length = 0
        self._data.clear()

    def _is_valid_key_subkey(self,key,subkey):
        if subkey not in self.vector_subkinds:
            raise Exception(f'{subkey=} is not a valid vector_subkinds={list(self.vector_subkinds.keys())!r}')

    def unset(self,key,subkey='value'):
        """Delete (key,subkey) data.  Also clean up inferences."""
        if not self.permit_indexing and (subkey == 'value' and not self.is_inferred(key)):
            raise Exception(f'Cannot unset explicit {(key,subkey)!r}: {self.permit_indexing=}')
        elif not self.permit_dereferencing:
            raise Exception(f'Cannot unset {(key,subkey)!r}: {self.permit_dereferencing=}')
        if key in self:
            self._is_valid_key_subkey(key,subkey)
            if subkey == 'value':
                self.unlink_inferences(key)
                self._data.pop(key)
            else:
                data = self._data[key]
                if subkey in data:
                    data.pop(subkey)

    def pop(self,key,subkey='value'):
        """Return (key,subkey) and unset it."""
        if not self.permit_indexing:
            raise Exception(f'Cannot pop {(key,subkey)!r}: {self.permit_indexing=}')
        elif not self.permit_dereferencing:
            raise Exception(f'Cannot pop {(key,subkey)!r}: {self.permit_dereferencing=}')
        retval = self[key,subkey]
        self.unset(key,subkey)
        return retval

    def is_explicit(self,key):
        """True if key is explicitly set. False if inferred data or
        not set."""
        retval = self.is_set(key) and not self.is_inferred(key)
        return retval
    
    def is_inferred(self,key):
        """Test whether this key is inferred."""
        if '_inferred_from' in self._data[key]:
            return True
        else:
            return False
   
    def unset_inferred(self):
        """Delete all inferred data everywhere."""
        for key in list(self):
            if key in self and self.is_inferred(key):
                self.unlink_inferences(key)
                self.unset(key)
   
    def unlink_inferences(self,keys,recompute_inferences=False):
        """Delete any record of keys begin inferred. Also recursively unset
        any key inferred from these and not among keys itself."""
        keys = tools.ensure_iterable(keys)
        for key in keys:
            self.assert_known(key)
        ## no longer marked as inferred from something else
        for key in keys:
            if key not in self:
                continue
            if self.is_inferred(key):
                for tkey in self._data[key]['_inferred_from'][0]:
                    if tkey in self._data:
                        if key in self._data[tkey]['_inferred_to']:
                            self._data[tkey]['_inferred_to'].remove(key)
                self._data[key].pop('_inferred_from')
            ## recursively delete or recompute everything inferred from key
            for tkey in copy(self._data[key]['_inferred_to']):
                if tkey not in keys and tkey in self:
                    if recompute_inferences:
                        ## recompute value and uncertainty and substitute data into the existing arrays
                        dependencies,function = self[tkey,'_inferred_from']
                        value,uncertainty = self._infer_compute(dependencies,function)
                        cast = self[tkey,'cast']
                        self[tkey][:] = value
                        if uncertainty is not None:
                            cast_uncertainty = self.vector_subkinds['unc']['cast']
                            self[tkey,'unc'][:] = uncertainty
                    else:
                        ## unset 
                        self.unset(tkey)

    def unlink_all_inferences(self):
        """Mark all data as not inferred."""
        self.unlink_inferences(self.keys())

    def add_infer_function(self,key,dependencies,function):
        """Add a new method of data inference. DOES NOT SUPPORT
        EXPLICIT UNCERTAINTY FUNCTION."""
        self.prototypes[key]['infer'].append((dependencies,function))

    def index(self,index):
        """Index all array data in place."""
        if not self.permit_indexing:
            raise Exception(f'Cannot index: {self.permit_indexing=}')
        elif not self.permit_dereferencing:
            raise Exception(f'Cannot index: {self.permit_dereferencing=}')
        if isinstance(index,(int,np.int64)):
            ## allow single integer indexing
            index = [index]
        original_length = len(self)
        for key,data in self._data.items():
            for subkey in data:
                if subkey in self.vector_subkinds:
                    data[subkey] = data[subkey][:original_length][index]
            self._length = len(data['value'])

    def remove(self,index):
        """Remove indices."""
        index = self.get_combined_index(index, return_as='bool')
        self.index(~index)

    # ## caused a memory leak somehow. Is it really faster?
    # ## def __deepcopy__(self,memo):
    # ##     """Manually controlled deepcopy which does seem to be faster than the
    # ##     default for some reason. Relies on all mutable attributes
    # ##     being included in attr_to_deepcopy."""
    # ##     retval = copy(self)
    # ##     memo[id(self)] = retval # add this in case of circular references to it below
    # ##     for attr_to_deepcopy in (
    # ##             '_data',
    # ##             '_row_modify_time',
    # ##             'prototypes',
    # ##             'attributes',
    # ##             '_construct_functions',
    # ##             '_post_construct_functions',
    # ##             '_plot_functions',
    # ##             '_monitor_functions',
    # ##             '_save_to_directory_functions',
    # ##             '_format_input_functions',
    # ##             '_suboptimisers',
    # ##             'residual',
    # ##             'combined_residual',
    # ##             'store',
    # ##     ):
    # ##         setattr(retval,attr_to_deepcopy, deepcopy(getattr(self,attr_to_deepcopy), memo))
    # ##     return retval

    def __deepcopy__(self,memo):
        """Manually controlled deepcopy which does seem to be faster than the
        default for some reason. Relies on all mutable attributes
        being included in attr_to_deepcopy."""
        retval = copy(self)
        memo[id(self)] = retval # add this in case of circular references to it below
        for attr_to_deepcopy in (
                '_data',
                '_row_modify_time',
                'prototypes',
                'attributes',
                '_construct_functions',
                '_post_construct_functions',
                '_plot_functions',
                '_monitor_functions',
                '_save_to_directory_functions',
                '_format_input_functions',
                '_suboptimisers',
                'residual',
                'combined_residual',
                'store',
        ):
            setattr(retval, attr_to_deepcopy,
                    deepcopy(getattr(self,attr_to_deepcopy), memo))
        return retval

    # def __deepcopy__(self,memo):
        # """Manually controlled deepcopy."""
        # retval = self.__class__(name='copy_of_self.name') # new version of self
        # retval.pop_format_input_function()
        # retval.prototypes = deepcopy(self.prototypes)
        # retval.attributes = deepcopy(self.attributes)
        # retval.permit_indexing = self.permit_indexing 
        # retval._length = self._length
        # retval._data = deepcopy(self._data)
        # memo[id(self)] = retval # add this in case of circular references to it below
        # return retval

    def copy(self,name=None,*args_copy_from,**kwargs_copy_from):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if name is None:
            name = f'copy_of_{self.name}'
        retval = self.__class__(name=name, description=f'Copy of {self.name}.',)
        retval.copy_from(self,*args_copy_from,**kwargs_copy_from)
        return retval

    @format_input_method(format_multi_line=3)
    def copy_from(
            self,
            source,
            keys=None,
            keys_re=None,
            skip_keys=None,
            subkeys=('value','unc','description','units','fmt'),
            copy_attributes=True,
            copy_prototypes=True,
            copy_inferred_data=False,
            optimise=False,
            **get_combined_index_kwargs
    ):
        """Copy all values and uncertainties from source
        Dataset. Clear all existing data."""
        ## adding suboptimiser causes construct of source
        if optimise:
            self.add_suboptimiser(source)
        ## total data reset
        self.clear()
        ## collect keys to copy
        if keys_re is not None:
            if keys is None:
                keys = []
            else:
                keys = list(keys)
            keys.extend(source.match_keys(regexp=keys_re))
        if keys is None:
            if copy_inferred_data:
                keys = source.keys()
            else:
                keys = source.explicitly_set_keys()
        if skip_keys is not None:
            keys = [key for key in keys if key not in tools.ensure_iterable(skip_keys)]
        if 'value' not in subkeys:
            raise Exception(f"{subkeys=} must contain 'value'")
        ## copy other things
        self.permit_nonprototyped_data = source.permit_nonprototyped_data
        if copy_attributes:
            self.attributes = deepcopy(source.attributes)
        if copy_prototypes:
            self.prototypes = deepcopy(source.prototypes)
        ## get matching indices
        source_index = source.get_combined_index(**get_combined_index_kwargs)
        if source_index is None:
            source_index = slice(0,len(source))
            index = slice(0,len(source))
        else:
            index = slice(0,len(source_index))
        ## copy data
        for key in keys:
            ## value first
            self.set(key,'value',source[key,'value',source_index])
            for subkey in subkeys:
                if subkey == 'value':
                    ## already done
                    continue
                if source.is_set(key,subkey):
                    if subkey in self.vector_subkinds:
                        self.set(key,subkey,source[key,subkey,source_index])
                    else:
                        self.set(key,subkey,source[key,subkey])
        ## set up recopy on optimisation
        if optimise:
            def construct_function():
                ## need to update copied data in self
                for key in keys:
                    if (source[key,'_modify_time'] > self[key,'_modify_time']
                        or self[key,'_modify_time'] > self._last_construct_time):
                        ## this key needs updating
                        for subkey in ('value','unc'):
                            if source.is_set(key,subkey):
                                self.set(key,subkey, 
                                         source[key,subkey,source_index],
                                         index,
                                         set_changed_only=True)
            self.add_construct_function(construct_function)
            ## prevent optimisation-breaking changes
            source.permit_indexing = False
            self.permit_indexing = False

    def match_re(self,keys_vals=None,**kwarg_keys_vals):
        """Match kind='U' data to regular expressions."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals = keys_vals | kwarg_keys_vals
        retval = np.full(len(self),True)
        for key,regexp in keys_vals.items():
            retval &= np.array([re.match(regexp,val) for val in self[key]],dtype=bool)
        return retval

    def match(self,keys_vals=None,**kwarg_keys_vals):
        """Return boolean array of the intersection of matching
        key=val pairs in the dictionary keys_vals and in the form:
            key=value              -- matching values
            key=(value0,value1...) -- matches something in the list
            not_key=...            -- not a match (also e.g., not_range_key is possible)
            min_key=value          -- at least this value
            max_key=value          -- at most this value
            range_key=(min,max)    -- in this range
            re_key=string          -- match to this regular expression
        If key is a (key,vector_subkey) pair then match these.
        """
        ## joint kwargs to keys_vals dict
        if keys_vals is None:
            keys_vals = {}
        keys_vals = keys_vals | kwarg_keys_vals
        ## update match by key/val
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            ## key is either a string or a (key,subkey) pair
            if isinstance(key,str):
                subkey = 'value'
            else:
                key,subkey = key
            if len(key) > 4 and key[:4] == 'not_' and not self.is_known(key,subkey):
                ## negate match
                i &= ~self.match({(key[4:],subkey):val})
            elif not np.isscalar(val):
                if len(key) > 6 and key[:6] == 'range_' and not self.is_known(key,subkey):
                    ## find all values in a the range of a pair
                    if len(val) != 2:
                        raise Exception(r'Invalid range: {val!r}')
                    i &= (self[key[6:],subkey] >= val[0]) & (self[key[6:],subkey] <= val[1])
                else:
                    ## multiple possibilities to match against
                    i &= np.any([self.match({(key,subkey):vali}) for vali in val],0)
            else:
                ## a single value to match against
                if self.is_known(key,subkey):
                    ## a simple equality
                    if val is np.nan:
                        ## special case for equality with nan
                        i &= np.isnan(self[key,subkey])
                    else:
                        ## simple equality
                        i &= self[key,subkey]==val
                elif len(key) > 4 and key[:4] == 'min_':
                    ## find all larger values
                    i &= (self[key[4:],subkey] >= val)
                elif len(key) > 8 and key[:8] == 'nonzero_':
                    ## find all nonzero values
                    i &= (self[key[8:],subkey] != 0)
                elif len(key) > 10 and key[:10] == 'abovezero_':
                    ## find all values greater than zero
                    i &= (self[key[10:],subkey] > 0)
                elif len(key) > 10 and key[:10] == 'belowzero_':
                    ## find all values greater than zero
                    i &= (self[key[10:],subkey] < 0)
                elif len(key) > 4 and key[:4] == 'max_':
                    ## find all smaller values
                    i &= (self[key[4:],subkey] <= val)
                elif len(key) > 3 and key[:3] == 're_':
                    ## recursively get reverse match for this key
                    i &= self.match_re({(key[3:],subkey):val})
                elif len(key) > 5 and key[:5] == 'even_':
                    ## even integer val = True or False
                    assert val in (True,False)
                    if val:
                        i &= self[key[5:]]%2==0
                    else:
                        i &= self[key[5:]]%2==1
                else:
                    ## total failure
                    raise InferException(f'Could not match key: {repr(key)}')
        return i

    def find(self,*match_args,**match_kwargs):
        """Like match but returns an array of integer indices."""
        i = tools.find(self.match(*match_args,**match_kwargs))
        return i

    def matches(self,*args,**kwargs):
        """Returns a copy of self reduced to matching values."""
        return self.copy(index=self.match(*args,**kwargs),copy_inferred_data=False)

    def limit_to_match(self,*match_args,**match_kwargs):
        """Reduces self to matching values."""
        self.index(self.match(*match_args,**match_kwargs))

    def remove_match(self,*match_args,**match_keys_vals):
        """Removes all matching values for self."""
        self.index(~self.match(*match_args,**match_keys_vals))

    def unique(self,key,subkey='value'):
        """Return unique values of one (key,subkey)."""
        self.assert_known(key,subkey)
        if self.get_kind(key) == 'O':
            raise ImplementationError()
            return self[key]
        else:
            return np.unique(self[key,subkey])

    def unique_combinations(self,*keys):
        """Return a list of all unique combination of key values."""
        return tools.unique_combinations(*[self[key] for key in keys])

    def unique_dataset(self,*keys):
        """Return a dataset summarising the unique combination of keys."""
        retval = self.__class__()
        for data in self.unique_dicts(*keys):
            retval.append(**data)
        retval.sort(keys)
        return retval

    def unique_dicts(self,*keys):
        """Return an iterator where each element is a dictionary
        containing a unique combination of keys."""
        if len(keys)==0:
            return ({},)
        retval = [{key:val for key,val in zip(keys,vals)} for vals in self.unique_combinations(*keys)]
        retval = sorted(retval, key=lambda t: [t[key] for key in keys])
        return retval 

    def unique_dicts_match(self,*keys,return_bool=True):
        """Finds unique combinations of keys and returns pairs
        consisting of a dictionary containing this combination and an
        array indexing its matches."""
        retval = []
        for d in self.unique_dicts(*keys):
            if return_bool:
                retval.append((d,self.match(**d)))
            else:
                retval.append((d,self.find(**d)))
        return retval

    def unique_dicts_matches(self,*keys):
        """Finds unique combinations of keys and returns pairs consisting of a
        dictionary containing this combination and copy of limited to
        its matches."""
        retval = []
        for d in self.unique_dicts(*keys):
            retval.append((d,self.matches(**d)))
        return retval
                          
    def _infer(self,key,already_attempted=None,depth=0):
        """Return value of key or attempt to recursively attempting to
        calculate it from existing data.."""
        if key in self:
            return
        ## avoid getting stuck in a cycle
        if already_attempted is None:
            already_attempted = []
        if key in already_attempted:
            raise InferException(f"Already attempted or attempting to infer key: {repr(key)}")
        already_attempted.append(key)
        if key not in self.prototypes:
            raise InferException(f"{self.name}: No prototype for key: {repr(key)}")
        ## loop through possible methods of inferences.
        self.prototypes[key].setdefault('infer',[])
        for dependencies,function in self.prototypes[key]['infer']:
            if isinstance(dependencies,str):
                ## sometimes dependencies end up as a string instead
                ## of a list of strings
                dependencies = (dependencies,)
            if self.verbose:
                print(f'{self.name}:',
                      ''.join(['    ' for t in range(depth)])
                      +f'Attempting to infer {repr(key)} from {repr(dependencies)}')
            try:
                ## infer dependencies, or fail in the attempt, use a
                ## copy of already_attempted so it will not feed back
                ## here
                for dependency in dependencies:
                    self._infer(dependency,copy(already_attempted),depth=depth+1)
                ## actual calculation, might still fail
                value,uncertainty = self._infer_compute(dependencies,function)
                ## success — set values in self
                self._set_value(key,value,inferred_from=(dependencies,function))
                if self.verbose:
                    print(f'{self.name}:',''.join(['    ' for t in range(depth)])+f'Sucessfully inferred: {key!r} from {dependencies!r}')
                if uncertainty is not None:
                    self._set_subdata(key,'unc',uncertainty)
                    if self.verbose:
                        print(f'{self.name}:',
                              ''.join(['    ' for t in range(depth)])
                              +f'Sucessfully inferred uncertainty {key!r}')
                break           
            ## some kind of InferException, try next set of dependencies
            except InferException as err:
                if self.verbose:
                    print(f'{self.name}:',
                          ''.join(['    ' for t in range(depth)])
                          +'    InferException: '+str(err))
                continue     
        else:
            ## could not infer from any dependencies
            if key in self.prototypes and 'default' in self.prototypes[key]:
                ## use default value. Include empty dependencies so
                ## this is not treated as explicitly set data
                default_value = self.prototypes[key]['default']
                if self.verbose:
                    print(f'{self.name}:',''.join(['    ' for t in range(depth)])+f'Cannot infer {repr(key)} and setting to default value: {repr(default_value)}')
                self._set_value(key,default_value,inferred_from=((),()))
            else:
                ## complete failure to infer
                raise InferException(f"Could not infer key: {repr(key)}")

    def _infer_compute(self,dependencies,function):
        """Calculate inferred value and uncertainty from dependencies that are
        already known to be set."""
        ## if function is a tuple of two functions then the second
        ## is for computing uncertainties
        if tools.isiterable(function):
            function,uncertainty_function = function
        else:
            function,uncertainty_function = function,None
        ## inferred.  If value is None then the data and
        ## dependencies are set internally in the infer
        ## function.
        value = function(self,*[self[dependency] for dependency in dependencies])
        ## compute uncertainties by linearisation
        if uncertainty_function is None:
            squared_contribution = []
            parameters = [self[t] for t in dependencies]
            for i,dependency in enumerate(dependencies):
                if self.is_set(dependency,'unc'):
                    step = self.get(dependency,'step')
                    parameters[i] = self[dependency] + step # shift one
                    dvalue = value - function(self,*parameters)
                    parameters[i] = self[dependency] # put it back
                    squared_contribution.append((self.get(dependency,'unc')*dvalue/step)**2)
            if len(squared_contribution)>0:
                uncertainty = np.sqrt(np.sum(squared_contribution,axis=0))
            else:
                uncertainty = None
        else:
            ## args for uncertainty_function.  First is the
            ## result of calculating keys, after that paris of
            ## dependencies and their uncertainties, if they
            ## have no uncertainty then None is substituted.
            args = [self,value]
            for dependency in dependencies:
                if self.is_set(dependency,'unc'):
                    t_uncertainty = self.get(dependency,'unc')
                else:
                    t_uncertainty = None
                args.extend((self[dependency],t_uncertainty))
            try:
                uncertainty = uncertainty_function(*args)
            except InferException:
                uncertainty = None
        return value,uncertainty


    def as_flat_dict(self,keys=None,index=None):
        """Return as a dict of arrays, including uncertainties encoded as
        'key:unc'."""
        if keys is None:
            keys = self.keys()
        retval = {}
        for key in keys:
            retval[key] = self.get(key,index=index)
            if self.is_set(key,'unc'):
                retval[f'{key}:unc'] = self.get(key,'unc',index=index)
        return retval

    def as_dict(
            self,
            keys=None,
            index=None,
            subkeys=('value','unc','description','units'),
    ):
        """Return as a structured dict including subkeys."""
        ## default to all data
        if keys is None: 
            keys = list(self.keys())
        ## add data
        retval = {}
        retval['description'] = self.description
        if len(self.attributes) > 0:
            retval['attributes'] = self.attributes
        retval['data'] = {}
        for key in keys:
            retval['data'][key] = {}
            for subkey in subkeys:
                if self.is_set(key,subkey):
                    retval['data'][key][subkey] = self.get(key,subkey)
        ## add data format information
        retval['format'] = {
            'classname':self.classname,
            'description': f'spectr.Dataset directory format version {__version__} (https://github.com/aheays/spectr)',
            'version':__version__,
            'creation_date':tools.date_string(),
            }
        return retval
     
    def row(self,index,keys=None):
        """Return dictionary of data corresopnding to one row with
        integer index."""
        if keys is None:
            keys = self.keys()
        return {key:self.get(key,'value',int(index)) for key in keys}
        
        
    def rows(
            self,
            keys=None,
            subkeys=('value',),
            **get_combined_index_kwargs
    ):
        """Iterate value data row by row, returns as a dictionary of
        scalar values."""
        ## get keys/subkeys
        keys = tools.ensure_iterable(keys)
        if keys is None:
            keys = self.keys()
        keys_subkeys_outkeys = []
        for key in keys:
            for subkey in tools.ensure_iterable(subkeys):
                if subkey == 'value':
                    outkey = key
                else:
                    outkey = f'{key}:{subkey}'
                keys_subkeys_outkeys.append((key,subkey,outkey))
        ## get indices to yield
        index = self.get_combined_index(
            return_as='int',**get_combined_index_kwargs)
        for i in index:
            yield {outkey:self.get(key,subkey,i)
                   for key,subkey,outkey in keys_subkeys_outkeys}

    def row_data(self,keys=None,index=None):
        """Iterate rows, returning data in a tuple (faster than
        rows)."""
        if keys is None:
            keys = self.keys()
        if index is None:
            index = slice(0,len(self))
        for t in zip(*[self[key][index] for key in keys]):
            yield t

    def find_unique(self,*match_args,**match_kwargs):
        """Return index of a uniquely matching row."""
        i = self.find(*match_args,**match_kwargs)
        if len(i) == 0:
            raise NonUniqueValueException(f'No matching row found: {match_args=} {match_kwargs=}')
        if len(i) > 1:
            raise NonUniqueValueException(f'Multiple matching rows found: {match_args=} {match_kwargs=}')
        return i[0]

    def unique_row(self,*match_args,return_index=False,**matching_keys_vals):
        """Return uniquely-matching row as a dictionary."""
        i = self.find_unique(*match_args,**matching_keys_vals)
        d = self.as_flat_dict(index=i)
        if return_index:
            return d,i
        else:
            return d

    def unique_value(self,key,*match_args,**match_kwargs):
        """Return value of key from a row that uniquely matches
        keys_vals."""
        i = self.find_unique(*match_args,**match_kwargs)
        value = self.get(key,index=i)
        return value

    def sort(self,sort_keys,reverse=False):
        """Sort rows according to key or keys."""
        if isinstance(sort_keys,str):
            sort_keys = [sort_keys]
        i = np.argsort(self[sort_keys[0]])
        if reverse:
            i = i[::-1]
        for key in sort_keys[1:]:
            i = i[np.argsort(self[key][i])]
        self.index(i)

    def format(
            self,
            keys=None,
            keys_re=None,
            delimiter=' | ',
            line_ending='\n',
            simple=False,       # print data in a table
            unique_values_as_default=False,
            subkeys=('value','unc','vary','step','ref','default','description','units','fmt','kind',),
            exclude_keys_with_leading_underscore=True, # if no keys specified, do not include those with leading underscores
            exclude_inferred_keys=False, # if no keys specified, do not include those which are not explicitly set
            quote=False,                 # quote keys and strings in data
            comment_header=False, # prepend all non-vector-data lines with #
    ):
        """Format data into a string representation."""
        if keys is None:
            if keys_re is None:
                keys = self.keys()
                if exclude_keys_with_leading_underscore:
                    keys = [key for key in keys if key[0]!='_']
                if exclude_inferred_keys:
                    keys = [key for key in keys if not self.is_inferred(key)]
            else:
                keys = []
        if keys_re is not None:
            keys = {*keys,*self.match_keys(regexp=keys_re)}
        ##
        for key in keys:
            self.assert_known(key)
        ## collect columns of data -- maybe reducing to unique values
        columns = []
        header_values = {}
        for key in keys:
            self.assert_known(key)
            if len(self) == 0:
                break
            formatted_key = ( f'"{key}"' if quote else key )
            if (
                    not simple and unique_values_as_default # input parameter switch
                    and not np.any([self.is_set(key,subkey) for subkey in subkeys if subkey in self.vector_subkinds and subkey != 'value']) # no other vector subdata 
                    and len((tval:=self.unique(key))) == 1 # q unique value
            ): 
                ## value is unique, format value for header
                header_values[key] = tval[0] 
            else:
                ## format columns
                for subkey in subkeys:
                    if self.is_set(key,subkey) and subkey in self.vector_subkinds:
                        if subkey == 'value':
                            formatted_key = (f'"{key}"' if quote else f'{key}')
                        else:
                            formatted_key = (f'"{key}:{subkey}"' if quote else f'{key}:{subkey}')
                        fmt = self._get_subkey_attribute_property(key,subkey,'fmt')
                        kind = self._get_subkey_attribute_property(key,subkey,'kind')
                        if quote and kind == 'U':
                            vals = ['"'+format(t,fmt)+'"' for t in self[key,subkey]]
                        else:
                            vals = [format(t,fmt) for t in self[key,subkey]]
                        width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                        columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
        ## format key metadata
        formatted_metadata = []
        if not simple:
            ## include description of keys
            for key in keys:
                metadata = {}
                if key in header_values:
                    metadata['default'] = header_values[key]
                ## include much metadata in description. If it is
                ## dictionary try to semi-align the keys
                for subkey in subkeys:
                    if subkey in self.scalar_subkinds and self.is_set(key,subkey):
                        metadata[subkey] = self[key,subkey]
                if isinstance(metadata,dict):
                    line = f'{key:20} = {{ '
                    for tkey in subkeys:
                        if tkey not in self.scalar_subkinds:
                            continue
                        if tkey not in metadata:
                            continue
                        tval = metadata[tkey]
                        line += f'{tkey!r}: {tval!r}, '
                    line += '}'
                else:
                    line = f'{key:20} = {metadata!r}'
                formatted_metadata.append(line)
        ## make full formatted string
        retval = ''
        if not simple:
            retval += f'[classname]\n{self.classname}\n'
        if not simple and self.description not in (None,''):
            retval += f'[description]\n{self.description}\n'
        if len(self.attributes) > 0:
            retval += f'[attributes]\n'+'\n'.join([f'{repr(tkey)}: {repr(tval)}' for tkey,tval in self.attributes.items()])+'\n'
        if formatted_metadata != []:
            retval += '[metadata]\n'+'\n'.join(formatted_metadata)
        if columns != []:
            if len(retval) > 0:
                retval += '\n[data]\n'
        if comment_header:
            retval = '# '+retval.replace('\n','\n# ')
        if columns != []:
            retval += line_ending.join([delimiter.join(t) for t in zip(*columns)])+line_ending
        return retval

    def format_metadata(
            self,
            keys=None,
            keys_re=None,
            subkeys=('default','description','units','fmt','kind',),
            exclude_keys_with_leading_underscore=True, # if no keys specified, do not include those with leading underscores
            exclude_inferred_keys=False, # if no keys specified, do not include those which are not explicitly set
    ):
        """Format metadata into a string representation."""
        ## determine keys to include
        if keys is None:
            if keys_re is None:
                keys = self.keys()
                if exclude_keys_with_leading_underscore:
                    keys = [key for key in keys if key[0]!='_']
                if exclude_inferred_keys:
                    keys = [key for key in keys if not self.is_inferred(key)]
            else:
                keys = []
        if keys_re is not None:
            keys = {*keys,*self.match_keys(regexp=keys_re)}
        for key in keys:
            self.assert_known(key)
        ## format key metadata
        metadata = {}
        for key in keys:
            ## include much metadata in description. If it is
            ## dictionary try to semi-align the keys
            metadata[key] = {}
            for subkey in subkeys:
                if subkey in self.scalar_subkinds and self.is_set(key,subkey):
                    metadata[key][subkey] = self[key,subkey]
        retval = tools.format_dict(metadata,newline_depth=0,enclose_in_braces=False)
        return retval

    # def print_metadata(self,*args_format_metadata,**kwargs_format_metadata):
        # """Print a string representation of metadata"""
        # string = self.format_metadata(*args_format_metadata,**kwargs_format_metadata)
        # print( string)

    def describe(self):
        """Print a convenient description of this Dataset."""
        print(self.get_description())

    def get_description(self):
        data = Dataset(
            prototypes={
                'key':{'kind':'U',},
                'subkey':{'kind':'U',},
                'inferred':{'kind':'b'},
                'kind':{'kind':'U'},
                'dtype':{'kind':'U'},
                'memory':{'kind':'f','fmt':"0.1e"},
                'units':{'kind':'U'},
                'unique':{'kind':'i'},
                'description':{'kind':'U'},
            },
        )
        import gc,sys
        total_memory = 0
        for key in self:
            for subkey in self.vector_subkinds:
                if self.is_set(key,subkey):
                    d = {}
                    d['key'] = key
                    d['subkey'] = subkey
                    d['inferred'] = self.is_inferred(key)
                    ### d['kind'] = (self[key,'kind'] if subkey == 'value' else '')
                    d['dtype'] = str(self[key,subkey].dtype)
                    d['memory'] = sys.getsizeof(self._data[key][subkey])
                    total_memory += d['memory']
                    d['unique'] = len(self.unique(key))
                    d['units'] = (self[key,'units'] if subkey == 'value' and self.is_known(key,'units') else '')
                    d['description'] = (self[key,'description'] if subkey == 'value' else self.vector_subkinds[subkey]['description'])
                    data.append(d)
        description = []
        description.append(f'name: {self.name!r}')
        description.append(f'length: {len(self)}')
        description.append(f'number of keys: {len(self.keys())}')
        description.append(f'total memory: {total_memory:0.1e}')
        description.append(f'classname: {self.classname!r}')
        if len(self.description) > 0:
            description.append(f'description: {self.description!r}')
        for key,val in self.attributes.items():
            description.append(f'attribute: {key}: {val!r}')
        if len(data) > 0:
            if np.all(data['units']==''):
                data.pop('units')
            if np.all(data['subkey']=='value'):
                data.pop('subkey')
            description.append(str(data))
        description = '\n'.join(description)
        return description

    
    def format_as_list(self):
        """Form as a valid python list of lists. OBSOLETE?"""
        retval = f'[ \n'
        data = self.format(
            delimiter=' , ',
            simple=True,
            quote=True,
        )
        for line in data.split('\n'):
            if len(line)==0:
                continue
            retval += '    [ '+line+' ],\n'
        retval += ']'
        return retval

    def __str__(self):
        return self.format(simple=True)

    def __repr__(self):
        return self.name
            
    def save(
            self,
            filename,
            keys=None,
            subkeys=None,
            exclude_subkeys=None, 
            filetype=None,           # 'text' (default), 'hdf5', 'directory'
            explicit_keys_only=False,
            **format_kwargs,
    ):
        """Save some or all data to a file.  Valid filetypes are: [
        'text' (default), 'hdf5', 'directory', 'npz', 'rs', 'psv',
        'csv']"""
        if filetype is None:
            ## if not provided as an input argument then get save
            ## format form filename, or default to text
            filetype = tools.infer_filetype(filename)
            if filetype == None:
                filetype = 'text'
        if keys is None:
            keys = self.keys()
        if explicit_keys_only:
            keys = [key for key in keys if self.is_explicit(key)]
        if subkeys is None:
            ## get a list of default subkeys, ignore those beginning
            ## with "_" and some specific keys
            subkeys = [subkey for subkey in self.all_subkinds if
                       (subkey[0] != '_') and subkey not in ('infer','cast')]
        if exclude_subkeys is not None:
            subkeys = [t for t in subkeys if t not in exclude_subkeys]
        if filetype == 'hdf5':
            ## hdf5 file
            tools.dict_to_hdf5(filename,self.as_dict(keys=keys,subkeys=subkeys),verbose=False)
        elif filetype == 'npz':
            ## numpy archive
            np.savez(filename,self.as_dict(keys=keys,subkeys=subkeys))
        elif filetype == 'directory':
            ## directory of npy files
            tools.dict_to_directory(
                filename,
                self.as_dict(keys=keys,subkeys=subkeys),
                repr_strings= True)
        elif filetype == 'text':
            ## space-separated text file
            format_kwargs.setdefault('delimiter',' ')
            tools.string_to_file(filename,self.format(keys,subkeys=subkeys,**format_kwargs))
        elif filetype == 'rs':
            ## ␞-separated text file
            format_kwargs.setdefault('delimiter',' ␞ ')
            tools.string_to_file(filename,self.format(keys,subkeys=subkeys,**format_kwargs))
        elif filetype == 'psv':
            ## |-separated text file
            format_kwargs.setdefault('delimiter',' | ')
            tools.string_to_file(filename,self.format(keys,subkeys=subkeys,**format_kwargs))
        elif filetype == 'csv':
            ## comma-separated text file
            format_kwargs.setdefault('delimiter',', ')
            tools.string_to_file(filename,self.format(keys,subkeys=subkeys,**format_kwargs))
        else:
            raise Exception(f'Do not know how save to {filetype=}')
            
    def load(
            self,
            filename,
            filetype=None,
            _return_classname_and_data=False, # hack used internally
            _data_dict_provided=None, # hack used internally
            **kwargs
    ):
        '''Load data from a file. Valid filetypes are ["hdf5",
        "directory", "npz", "org", "text", "rs", "psv", "csv"], "simple_text"'''
        ## kwargs are for load_from_dict or the load method.  Divide these up.
        import inspect
        all_load_from_dict_kwargs = inspect.getfullargspec(self.load_from_dict).args
        load_from_dict_kwargs = {key:kwargs.pop(key) for key in list(kwargs)
                                 if key in all_load_from_dict_kwargs}
        load_function_kwargs = kwargs
        load_function_kwargs['filename'] = filename
        ## kind of a hack to bypass reloading file if initially loaded
        ## by dataset.load to get the correct classname
        if _data_dict_provided:
            data_dict = _data_dict_provided
            self.load_from_dict(data_dict,**load_from_dict_kwargs )
            return
        ## determine filetype if not given
        if filetype is None:
            ## if not provided as an input argument then get save
            ## format form filename, or default to text
            filetype = tools.infer_filetype(filename)
            if filetype == None:
                filetype = 'text'
        ## load to dict according to filetype. Each load function
        ## returns two dictionaries, the data, and any additional
        ## keyword arguments required for load_from_dict
        if filetype in self.load_functions:
            load_function = self.load_functions[filetype]
        else:
            raise Exception(f"No load_function found for file {filename!r} with filetype {filetype!r}") 
        data_dict = load_function(**load_function_kwargs)
        ## deduce classname
        if 'classname' in data_dict:
            classname = str(data_dict.pop('classname'))
        elif ('format' in data_dict
              and isinstance(data_dict['format'],dict)
              and 'classname' in data_dict['format']):
            classname = str(data_dict['format']['classname'])
        else:
            classname = None
        ## hack -- sometimes classname is quoted, so remove them
        if isinstance(classname,str):
            classname = str(classname).strip('" \'"')
        ## hacks for changed deprecated classnames
        hack_changed_classnames = {
            'levels.Atomic':'levels.Atom',
            'levels.Diatomic':'levels.Diatom',
            'levels.LinearDiatomic':'levels.Diatom',
            'levels.Triatomic':'levels.Triatom',
            'levels.LinearTriatomic':'levels.LinearTriatom',
            'lines.Atomic':'lines.Atom',
            'lines.Diatomic':'lines.Diatom',
            'lines.LinearDiatomic':'lines.Diatom',
            'lines.Triatomic':'lines.Triatom',
            'lines.LinearTriatomic':'lines.LinearTriatom',
        }
        if classname in hack_changed_classnames:
            warnings.warn(f'Changing old classname {classname!r} into new {hack_changed_classnames[classname]!r}')
            classname = hack_changed_classnames[classname]
        ## return classname only
        if _return_classname_and_data:
            return classname,data_dict
        ## test loaded classname matches self
        if classname is not None and classname != self.classname:
            warnings.warn(f'The loaded classname {repr(classname)} does not match self {repr(self.classname)}')
        ## load data into self
        self.load_from_dict(data_dict,**load_from_dict_kwargs)

    def load_from_dict(
            self,
            data_dict,
            keys=None,          # load only this data
            flat=False,
            metadata=None,
            translate_keys=None, # from key in file to key in self, None for skip
            translate_keys_regexp=None, # a list of (regexp,subs) pairs to translate keys -- operate successively on each key
            translate_from_anh_spectrum=False, # HACK to translate keys from spectrum module
            match=None,         # limit what is loaded
    ):
        """Load from a structured dictionary as produced by as_dict."""
        ## add metadata to data_dict dictionary, if metadata key is not
        ## present in data_dict then ignore it
        if metadata is not None:
            for key,info in metadata.items():
                if key in data_dict['data']:
                    for subkey,val in info.items():
                        data_dict['data'][key][subkey] = val
        ## 2021-06-11 HACK TO ACCOUNT FOR DEPRECATED ATTRIBUTES DELETE ONE DAY
        if 'default_step' in data_dict: # HACK
            data_dict.pop('default_step') # HACK
        ## END OF HACK
        ## description is saved in data_dict
        if 'description' in data_dict:
            self.description = str(data_dict.pop('description'))
        ## attributes are saved in data_dict, try to evalute as literal, or keep as string on fail
        if 'attributes' in data_dict:
            for key,val in data_dict.pop('attributes').items():
                self.attributes[key] = tools.safe_eval_literal(val)
        ## actual data dict stored in 'data'
        if 'data' in data_dict:
            data = data_dict['data']
            ## translate keys if given in input arguments
            if translate_keys is None:
                translate_keys = {}
            ## this block should be part of lines.py and levels.py not here
            if translate_from_anh_spectrum:
                translate_keys.update({
                    'Jp':'J_u', 'Sp':'S_u', 'Tp':'E_u',
                    'labelp':'label_u', 'sp':'s_u',
                    'speciesp':'species_u', 'Λp':'Λ_u', 'vp':'v_u',
                    'column_densityp':'Nself_u', 'temperaturep':'Teq_u',
                    'Jpp':'J_l', 'Spp':'S_l', 'Tpp':'E_l',
                    'labelpp':'label_l', 'spp':'s_l',
                    'speciespp':'species_l', 'Λpp':'Λ_l', 'vpp':'v_l',
                    'column_densitypp':'Nself_l', 'temperaturepp':'Teq_l',
                    'Treduced_common_polynomialp':None, 'Tref':'Eref',
                    'branch':'branch', 'dfv':None,
                    'level_transition_type':None, 'partition_source':None,
                    'Γ':'Γ','df':None,
                })
            for from_key,to_key in translate_keys.items():
                if from_key in data:
                    if to_key is None:
                        data.pop(from_key)
                    else:
                        data[to_key] = data.pop(from_key)
            ## translate keys with regexps
            if translate_keys_regexp is not None:
                for key in list(data.keys()):
                    original_key = key
                    for match_re,sub_re in translate_keys_regexp:
                        key = re.sub(match_re,sub_re,key)
                    if key != original_key:
                        data[key] = data.pop(original_key)
            ## Set data in self and selected attributes
            scalar_data = {}
            for key in data:
                ## only load requested keys
                if keys is not None and key not in keys:
                    continue
                ## no data
                if 'value' not in data[key]:
                    raise Exception(f'No "value" subkey in data {repr(keys):0.50s} with subkeys {repr(list(data[key])):0.50s}')
                ## if kind is then add a prototype (or replace
                ## existing if the kinds do not match)
                if 'kind' in data[key]:
                    kind = str(data[key].pop('kind'))
                    if key not in self.prototypes or self.prototypes[key]['kind'] != kind:
                        self.set_prototype(key,kind)
                ## vector data but given as a scalar -- defer loading
                ## until after vector data so the length of data is known
                if np.isscalar(data[key]['value']):
                    scalar_data[key] = data[key]
                ## vector data -- add value and subkeys
                else:
                    self[key,'value'] = data[key].pop('value')
                    for subkey in data[key]:
                        self[key,subkey] = data[key][subkey]
            ## load scalar data
            for key in scalar_data:
                self[key,'value'] = scalar_data[key].pop('value')
                for subkey in scalar_data[key]:
                    self[key,subkey] = scalar_data[key][subkey]
            ## limit to match if requested
            if match is not None:
                self.limit_to_match(match)

    def load_from_parameters_dict(
            self,
            parameters,         # a dictionary
            index_key='key',
    ):
        """Load a dictionary of dictionaries
        recursively. Subdictionary keys are joined to their parent key
        with '_'.  Only scalar data, Parameters, and dictionaries with
        string keys are added. Everything else is ignored completely
        and silently."""
        def recursively_flatten_scalar_dict(data,prefix=''):
            """Join keys of successive subdicts with '_' until a scalar value of Parameter is reach."""
            from .optimise import Parameter
            retval = {}
            for key,val in data.items():
                if isinstance(key,str):
                    key = prefix + key
                    if np.isscalar(val):
                        retval[key] = val
                    elif isinstance(val,Parameter):
                        retval[key] = val.value
                        retval[key+':unc'] = val.unc
                    elif isinstance(val,dict):
                        ## subdict -- join to key
                        tdata = recursively_flatten_scalar_dict(val,prefix=key+'_')
                        for tkey,tval in tdata.items():
                            retval[tkey] = tval
                    else:
                        ## ignore other kinds of data
                        pass
            return retval
        ## load data into vectors
        data = {}               
        for i,(keyi,datai) in enumerate(parameters.items()):
            ## new data point
            datai = recursively_flatten_scalar_dict(datai)
            datai[index_key] = keyi 
            ## initialise vectors
            for key in datai:
                if key not in data:
                    if i==0:
                        data[key] = []
                    else:
                        ## missing data thus far, initalise nans -- SHOULD BE
                        ## GENERALISED
                        data[key] = [nan for i in range(i)]
            ## append to vectors
            for key in data:
                ## missing data in this row, set to nan, -- SHOULD BE
                ## GENERALISED
                if key not in datai:
                    datai[key] = nan
            for key,val in datai.items():
                data[key].append(val)
        ## unflatten
        data = _convert_flat_data_dict(data)
        ## add to self
        self.load_from_dict(data,flat=True)

    def load_from_string(
            self,
            string,             # multi line string in the format expected by self._load_from_text
            delimiter='|',      # column delimiter
            **load_kwargs       # other kwargs passed to self.load
    ):     
        """Load data from a delimter and newline separated string."""
        ## Write a temporary file and then uses the regular file load
        tmpfile = tools.tmpfile()
        tmpfile.write(string.encode())
        tmpfile.flush()
        tmpfile.seek(0)
        self.load_from_dict(
            _load_from_text(
                tmpfile.name,
                delimiter=delimiter,
                **load_kwargs))

    def load_from_lists(self,keys,*values):
        """Add many lines of data efficiently, with values possible
        optimised."""
        cache = {}
        if len(cache) == 0:
            ## first construct
            cache['ibeg'] = len(self)
            cache['keys_vals'] = {key:[t[j] for t in values] for j,key in enumerate(keys)}
        for key,val in cache['keys_vals'].items():
            self[key] = val
        def format_input_function():
            retval = self.format_as_list()
            retval = f'{self.name}.load_from_lists(' + retval[1:-1] + ')'
            return retval
        self.add_format_input_function(format_input_function)

    @format_input_method()
    def concatenate(
            self,
            new_dataset,        # a Datset to concatenate
            keys=None,          # keys to include in concatenation, defaults to explicitly set
            default=None,      # use these keys_vals data if missing in self or in new_dataset
            optimise=False,     # include an optimisation function that updates concatenated data
            match=None,         # only concatenate matching keys_vals
    ):
        """Extend self by new_dataset. If keys=None then existing and
        new_dataset must have a complete of explicity set keys, or a
        default value set in the input dictionary 'default' """
        ## determine index of new_data to concatenate
        if match is None:
            ## all of it
            new_data_index = slice(0,len(new_dataset))
            new_data_length = len(new_dataset)
        else:
            new_data_index = new_data.match(match)
            new_data_length = np.sum(new_data_index)
        ## process default, keys that are missing a subkey are
        ## converted to (key,'value'). Add default in self to list of
        ## default.  Set default in self now so they are available
        ## when concatenating.
        if default is None:
            default = {}
        for key in list(default):
            if isinstance(key,str):
                default[key,'value'] = default.pop(key)
        for key,subkey in default:
            if not self.is_set(key,subkey):
                self[key,subkey] = default[key,subkey]
        ## if keys to concatenate not specified as an input argument
        ## then use all explicitly set keys in self or new_data
        if keys == None:
            keys = copy(self.explicitly_set_keys())
            for key in new_dataset.explicitly_set_keys():
                if key not in keys:
                    keys.append(key)
        ## determine which subkeys are set in either self or new_dataset,
        ## these will be concatenated.  Value must be included.
        keys_subkeys = []
        for key in keys:
            for subkey in self.vector_subkinds:
                if (
                        ## this weird .and. construct is needed for
                        ## subkey=='unc'. The is_known(key) statement
                        ## ensures that this is_set if key is known.
                        ## Testing is_known(key,'unc') will unnecessarily set all
                        ## uncertainties, defaulting to zero.
                        (self.is_known(key) and self.is_set(key,subkey)) or
                        (new_dataset.is_known(key) and new_dataset.is_set(key,subkey))
                ):
                    keys_subkeys.append((key,subkey))
        ## test if self is totally empty or has zero length and all
        ## keys have default set. If so then permit concatenation of
        ## unset keys by initialising them here to empty arrays
        if ( len(self) == 0
             and np.all([self.is_set(key,'default') for key in self]) ):
            ## currently no data at all then, initialise keys as empty
            ## arrays to be concatenated, complex indexing preserves
            ## default if it exists
            for key in keys:
                if not key in self:
                    if key in self.prototypes:
                        self[key] = []
                    else:
                        self.set_new(key,value=[],kind=new_dataset[key,'kind'])
        ## make sure keys are known to self
        for key,subkey in keys_subkeys:
            if not self.is_known(key,subkey):
                raise Exception(f'Concatenated (key,subkey) not known to self: {(key,subkey)!r}')
        ## delete any inferences to concatenated keys in self, they
        ## will now be explicitly set
        for key in keys:
            self.unlink_inferences(keys)
        ## remove unwanted keys from self
        for key in list(self):
            if key not in keys:
                self.unset(key)
        ## extend self to fit new data
        old_length = len(self)
        total_length = len(self) + new_data_length
        index = slice(old_length,total_length)
        self._reallocate(total_length)
        ## function to get new data for this (key,subkey) pair from
        ## somewhere
        def _get_new_data(key,subkey):
            if new_dataset.is_known(key,subkey):
                ## from new_data
                new_val = new_dataset[key,subkey,new_data_index]
            elif (key,subkey) in default:
                ## from input default
                new_val = default[key,subkey]
            else:
                ## from default in self
                new_val = self.get_default(key,subkey)
            return new_val
        ## set data now
        for key,subkey in keys_subkeys:
            self.set(key,subkey,_get_new_data(key,subkey),index)
        ## if optimised then add as a construct function
        if optimise:
            def construct_function():
                ## test if anything has changed and needs updating,
                ## either new_dataset has changed or self has change
                ## and needs to be reverted
                if ( new_dataset._global_modify_time > self._global_modify_time
                    or self._global_modify_time > self._last_construct_time):
                    ## only modify (key,subkey) pairs that have
                    ## changed
                    for key,subkey in keys_subkeys:
                        if (
                                (new_dataset.is_known(key,subkey) and (new_dataset[key,'_modify_time'] > self[key,'_modify_time']))
                                or (self[key,'_modify_time'] > self._last_construct_time)
                        ):
                            self.set(key,subkey,_get_new_data(key,subkey),index,set_changed_only=True)
            self.add_construct_function(construct_function)
            self.add_suboptimiser(new_dataset)
            new_dataset.permit_indexing = False
            self.permit_indexing = False

    def append(self,keys_vals=None,**keys_vals_as_kwargs):
        """Append a single row of data from kwarg scalar values."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals |= keys_vals_as_kwargs
        for key in keys_vals:
            keys_vals[key] = [keys_vals[key]]
        self.extend(keys_vals)

    def extend(self,keys_vals=None,**keys_vals_as_kwargs):
        """Extend self with new data.  All keys or their defaults must be set
        in both new and old data."""
        ## get lists of new data
        if keys_vals is None:
            keys_vals = {}
        keys_vals |= keys_vals_as_kwargs
        ## separate subkeys
        subkeys_vals = {}
        for key in list(keys_vals):
            if not isinstance(key,str):
                tkey,tsubkey = key
                if tsubkey == 'value':
                    ## no need to store subkey
                    keys_vals[tkey] = keys_vals.pop(key)
                else:
                    subkeys_vals[tkey,tsubkey] = keys_vals.pop(key)
        ## new keys
        keys = list(keys_vals.keys())
        ## ensure new data includes all explicitly set keys in self
        for key in self.explicitly_set_keys():
            if key not in keys:
                if self.is_set(key,'default'):
                    keys_vals[key] = self[key,'default']
                    keys.append(key)
                else:
                    raise Exception(f'Extending data missing key: {repr(key)}')
        ## Ensure all new keys are existing data, unless the current
        ## Dataset is zero length, then add them.
        for key in keys:
            if not self.is_known(key):
                if len(self) == 0:
                    if key in self.prototypes:
                        kind = None
                    elif len(keys_vals[key]) > 0:
                        kind = array(keys_vals[key]).dtype.kind
                    else:
                        kind ='f'
                    self.set(key,'value',[],kind=kind)
                else:
                    raise Exception(f"Extending key not in existing data: {repr(key)}")
        ## determine length of new data
        original_length = len(self)
        extending_length = None
        for key,val in keys_vals.items():
            if tools.isiterable(val):
                if extending_length is None:
                    extending_length = len(val)
                elif extending_length != len(val):
                    raise Exception(f'Mismatched lengths in extending data')
        if extending_length is None:
            raise Exception("No vector data in new data")
        total_length = original_length + extending_length
        ## reallocate and lengthen arrays if necessary
        self._reallocate(total_length)
        ## add new data to old, set values first then other subdata
        ## afterwards
        if original_length == 0:
            index = None
        else:
            index = slice(original_length,total_length)
        for key in keys:
            self.set(key,'value',keys_vals[key],index)
        for (key,subkey),val in subkeys_vals.items():
            self.set(key,subkey,val,index)

    def join(self,data):
        """Join keys from a new dataset set onto this one.  No key overlap allowed."""
        ## add data if it is a Dataset
        if isinstance(data,Dataset):
            ## error checks
            if len(self) != len(data):
                raise Exception(f'Length mismatch between self and new dataset: {len(self)} and {len(data)}')
            i,j = tools.common(self.keys(),data.keys())
            if len(i) > 0:
                raise Exception(f'Overlapping keys between self and new dataset: {repr(self.keys()[i])}')
            ## add from data
            for key in data:
                if key in self.prototypes:
                    self[key] = data[key]
                else:
                    if not self.permit_nonprototyped_data:
                        raise Exception(f'Key from new dataset is not prototyped in self: {repr(key)}')
                    self._data[key] = deepcopy(data._data[key])
        ## add data if it is a dictionary
        elif isinstance(data,dict):
            for key in data:
                if self.is_set(key):
                    raise Exception(f'Key already present: {key!r}')
                self[key] = data[key]

    def _reallocate(self,new_length):
        """Lengthen data arrays."""
        ## if not self.permit_dereferencing and new_length > len(self):
        if not self.permit_dereferencing:
            raise Exception(f'Cannot reallocate: {self.permit_dereferencing=}')
        ## copy individual arrays, increasing length if necessary
        for key in self:
            for subkey in self._data[key]:
                if subkey in self.vector_subkinds:
                    val = self._data[key][subkey]
                    old_length = len(val)
                    if new_length > old_length:
                        self._data[key][subkey] = np.concatenate(
                            (val,
                             np.empty(int(new_length*self._over_allocate_factor-old_length),
                                      dtype=val.dtype)))
        self._length = new_length
        ## increase length of modify time array
        if len(self._row_modify_time) < new_length:
            self._row_modify_time = np.concatenate((
                self._row_modify_time,
                np.full(new_length*self._over_allocate_factor
                        -len(self._row_modify_time),timestamp())))

    def plot(
            self,
            xkeys=None,         # key to use for x-axis data
            ykeys=None,         # list of keys to use for y-axis data
            zkeys=None,         # plot x-y data separately for unique combinations of zkeys
            ykeys_re=None,
            fig=None,           # otherwise automatic
            ax=None,            # otherwise automatic
            xnewaxes=True,      # plot x-keys on separates axes -- else as different lines
            ynewaxes=True,      # plot y-keys on separates axes -- else as different lines
            znewaxes=False,     # plot z-keys on separates axes -- else as different lines
            legend=True,        # plot a legend or not
            legend_loc='best',
            annotate_lines=False, # annotate lines with their labels
            zlabel_format_function=None, # accept key=val pairs, defaults to printing them
            label_prefix='', # put this before label otherwise generated
            plot_errorbars=True, # if uncertainty available
            plot_xlabel=True,
            plot_ylabel=True,
            xlog=False,
            ylog=False,
            ncolumns=None,       # number of columsn of subplot -- None to automatically select
            show=False,          # show figure after issuing plot commands
            xlim=None,           # set a (xbeg,xend) pair
            ylim=None,           # set a (
            title=None,
            xsort=True,         # True sort by xkey, False, do not sort, or else a key or list of keys to sort by
            annotate_points_keys=None,
            plot_kwargs=None,      # e.g. color, linestyle, label etc
    ):
        """Plot data. Good luck."""
        from matplotlib import pyplot as plt
        from spectr import plotting
        ## re-use or make a new figure/axes
        if ax is not None:
            fig = ax.figure
        elif fig is None:
            fig = plt.gcf()
            fig.clf()
        elif isinstance(fig,int):
            fig = plotting.qfig(fig)
        ## no data, do nothing
        if len(self)==0:
            return
        ## xkey, ykeys, zkeys
        if xkeys is None:
            if self.default_xkeys is not None:
                xkeys = copy(self.default_xkeys)
        xkeys = list(tools.ensure_iterable(xkeys))
        if ykeys is None:
            if self.default_ykeys is not None:
                ykeys = copy(self.default_ykeys)
            else:
                ykeys = []
        ykeys = list(tools.ensure_iterable(ykeys))
        if ykeys_re is not None:
            ykeys += [key for key in self if re.match(ykeys_re,key)]
        if zkeys is None:
            zkeys = self.default_zkeys
        zkeys = list(tools.ensure_iterable(zkeys))
        zkeys = [key for key in zkeys if key not in xkeys+ykeys and self.is_known(key)] # remove xkey and ykeys from zkeys
        for t in xkeys+ykeys+zkeys:
            self.assert_known(t)
        ## total number of subplots in figure
        nsubplots = 1
        if xnewaxes:
            nsubplots *= len(xkeys)
        if ynewaxes:
            nsubplots *= len(ykeys)
        if znewaxes:
            nsubplots *= len(zkeys)
        ## plot each xkey/ykey/zkey combination
        for ix,xkey in enumerate(xkeys):
            for iy,ykey in enumerate(ykeys):
                for iz,(dz,z) in enumerate(self.unique_dicts_matches(*zkeys)):
                    ## sort data
                    if xsort == True:
                        z.sort(xkey)
                    elif xsort == False:
                        pass
                    else:
                        z.sort(xsort)
                    ## get axes
                    if ax is None:
                        isubplot = 0
                        subplot_multiplier = 1
                        if xnewaxes:
                            isubplot += subplot_multiplier*ix
                            subplot_multiplier *= len(xkeys)
                        if ynewaxes:
                            isubplot += subplot_multiplier*iy
                            subplot_multiplier *= len(ykeys)
                        if znewaxes:
                            isubplot += subplot_multiplier*ix
                            subplot_multiplier *= len(zkeys)
                        tax = plotting.subplot(n=isubplot,fig=fig,ncolumns=ncolumns,ntotal=nsubplots)
                    else:
                        tax = ax 
                    ## get axis labels and perhaps convert them to legend labesl
                    label = ''
                    ## x-axis
                    xlabel = xkey
                    if self.is_known(xkey,'units'):
                        xlabel += ' ('+self[xkey,'units']+')'
                    if not xnewaxes:
                        label += f' {xlabel}'
                        xlabel = None
                    ## y-axis
                    ylabel = ykey
                    if self.is_known(ykey,'units'):
                        ylabel += ' ('+self[ykey,'units']+')'
                    if not ynewaxes:
                        label += f' {ylabel}'
                        ylabel = None
                    ## z-axis
                    if zlabel_format_function is None:
                        # zlabel_format_function = self.default_zlabel_format_function
                        zlabel_format_function = tools.dict_to_kwargs
                    zlabel = zlabel_format_function(dz)
                    if not znewaxes:
                        label += f' {zlabel}'
                        zlabel = None
                    ## get color/marker/linestyle
                    if xnewaxes and ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(0),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif not xnewaxes and ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(ix),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif xnewaxes and not ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif xnewaxes and ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif not xnewaxes and not ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(ix),plotting.newlinestyle(iy)
                    elif not xnewaxes and ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(ix),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    elif xnewaxes and not ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    elif not xnewaxes and not ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(ix),plotting.newlinestyle(iz)
                    else:
                        assert('should not happen')
                    ## plotting kwargs
                    kwargs = {} if plot_kwargs is None else copy(plot_kwargs)
                    kwargs.setdefault('marker',marker)
                    kwargs.setdefault('ls',linestyle)
                    kwargs.setdefault('mew',1)
                    kwargs.setdefault('markersize',7)
                    kwargs.setdefault('color',color)
                    kwargs.setdefault('mec',kwargs['color'])
                    if label is not None:
                        kwargs.setdefault('label',label_prefix+label)
                    ## plotting data
                    if self[xkey,'kind'] == 'U':
                        ## if string xkey then ensure different plots are aligned on the axis
                        xkey_unique_strings = self.unique(xkey)
                        x = tools.findin(z[xkey],xkey_unique_strings)
                    else:
                        x = z[xkey]
                    y = z[ykey]
                    if plot_errorbars and (z.is_set(ykey,'unc') or z.is_set(xkey,'unc')):
                        ## get uncertainties if they are known
                        if z.is_set(xkey,'unc'):
                            dx = z.get(xkey,'unc')
                            dx[np.isnan(dx)] = 0.
                        else:
                            dx = np.full(len(z),0.)
                        if z.is_set(ykey,'unc'):
                            dy = z.get(ykey,'unc')
                            dy[np.isnan(dy)] = 0.
                        else:
                            dy = np.full(len(z),0.)
                        ## plot errorbars
                        kwargs.setdefault('mfc','none')
                        i = ~np.isnan(x) & ~np.isnan(y)
                        tax.errorbar(x[i],y[i],dy[i],dx[i],**kwargs)
                        ## plot zero/undefined uncertainty data as filled symbols
                        i = np.isnan(dy)|(dy==0)
                        if np.any(i):
                            kwargs['mfc'] = kwargs['color']
                            if 'fillstyle' not in kwargs:
                                kwargs['fillstyle'] = 'full'
                            if 'ls' in kwargs:
                                kwargs['ls'] = ''
                            else:
                                kwargs['linestyle'] = ''
                            kwargs['label'] = None
                            line = tax.plot(x[i],z[ykey][i],**kwargs)
                    else:
                        kwargs.setdefault('mfc',kwargs['color'])
                        kwargs.setdefault('fillstyle','full')
                        line = tax.plot(x,y,**kwargs)
                    if annotate_points_keys is not None:
                        ## annotate all points with the value of this key
                        # for li,xi,yi in zip(z[annotate_points_keys],x,y): 
                        #     if ~np.isnan(xi) and ~np.isnan(yi):
                        #         plt.annotate(format(li),(xi,yi),fontsize='x-small',in_layout=False)
                        for i in range(len(z)):
                            if ~np.isnan(x[i]) and ~np.isnan(y[i]):
                                annotation = '\n'.join([tkey+'='+format(z[tkey][i],self[tkey,'fmt']) for tkey in tools.ensure_iterable(annotate_points_keys)])
                                # annotation = pformat({tkey:format(z[tkey][i]) for tkey in tools.ensure_iterable(annotate_points_keys)})
                                plt.annotate(annotation,(x[i],y[i]),fontsize='x-small',in_layout=False)
                    if title is not None:
                        tax.set_title(title)
                    if ylabel is not None and plot_ylabel:
                        tax.set_ylabel(ylabel)
                    if xlabel is not None and plot_xlabel:
                        tax.set_xlabel(xlabel)
                    if 'label' in kwargs:
                        if legend:
                            plotting.legend(fontsize='medium',loc=legend_loc,show_style=True,ax=tax)
                        if annotate_lines:
                            plotting.annotate_line(line=line)
                    if xlim is not None:
                        tax.set_xlim(*xlim)
                    if xlog:
                        tax.set_xscale('log')
                    if ylog:
                        tax.set_yscale('log')
                    tax.grid(True,color='gray',zorder=-5)
                    if self[xkey,'kind'] == 'U':
                        plotting.set_tick_labels_text(
                            xkey_unique_strings,
                            axis='x',
                            ax=tax,
                            rotation=70,
                            fontsize='x-small',
                            ha='right',
                        )
                ## set ylim for all axes
                if ylim is not None:
                    for tax in fig.axes:
                        if ylim == 'data':
                            t,t,ybeg,yend = plotting.get_data_range(tax)
                            tax.set_ylim(ybeg,yend)
                        elif tools.isiterable(ylim) and len(ylim) == 2:
                            ybeg,yend = ylim
                            if ybeg is not None:
                                if ybeg == 'data':
                                    t,t,ybeg,t = plotting.get_data_range(tax)
                                tax.set_ylim(ymin=ybeg)
                            if yend is not None:
                                if yend == 'data':
                                    t,t,t,yend = plotting.get_data_range(tax)
                                tax.set_ylim(ymax=yend)
        if show:
            plotting.show()
        return fig

    # def plot_bar(self,xlabelkey=None,ykeys=None):
        # from matplotlib import pyplot as plt
        # from spectr import plotting
        # ax = plotting.gca()
        # if ykeys is None:
            # ykeys = self.keys()
            # if xlabelkey is not None:
                # ykeys.remove(xlabelkey)
        # x = arange(len(self))
        # if xlabelkey is None:
            # xlabels = x
        # else:
            # xlabels = self[xlabelkey]
        # labels = []
        # for iykey,ykey in enumerate(ykeys):
            # ax.bar(
                # x=x+0.1*(iykey-(len(ykeys)-1)/2),
                # height=self[ykey],
                # width=-0.1,
                # tick_label=[format(t) for t in xlabels],
                # color=plotting.newcolor(iykey),
            # )
            # labels.append(dict(color=plotting.newcolor(iykey),label=ykey))
        # plotting.legend(*labels)
        # for t in ax.xaxis.get_ticklabels():
            # t.set_size('small')
            # t.set_rotation(-45)

    def polyfit(self,xkey,ykey,index=None,**polyfit_kwargs):
        """Compute least-squares fit polynomial coefficients."""
        return tools.polyfit(
            self.get(xkey,index=index),
            self.get(ykey,index=index),
            self.get(ykey,'unc',index=index),
            **polyfit_kwargs)


    def __add__(self,other):
        """Adding dataset concatenates data in all keys."""
        retval = self.copy()
        retval.extend(other)
        return retval

    def __radd__(self,other):
        """Adding dataset concatenates data in all keys."""
        retval = self.copy()
        retval.extend(other)
        return retval

    def __ior__(self,other_dict_like):
        """In place addition of key or substitution like a dictionary using |=."""
        for key in other_dict_like.keys():
            self[key] = other_dict_like[key]
        return self

    def __iadd__(self,other_dataset):
        """Concatenate self with another dataset using +=."""
        self.concatenate(other_dataset)
        return self

    ## other possible in place operators: __iadd__ __isub__ __imul__
    ##  __imatmul__ __itruediv__ __ifloordiv__ __imod__ __ipow__
    ##  __ilshift__ __irshift__ __iand__ __ixor__ __ior__
    

def find_common(x,y,keys=None,verbose=False):
    """Return indices of two Datasets that have uniquely matching
    combinations of keys."""
    keys = tools.ensure_iterable(keys)
    ## if empty list then nothing to be done
    if len(x)==0 or len(y)==0:
        return(np.array([],dtype=int),np.array([],dtype=int))
    ## use quantum numbers as default keys -- could use qnhash instead
    if keys is None:
        from . import levels
        if isinstance(x,levels.Base):
            ## a hack to use defining_qn for levels/lines as defalt
            ## match keys
            keys = [key for key in x.defining_qn if x.is_known(key)]
        else:
            raise Exception("No keys provided and defining_qn unavailable x.")
    if verbose:
        print('find_commmon keys:',keys)
    for key in keys:
        x.assert_known(key)
        y.assert_known(key)
    ## sort by first calculating a hash of sort keys
    xhash = np.array([hash(t) for t in x.row_data(keys=keys)])
    yhash = np.array([hash(t) for t in y.row_data(keys=keys)])
    ## get sorted hashes, checking for uniqueness
    xhash,ixhash,inv_xhash,count_xhash = np.unique(xhash,return_index=True,return_inverse=True,return_counts=True)
    if len(xhash) != len(x):
        if verbose:
            print("Duplicate key combinations in x:")
            for i in tools.find(count_xhash>1):
                print(f'    count = {count_xhash[i]},',repr({key:x[key][i] for key in keys}))
        raise Exception(f'There is {len(x)-len(xhash)} duplicate key combinations in x: {repr(keys)}. Set verbose=True to list them.')
    yhash,iyhash = np.unique(yhash,return_index=True)
    if len(yhash) != len(y):
        if verbose:
            print("Duplicate key combinations in y:")
            for i in tools.find(count_yhash>1):
                print(f'    count = {count_yhash[i]},',repr({key:y[key][i] for key in keys}))
        raise Exception(f'Non-unique combinations of keys in y: {repr(keys)}')
    ## use np.searchsorted to find one set of hashes in the other
    iy = np.arange(len(yhash))
    ix = np.searchsorted(xhash,yhash)
    ## remove y beyond max of x
    i = ix<len(xhash)
    ix,iy = ix[i],iy[i]
    ## requires removing hashes that have no searchsorted partner
    i = yhash[iy]==xhash[ix]
    ix,iy = ix[i],iy[i]
    ## undo the effect of the sorting above
    ix,iy = ixhash[ix],iyhash[iy]
    ## sort by index of first array -- otherwise sorting seems to be arbitrary
    i = np.argsort(ix)
    ix,iy = ix[i],iy[i]
    ix = np.asarray(ix,dtype=int)
    iy = np.asarray(iy,dtype=int)
    return ix,iy

def get_common(x,y,keys=None,**limit_to_matches):
    """Return copies of dataset x and y indexed to have aligned common
    keys."""
    if limit_to_matches is not None:
        x = x.matches(**limit_to_matches)
        y = y.matches(**limit_to_matches)
    i,j = find_common(x,y,tools.ensure_iterable(keys))
    return x[i],y[j]

def _get_class(classname):
    """Find a class matching class name."""
    if classname is None:
        classname = 'dataset.Dataset'
    ## hack -- old classnames
    if classname == 'levels.LinearDiatomic':
        classname = 'levels.Diatomic'
    if classname == 'lines.LinearDiatomic':
        classname = 'lines.Diatomic'
    ## end of hack
    if classname == 'dataset.Dataset':
        return Dataset
    else:
        module,subclass = classname.split('.')
        if module == 'lines':
            from . import lines
            return getattr(lines,subclass)
        elif module == 'levels':
            from . import levels
            return getattr(levels,subclass)
        elif module == 'spectrum':
            from . import spectrum
            return getattr(spectrum,subclass)
        elif module == 'atmosphere':
            from . import atmosphere
            return getattr(atmosphere,subclass)
    raise Exception(f'Could not find a class matching {classname=}')
    
def make(classname='dataset.Dataset',*init_args,**init_kwargs):
    """Make an instance of the this classname."""
    class_object = _get_class(classname)
    dataset = class_object(*init_args,**init_kwargs)
    return dataset

def load(
        filename,
        classname=None,
        prototypes=None,
        permit_nonprototyped_data=None,
        name=None,
        **load_kwargs):
    """Load a Dataset.  Attempts to automatically find the correct
    subclass if it is not provided as an argument, but this requires
    loading the file twice."""
    ## get classname, and save data_dict so it doesn't have to be
    ## loaded again below -- CONFUSING!
    if classname is None:
        d = Dataset()
        classname,data = d.load(
            filename,_return_classname_and_data=True,**load_kwargs)
    else:
        data = None
    ## make Dataset
    init_kwargs = {}
    if prototypes is not None:
        init_kwargs['prototypes'] = prototypes
    if permit_nonprototyped_data is not None:
        init_kwargs['permit_nonprototyped_data'] = permit_nonprototyped_data
    if name is not None:
        init_kwargs['name'] = name
    retval = make(classname,**init_kwargs)
    retval.load(filename,_data_dict_provided=data,**load_kwargs)
    return retval

@cache
def load_and_cache(*args,**kwargs):
    """Load Dataset using dataset.load and cache the result for faster
    reload."""
    return load(*args,**kwargs)

def copy_from(dataset,*args,**kwargs):
    """Make a copy of dataset with additional initialisation args and
    kwargs."""
    classname = dataset.classname # use the same class as dataset
    retval = make(classname,*args,copy_from=dataset,**kwargs)
    return retval

def decode_flat_key(key):
    if r:=re.match(r'([^:]+):([^:]+)',key):
        key,subkey = r.groups()
    else:
        key,subkey = key,'value'
    return key,subkey



