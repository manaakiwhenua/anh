import re

from scipy import constants
import numpy as np
from bidict import bidict

from . import tools

"""Convert between units and rigorously related quantities."""


#####################
## unit conversion ##
#####################

## relationship between units and a standard SI unit.  Elements can be
## conversion factors to SI, or a pair of conversion and inverse
## conversion functions
groups = {

    'length': {
        'm'           :1.                ,  # astronomical units       ,
        'pc'          :3.0857e16       ,    # parsecs
        'fm'          :1e-15               ,
        'pm'          :1e-12               ,
        'nm'          :1e-9               ,
        'μm'          :1e-6               ,
        'mm'          :1e-3               ,
        'cm'          :1e-2               ,
        'km'          :1e3              ,
        'Mm'          :1e6              ,
        'Gm'          :1e9              ,
        'solar_radius':6.955e8         ,
        'AU'          :1.496e11        ,  # astronomical units       ,
        'au'          :5.2917721092e-11, #  atomic units (Bohr radius, a0),
        'Bohr'        :5.2917721092e-11, #  atomic units (Bohr radius, a0),
        'a0'          :5.2917721092e-11, #  atomic units (Bohr radius, a0),
        'Å'           :1e-10             ,
    },

    'inverse_length': {
        'm-1' :1   ,
        'cm-1':1e2,
        'mm-1':1e3,
        'μm-1':1e6,
        'nm-1':1e9,
    },

    'area': {
        'm2'   :  1., 
        'cm2'  :  1e-4, 
        'Mb'   :  1e-20, 
    },

    'inverse_area': {
        'm-2' : 1., 
        'cm-2': 1e4,
    },

    'volume': {
        'm3'  : 1., 
        'cm3' : 1e-6,
        'l'   : 1e-3,
        'L'   : 1e-3,
    },

    'inverse_volume': {
        'm-3' : 1., 
        'cm-3': 1e6,
    },

    'time': {
        's':1.,
        'ms':1e-3,
        'μs':1e-6,
        'ns':1e-9,
        'ps':1e-12,
        'fs':1e-15,
        'minute':60,
        'hour':60*60,
        'day':60*60*24,
        'week':60*60*24*7,
        'year':60*60*24*7*365,
    },

    'frequency': {
        'Hz' :1  ,
        'kHz':1e3,
        'MHz':1e6,
        'GHz':1e9,
        'radians':2*constants.pi,
    },

    'energy': {
        'J'         :1.                    ,
        'kJ'         :1e3                    ,
        'K'         :constants.Boltzmann ,
        'cal'       :4.184               ,
        'eV'        :1.602176634e-19     ,
        'erg'       :1e-7                   ,
        'Hartree'   :4.3597447222071e-18      , # atomic units /hartree, ref: wikipedia 2022-07-03
        'au'        :4.3597447222071e-18      , # atomic units /hartree, ref: wikipedia 2022-07-03
        'kJ.mol-1'  :1e3/constants.Avogadro,
        'kcal.mol-1':6.9477e-21          ,
    },

    'mass': {
        'kg'        :1                      ,
        'g'         :1e-3                    ,
        'solar_mass':1.98855e30           ,
        'amu'       :constants.atomic_mass,
    },
    
    'velocity': {
        'm.s-1'     :1.         ,
        'km.s-1'    :1e3         ,
    },

    'dipole moment': {
        'Debye' : 1.,
        'au'    : 2.541765,
    },

    'pressure': {
        'Pa'      :  1.        ,
        'kPa'     :  1e3      ,
        'bar'     :  1e5      ,
        'mbar'    :  1e2      ,
        'mb'      :  1e2      ,
        'atm'     :  101325.,
        'Torr'    :  133.322 ,
        'dyn.cm-2':  1/(1e5*1e-4)  ,
    },

    'photon': {
        ## canonical unit is frequency
        'Hz'   : 1,
        'kHz'  : 1e3,
        'MHz'  : 1e6,
        'GHz'  : 1e9,
        ## energy
        'J'    : 1/constants.h,
        'eV'   : constants.electron_volt/constants.h,
        'meV'  : 1e-3*constants.electron_volt/constants.h,
        'Hartree'   :4.3597447222071e-18/constants.h      , # atomic units /hartree, ref: wikipedia 2022-07-03
        'au'        :4.3597447222071e-18/constants.h      , # atomic units /hartree, ref: wikipedia 2022-07-03
        ## wavelength
        'm'    : (lambda m     : constants.c/m,lambda Hz           : constants.c/Hz),
        'μm'   : (lambda μm    : constants.c/(μm*1e-6),lambda Hz   : 1e6*constants.c/Hz),
        'nm'   : (lambda nm    : constants.c/(nm*1e-9),lambda Hz   : 1e9*constants.c/Hz),
        'Å'    : (lambda Å     : constants.c/(Å*1e-10),lambda Hz   : 1e10*constants.c/Hz),
        ## wavenumber
        'm-1'  : (lambda m     : m*constants.c,lambda Hz           : Hz/constants.c,),
        'cm-1' : (lambda invcm : constants.c*(1e2*invcm),lambda Hz : 1e-2/(constants.c/Hz)),
    },

    # unit_conversions[('Debye','au')] = lambda x: x/2.541765
    # unit_conversions[('au','Debye')] = lambda x: x*2.541765
    # unit_conversions[('Debye','Cm')] = lambda x: x*3.33564e-30 # 1 Debye is 1e-18.statC.cm-1 -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('Cm','Debye')] = lambda x: x/3.33564e-30 # 1 Debye is 1e-18.statC.cm-1 -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('C','statC')] = lambda x: 3.33564e-10*x # 1 Couloub = sqrt(4πε0/1e9)×stat Coulomb -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('statC','C')] = lambda x: 2997924580*x  # 1 stat Couloub = sqrt(1e9/4πε0)×Coulomb -- WARNING: these are not dimensionally similar units!!!


}
def units(value,unit_in,unit_out,group=None):       
    """Convert units. Group might be needed for units with common
    names."""
    ## cast to array if necessary
    if not np.isscalar(value):
        value = np.asarray(value)
    ## trivial case
    if unit_in == unit_out:
        return value
    ## find group containing this conversion if not specified
    if group is None:
        for factors in groups.values():
                if unit_in in factors and unit_out in factors:
                    break
        else:
            raise Exception(f"Could not find conversion group for {unit_in=} to {unit_out=}.")
    else:
        try:
            factors = groups[group]
        except KeyError:
            raise Exception(f"No conversion for {unit_in=} to {unit_out=} in {group=}.")
    ## convert to SI from unit_in
    factor = factors[unit_in]
    if isinstance(factor,(float,int)):
        value = value*factor
    else:
        value = factor[0](value)
    ## convert to unit_out from SI
    factor = factors[unit_out]
    if isinstance(factor,(float,int)):
        value = value/factor
    else:
        value = factor[1](value)
    return value

def difference(difference,value,unit_in,unit_out,group=None):
    """Convert an absolute finite difference -- not a linear
    approximation."""
    return(np.abs(
        +units(value+difference/2.,unit_in,unit_out,group)
        -units(value-difference/2.,unit_in,unit_out,group)))


##############################
## translate chemical names ##
##############################


@tools.vectorise(cache=True)
def species(value,inp='ascii_or_unicode',out='unicode',verbose=False):
    """Translate species name between different formats."""
    retval = _species_internal(value,inp,out,error_on_fail=True,verbose=verbose,attempted=[])
    return retval

def _species_internal(
        value,inp,out,
        error_on_fail=True,
        verbose=False,
        attempted=(),
        depth = 0,
):
    attempted.append((inp,out))
    depth += 1
    if verbose:
        indent = ''.join(['    ' for i in range(depth)])
        print(f'{indent}Attempting to convert {value!r} from {inp!r} to {out!r}')
    if inp == out:
        ## trivial case
        if verbose:
            print(f'{indent}Succeeded to convert {value!r} from {inp!r} to {out!r}: {value!r}')
        return value
    elif inp in _convert_species_functions:
        if out in _convert_species_functions[inp]:
            retval = _convert_species_functions[inp][out](value)
            ## conversion function exists
            if verbose:
                print(f'{indent}Succeeded to convert {value!r} from {inp!r} to {out!r}: {retval!r}')
            return retval
        else:
            ## try any possible intermediate format
            test = None
            for intermediate in _convert_species_functions[inp]:
                if intermediate == 'description':
                    continue
                if (intermediate,out) in attempted:
                    continue
                test = _species_internal(
                    _species_internal(
                        value, inp, intermediate,
                        error_on_fail=False, verbose=verbose,
                        depth=depth, attempted=attempted,),
                    intermediate,out, error_on_fail=False,verbose=verbose,depth=depth,attempted=attempted,)
                if test is not None:
                    break
            else:
                if verbose:
                    print(f'{indent}Failed to convert {value!r} from {inp!r} to {out!r}')
            if test is not None:
                if verbose: 
                    print(f'{indent}Succeeded to convert {value!r} from {inp!r} to {out!r}: {test!r}')
            return test
    ## fail
    if error_on_fail:
        raise Exception(f"Could not convert species {value!r} from {inp!r} to {out!r} encoding.")
    else:
        return None

## dictionary and functions for converting a species, their keys are
## (inp,out) format strings. Each value is pair (translation_dict,
## translation_function) for direct and/or formulaic translation. The
## dictionary is tried first.
_convert_species_functions = {
    'ascii'              : {'description' : 'E.g., H2O+, 12C16O, or [12C][16O]2',},
    'unicode'            : {'description' : 'E.g., H₂O⁺, ¹²C¹⁶O  or ¹²C¹⁶O₂',},
    'ascii_or_unicode'   : {'description' : 'Guess which',},
    'tuple'              : {'description' : '(structure-prefix,(element0,mass0|None),(element1,mass1|None),...,charge). E.g., for H₂¹⁶O⁺ this is ('',("H",None),("H",None),("O",16),1)',},
    # 'tuple_all_isotopes' : {'description' : 'Same as tuple but all mass numbers are given, if None intially then assume isotope with the greatest natural abundance.',},
    # 'tuple_all_elements' : {'description' : 'Same as tuple but all mass numbers are removed.',},
    'matplotlib'         : {'description' : 'Looks good in a matplotlib string',},
    'stand'              : {'description' : 'As it appears in the STAND chemical network',},
    'kida'               : {'description' : 'As it appears in the KIDA chemical network',},
    'meudon'             : {'description' : '',},
    'cantera'            : {'description' : '',},
    'leiden'             : {'description' : '',},
    'inchikey'           : {'description' : '',},
    'inchi'              : {'description' : '',},
    'latex'              : {'description' : '',},
    'CASint'             : {'description' : ''},
}

def describe_species():
    """Summarise species encodings."""
    retval = ['The species encoding codes are:']
    for key,val in _convert_species_functions.items():
        retval.append(f'{key!r}')
        retval.append(f'    description: {val["description"]}')
        retval.append(f'    conversion functions: '+' '.join([repr(t) for t in val if t!='description']))
    retval = '\n'.join(retval)
    print(retval )

def _convert_ascii_or_unicode_to_unicode(name):
    if re.match(r'.*[0-9+-].*',name):
        name = species(name,'ascii','unicode')
    else:
        name = name
    return name
_convert_species_functions['ascii_or_unicode']['unicode'] = _convert_ascii_or_unicode_to_unicode

def _convert_unicode_to_tuple(name):
    """Turn standard name string into ordered isotope list and charge.  If
    any isotopic masses are given then they will be added to all
    elements."""
    ## e.g., ¹²C¹⁶O₂²⁺
    r = re.match(r'^((?:[⁰¹²³⁴⁵⁶⁷⁸⁹]*[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻]?)$',name)
    if not r:
        raise Exception(f'Could not decode unicode encoded species name: {name!r}')
    name_no_charge = r.group(1)
    if r.group(2) == '':
        charge = 0
    elif r.group(2) == '⁺':
        charge = +1
    elif r.group(2) == '⁻':
        charge = -1
    elif '⁺' in r.group(2):
        charge = int(tools.regularise_unicode(r.group(2)[:-1]))
    else:
        charge = -int(tools.regularise_unicode(r.group(2)[:-1]))
    prefix = ''               # prefix
    nuclei = []
    for part in re.split(r'([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]*[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)',name_no_charge):
        if part=='':
            continue
        elif r:= re.match(r'([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]*)([A-Z][a-z]?)([₀₁₂₃₄₅₆₇₈₉]*)',part):
            mass_number = ( int(tools.regularise_unicode(r.group(1))) if r.group(1) != '' else None )
            element = r.group(2)
            multiplicity = int(tools.regularise_unicode(r.group(3)) if r.group(3) != '' else 1)
            nuclei.append((mass_number,element,multiplicity))
    retval = tuple([prefix]+nuclei+[charge])
    return retval
_convert_species_functions['unicode']['tuple'] = _convert_unicode_to_tuple

def _convert_tuple_to_unicode(data):
    prefix,nuclei,charge = data[0],data[1:-1],data[-1]
    retval = []
    ## prefix
    if len(prefix)>0:
        retval.append(prefix+'-')
    ## elements
    for mass_number,element,multiplicity in nuclei:
        retval.append(
            ('' if mass_number is None else tools.superscript_numerals(str(mass_number)))
            + element
            + ('' if multiplicity == 1 else tools.subscript_numerals(str(multiplicity))))
    ## and the charge
    if charge == 0:
        pass
    elif charge < -1:
        retval.append(tools.superscript_numerals(str(-charge)+'-'))
    elif charge == -1:
        retval.append('⁻')
    elif charge == 1:
        retval.append('⁺')
    else:
        retval.append(tools.superscript_numerals(str(charge)+'+'))
    retval = ''.join(retval)
    return retval
_convert_species_functions['tuple']['unicode'] = _convert_tuple_to_unicode


## stored on disk table of translations
_species_dataset = {}
def _species_database_translate(species,inp,out):
    from . import dataset
    from . import database
    ## load datafile if needed
    if len(_species_dataset) == 0:
        filename = f'{database.data_directory}/species/translations.psv'
        _species_dataset['data'] = dataset.load(filename)
    ## get copy sorted by inp 
    if inp not in _species_dataset:
        data = _species_dataset['data']
        _species_dataset[inp] = _species_dataset['data'].copy()
        _species_dataset[inp].sort(inp)
    data = _species_dataset[inp]
    ## translate
    i = np.searchsorted(data[inp],species)
    if i == len(data) or species != data[i][inp]:
        ## input not found
        raise Exception(f'Could not translate {species!r} using from {inp!r} to {out!r}')
    retval = data[out][i]
    if retval in ['',0]:
        ## blank output data
        raise Exception(f'Could not translate {species!r} using from {inp!r} to {out!r}')
    return retval

# ## various translations possible using _species_dataset
# for inp,out in (
        # ('CASint'  , 'unicode') , 
        # # ('CASint'  , 'ascii')   , 
        # ('unicode' , 'CASint')  , 
        # # ('ascii'   , 'CASint')  , 
# ):
    # _f = lambda species,inp=inp,out=out:_species_database_translate(species,inp,out)
    # _convert_species_functions[inp][out] = _f

# ## experimental hash including isotopologues
# def _f(species):
    # from . import database
    # ## if isotopologue compute isotope hash, else retrun CAS as integer
    # if re.match(r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]',species):
        # ## hash is CAS of chemical species * 100000000000 plus a 32bit
        # ## sha1 hash of the isotopologue unicode name. This is
        # ## sensitive to atom order, better to hash the inchi or inchi
        # ## key perhaps
        # import hashlib
        # isohash = int(hashlib.sha1(species.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        # chemical_species = database.normalise_chemical_species(species)
        # chemical_species_CASint = _species_database_translate(chemical_species,'unicode','CASint')
        # retval = chemical_species_CASint*100000000000 + isohash
    # else:
        # retval = _species_database_translate(species,'unicode','CASint')
    # return retval
# _convert_species_functions['unicode']['hash'] = _f

def _f(name):
    """Translate ASCII name into unicode unicode name. E.g., echo
    NH3→NH₃, 14N16O→¹⁴N¹⁶O, AlBr3.6H2O→AlBr₃•6H₂O   
    """
    if r:=re.match(r'^([0-9]+)([A-Z][a-z]?)([0-9]+)([A-Z][a-z]?)(\+*|-*)$',name):
        ## e.g., 14N16O→¹⁴N¹⁶O
        retval = (tools.superscript_numerals(r.group(1))+r.group(2)
                  +tools.superscript_numerals(r.group(3))+r.group(4))
        if len(r.group(5)) == 1:
            retval += tools.superscript_numerals(r.group(5))
        elif len(r.group(5)) > 1:
            retval += tools.superscript_numerals(str(len(r.group(5)))+r.group(5)[0])
    elif name[0] in '0123456789':
        ## initial multiplicity not allowed -- could implement
        raise Exception(f'Cannot translate: {name}')
    elif r:=re.match(r'^\[([0-9]+)([A-Za-z])\]\[([0-9]+)([A-Za-z])\]',name):
        ## e.g., [14N][16O]→¹⁴N¹⁶O
        retval = (tools.superscript_numerals(r.group(1))+r.group(2)
                  +tools.superscript_numerals(r.group(3))+r.group(4))
    elif r:=re.match(r'^(.*[^+-])(\+*|-*)$', name):
        ## atoms and multipliciteis plus charge
        retval = tools.subscript_numerals(r.group(1))
        while r2:=re.match(r'(.*)\.([₀₁₂₃₄₅₆₇₈₉])(.*)',retval):
            retval = (r2.group(1) +'•' +tools.regularise_unicode(r2.group(2)) +r2.group(3))
        if len(r.group(2)) == 1:
            retval += tools.superscript_numerals(r.group(2))
        elif len(r.group(2)) > 1:
            retval += tools.superscript_numerals(str(len(r.group(2)))+r.group(2)[0])
    else:
        raise Exception(f'Cannot translate: {name}')
    return retval
_convert_species_functions['ascii']['unicode'] = _f


## matplotlib
_ascii_matplotlib_translation_dict = bidict({
    '14N2':'${}^{14}$N$_2$',
    '12C18O':r'${}^{12}$C${}^{18}$O',
    '32S16O':r'${}^{32}$S${}^{16}$O',
    '33S16O':r'${}^{33}$S${}^{16}$O',
    '34S16O':r'${}^{34}$S${}^{16}$O',
    '36S16O':r'${}^{36}$S${}^{16}$O',
    '32S18O':r'${}^{32}$S${}^{18}$O',
    '33S18O':r'${}^{33}$S${}^{18}$O',
    '34S18O':r'${}^{34}$S${}^{18}$O',
    '36S18O':r'${}^{36}$S${}^{18}$O',
})

def _f(name):
    """Translate from my normal species names into something that
    looks nice in matplotlib."""
    if name in _ascii_matplotlib_translation_dict:
        name = _ascii_matplotlib_translation_dict[name]
    else:
        name = re.sub(r'([0-9]+)',r'$_{\1}$',name) # subscript multiplicity 
        name = re.sub(r'([+-])',r'$^{\1}$',name) # superscript charge
    return name
_convert_species_functions['ascii']['matplotlib'] = _f

## cantera
def _f(name):
    """From cantera species name to standard. No translation actually"""
    return name
_convert_species_functions['cantera']['ascii'] = _f

## leiden
_leiden_ascii_translation_dict = bidict({
    'Ca':'ca', 'He':'he',
    'Cl':'cl', 'Cr':'cr', 'Mg':'mg', 'Mn':'mn',
    'Na':'na', 'Ni':'ni', 'Rb':'rb', 'Ti':'ti',
    'Zn':'zn', 'Si':'si', 'Li':'li', 'Fe':'fe',
    'HCl':'hcl', 'Al':'al', 'AlH':'alh',
    'LiH':'lih', 'MgH':'mgh', 'NaCl':'nacl',
    'NaH':'nah', 'SiH':'sih', 'Co':'cob'
})

def _f(leiden_name):
    """Translate from Leidne data base to standard."""
    if name in _leiden_ascii_translation_dict:
        name = _leiden_ascii_translation_dict[name]
    else:
        ## default to uper casing
        name = leiden_name.upper()
        name = name.replace('C-','c-')
        name = name.replace('L-','l-')
        ## look for two-letter element names
        name = name.replace('CL','Cl')
        name = name.replace('SI','Si')
        name = name.replace('CA','Ca')
        ## look for isotopologues
        name = name.replace('C*','13C')
        name = name.replace('O*','18O')
        name = name.replace('N*','15N')
        ## assume final p implies +
        if name[-1]=='P' and name!='P':
            name = name[:-1]+'+'
    return name
_convert_species_functions['leiden']['ascii'] = _f

def _f(standard_name):
    """Translate form my normal species names into the Leiden database
    equivalent."""
    standard_name  = standard_name.replace('+','p')
    return standard_name.lower()
_convert_species_functions['ascii']['leiden'] = _f

## meudon_pdr
_ascii_meudon_old_translation_dict = bidict({
    'Ca':'ca', 'Ca+':'ca+', 'He':'he', 'He+':'he+',
    'Cl':'cl', 'Cr':'cr', 'Mg':'mg', 'Mn':'mn', 'Na':'na', 'Ni':'ni',
    'Rb':'rb', 'Ti':'ti', 'Zn':'zn', 'Si':'si', 'Si+':'si+',
    'Li':'li', 'Fe':'fe', 'Fe+':'fe+', 'HCl':'hcl', 'HCl+':'hcl+',
    'Al':'al', 'AlH':'alh', 'h3+':'H3+', 'l-C3H2':'h2c3' ,
    'l-C3H':'c3h' , 'l-C4':'c4' , 'l-C4H':'c4h' , 'CH3CN':'c2h3n',
    'CH3CHO':'c2h4o', 'CH3OCH3':'c2h7o', 'C2H5OH':'c2h6o',
    'CH2CO':'c2h2o', 'HC3N':'c3hn', 'e-':'electr', # not sure
    ## non chemical processes
        'phosec':'phosec', 'phot':'phot', 'photon':'photon', 'grain':'grain',
    # '?':'c3h4',                 # one of either H2CCCH2 or H3CCCH
    # '?':'c3o',                  # not sure
    # '?':'ch2o2',                  # not sure
    })

def _f(name):
    """Standard to Meudon PDR with old isotope labellgin."""
    if name in _ascii_meudon_old_translation_dict:
        name = _ascii_meudon_old_translation_dict[name]
    else:
        name = re.sub(r'\([0-9][SPDF][0-9]?\)','',name) # remove atomic terms e.g., O(3P1) ⟶ O
        for t in (('[18O]','O*'),('[13C]','C*'),('[15N]','N*'),): # isotopes
            name = name.replace(*t)
    return name.lower()
_convert_species_functions['ascii']['meudon old isotope labelling'] = _f

def _f(name):
    """Standard to Meudon PDR."""
    name = re.sub(r'\([0-9][SPDF][0-9]?\)','',name) # remove atomic terms e.g., O(3P1) ⟶ O
    for t in (('[18O]','_18O'),('[13C]','_13C'),('[15N]','_15N'),): # isotopes
        name = name.replace(*t)
    name = re.sub(r'^_', r'' ,name)
    name = re.sub(r' _', r' ',name)
    name = re.sub(r'_ ', r' ',name)
    name = re.sub(r'_$', r'' ,name)
    name = re.sub(r'_\+',r'+',name)
    return(name.lower())
_convert_species_functions['ascii']['meudon'] = _f

## STAND reaction network used in ARGO model
_stand_ascii_translation_dict = bidict({
    'NH3':'H3N',
    'O2+_X2Πg':'O2+_P',
    'O2_a1Δg' :'O2_D',
    'O+_3P' : 'O_²P',       # error in stand O^+^(^3^P) should be O^+^(^2^P)?
    'O+_2D' : 'O_²D',
    'O_1S'  : 'O_¹S',
    'O_1D'  : 'O_¹D',
    'C_1S'  : 'C_¹S',
    'C_1D'  : 'C_¹D',
    'NH3':'H3N',
    'OH':'HO',
})
def _f(name):
    if name in _stand_ascii_translation_dict:
        name = _stand_ascii_translation_dict[name]
    else:
        name = name
    return name
_convert_species_functions['stand']['ascii'] = _f

## kida
def _f(name):
    return(name)
_convert_species_functions['kida']['ascii'] = _f

## latex
def _f(name):
    """Makes a nice latex version of species. THIS COULD BE EXTENDED"""
    try:
        return(get_species_property(name,'latex'))
    except:
        return(r'\ce{'+name.strip()+r'}')
_convert_species_functions['ascii']['latex'] = _f

# ## inchikey to unicode
# _unicode_inchikey_translation_dict = bidict({
    # 'NH₃':  'QGZKDVFQNNGYKY-UHFFFAOYSA-N',
# })
# _convert_species_functions['inchikey']['unicode'] = (_d,None)
# _convert_species_functions['unicode']['inchikey'] = (_d,None)



#################################
## quantum mechanical formulae ##
#################################

def lifetime_to_linewidth(lifetime):
    """Convert lifetime (s) of transition to linewidth (cm-1 FWHM). tau=1/2/pi/gamma/c"""
    return 5.309e-12/lifetime

def linewidth_to_lifetime(linewidth):
    """Convert linewidth (cm-1 FWHM) of transition to lifetime (s). tau=1/2/pi/gamma/c"""
    return 5.309e-12/linewidth

def linewidth_to_rate(linewidth):
    """Convert linewidth (cm-1 FWHM) of transition to lifetime (s). tau=1/2/pi/gamma/c"""
    return linewidth/5.309e-12

def rate_to_linewidth(rate):
    """Convert lifetime (s) of transition to linewidth (cm-1 FWHM)."""
    return 5.309e-12*rate

def transition_moment_to_band_strength(μv):
    """Convert electronic-vibrational transition moment (au) into a band
    summed line strength. This is not really that well defined."""
    return(μv**2)

def transition_moment_to_band_fvalue(μv,νv,Λp,Λpp):
    return(
        band_strength_to_band_fvalue(
            transition_moment_to_band_strength(μv),
            νv,Λp,Λpp))

def band_fvalue_to_transition_moment(fv,νv,Λp,Λpp):
    """Convert band f-value to the absolute value of the transition mometn
    |μ|= |sqrt(q(v',v'')*Re**2)|"""
    return(np.sqrt(band_fvalue_to_band_strength(fv,νv,Λp,Λpp)))

def band_strength_to_band_fvalue(Sv,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-summed f-value."""
    return(Sv*3.038e-6*νv*(2-tools.kronecker_delta(0,Λp+Λpp))/(2-tools.kronecker_delta(0,Λpp)))

def band_fvalue_to_band_strength(fv,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-summed f-value."""
    return fv/(3.038e-6*νv *(2-tools.kronecker_delta(0,Λp+Λpp)) /(2-tools.kronecker_delta(0,Λpp)))

def band_strength_to_band_emission_rate(Sv,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-averaged emission
    rate."""
    return(2.026e-6*νv**3*Sv *(2-tools.kronecker_delta(0,Λp+Λpp)) /(2-tools.kronecker_delta(0,Λp)))

def band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-averaged emission
    rate."""
    return(Aev/(2.026e-6*νv**3 *(2-tools.kronecker_delta(0,Λp+Λpp)) /(2-tools.kronecker_delta(0,Λp))))

def band_emission_rate_to_band_fvalue(Aev,νv,Λp,Λpp):
    return(band_strength_to_band_fvalue(
        band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp),
        νv,Λp,Λpp))

def band_fvalue_to_band_emission_rate(fv,νv,Λp,Λpp):
    return(band_strength_to_band_emission_rate(
        band_fvalue_to_band_strength(fv,νv,Λp,Λpp),
        νv,Λp,Λpp))

def fvalue_to_line_strength(f,Jpp,ν):
    """Convert f-value (dimensionless) to line strength (au). From Eq. 8 larsson1983."""
    return(f/3.038e-6/ν*(2*Jpp+1))

def line_strength_to_fvalue(Sij,Jpp,ν):
    """Convert line strength (au) to f-value (dimensionless). From Eq. 8 larsson1983."""
    return(3.038e-6*ν*Sij/(2*Jpp+1))

def emission_rate_to_line_strength(Ae,Jp,ν):
    """Convert f-value (dimensionless) to line strength (au). From Eq. 8 larsson1983."""
    return(Ae/(2.026e-6*ν**3/(2*Jp+1)))

def line_strength_to_emission_rate(Sij,Jp,ν):
    """Convert line strength (au) to f-value (dimensionless). From Eq. 8 larsson1983."""
    return(Sij*2.026e-6*ν**3/(2*Jp+1))

def fvalue_to_emission_rate(f,ν,Jpp,Jp):
    """Convert f-value to emission rate where upper and lower level
    degeneracies are computed from J."""
    return(line_strength_to_emission_rate(fvalue_to_line_strength(f,Jpp,ν), Jp,ν))

def emission_rate_to_fvalue(Ae,ν,Jpp,Jp):
    """Convert emission rate to f-value where upper and lower level
    degeneracies are computed from J."""
    return(line_strength_to_fvalue(emission_rate_to_line_strength(Ae,Jp,ν), Jpp,ν))

# def fvalue_to_band_fvalue(line_fvalue,symmetryp,branch,Jpp,symmetrypp='1σ'):
    # """Convert line f-value to band f-value. NOTE THAT honl_london_factor IS AN IMPROVEMENT OVER honllondon_factor, COULD IMPLEMENT THAT HERE."""
    # return(line_fvalue/honllondon_factor(Jpp=Jpp,branch=branch,symmetryp=symmetryp,symmetrypp=symmetrypp)*degen(Jpp,symmetryp))

# def band_fvalue_to_fvalue(fv,,branch,Jpp,symmetrypp='1σ'):
    # """ Convert line band f-value to f-value. NOTE THAT honl_london_factor IS AN IMPROVEMENT OVER honllondon_factor, COULD IMPLEMENT THAT HERE."""
    # return(band_fvalue*honllondonfactor(Jpp=Jpp,branch=branch,symmetryp=symmetryp,symmetrypp=symmetrypp)/degen(Jpp,symmetryp))

def band_cross_section_to_band_fvalue(σv):
    """Convert band integrated cross section band-summed f-value. From
    units of cm2*cm-1 to dimensionless."""
    return(1.1296e12*σv)

def band_fvalue_to_band_cross_section(fv):
    """Convert band integrated cross section band-summed f-value. From
    units of cm2*cm-1 to dimensionless."""
    return(fv/1.1296e12)

def cross_section_to_fvalue(σv,temperaturepp,**qnpp):
    """Convert line strength to line f-value. From units of cm2*cm-1
    to dimensionless. If temperature is none then the boltzmann distribution is not used."""
    return(1.1296e12*σv/database.get_boltzmann_population(temperaturepp,**qnpp).squeeze())

def fvalue_to_cross_section(f,temperaturepp,**qnpp):
    """Convert line strength to line f-value. From units of cm2*cm-1
    to dimensionless. If temperature is none then the boltzmann distribution is not used."""
    return(f/(1.1296e12/database.get_boltzmann_population(temperaturepp,**qnpp).squeeze()))

def differential_oscillator_strength_to_cross_section(
        df, units_in='(cm-1)-1', units_out='cm2'):
    if units_in=='(cm-1)-1':
        pass
    elif units_in=='eV-1':
        df /= convert_units(1,'eV','cm-1')
    σ = df/1.1296e12            # from (cm-1)-1 to cm2
    return(convert_units(σ,'cm2',units_out))
     
def pressure_temperature_to_density(p,T,punits='Pa',nunits='m-3'):
    """p = nkT"""
    p = units(p,punits,'Pa')
    n = p/(constants.Boltzmann*T)
    n = units(n,'m-3',nunits)
    return n

def pressure_to_column_density_density(p,T,L,punits='Pa',Lunits='cm',Nunits='cm-2'):
    """p = NLkT"""
    return units(units(p,punits,'Pa') /(constants.Boltzmann*T*unit(L,Lunits,'m')), 'm-3',Nunits)

def doppler_width(
        temperature,            # K
        mass,                   # amu
        ν,                      # wavenumber in cm-1
        units='cm-1.FWHM',      # Units of output widths.
):
    """Calculate Doppler width given temperature and species mass."""
    dk = 2.*6.331e-8*np.sqrt(temperature*32./mass)*ν
    if units=='cm-1.FWHM':
        return dk
    elif units in ('Å.FHWM','A.FWHM','Angstrom.FWHM','Angstroms.FWHM',):
        return tools.dk2dA(dk,ν)
    elif units=='nm.FWHM':
        return tools.dk2dnm(dk,ν)
    elif units=='km.s-1 1σ':
        return tools.dk2b(dk,ν)
    elif units=='km.s-1.FWHM':
        return tools.dk2bFWHM(dk,ν)
    else:
        raise ValueError('units not recognised: '+repr(units))
