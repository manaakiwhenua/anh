class InferException(Exception):
    """For failed inferences in Dataset."""
    pass

class DatabaseException(InferException):
    """Data not found in database or some other problem not indicating a code error."""
    pass             

class InvalidEncodingException(InferException):
    """Invalid encoding of level or transition name."""
    pass

class NonUniqueValueException(Exception):
    """A non-unique value is found in a Dataset when a unique value is expected.."""
    pass

class DecodeSpeciesException(InferException):
    """A species name could not be decoded."""
    pass
