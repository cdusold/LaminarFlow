import tensorflow as tf
import pickle as pkl
    
class _tf_temp():
    def __init__(self, name):
        self.name = name

class CruiseControl():
    """
    Laminar Flow's Cruise Control method automates several
    time saving tasks for Tensorflow, allowing for quicker
    prototyping and testing.
    
    Among these abilities is automatic saving in an encapsulated
    tensorflow session. This makes it so you don't need to
    keep open a session to keep the values. However, it does
    require you to use `tf.get_variable` to define variables
    instead of `tf.Variable` directly.
    
    On top of that, automatic initialization of uninitialized
    variables allows for this structure to be dynamically updated
    and have no cat and mouse hunt for errors.
    
    Pickling and unpickling has been implemented in this class
    under the strict condition that all non-variable tensors
    used in args to the `add` function are the result of previous
    functions input to the `add` function. That is, when calling
    `add`, only use direct attributes of that CruiseControl
    instance or variables controlled by it.
    
    Since TensorFlow objects are not pickleable directly, args
    and kwargs to `add` have to be easily sanitized, unless you
    want to do it yourself. More complex sanitization is possible,
    if somewhat difficult. To sanitize yourself, currently you must
    create the magic functions in whatever you pass to `add` if
    it isn't automatically taken care of already.
    
    More documentation to follow.
    """
#Constructor
    def __init__(self, save_file_name, unique_identifier = None):
        #collect_variables()
        self._vars = set()
        self._var_pkl = list()
        self._file_name = save_file_name
        self.saver = None
        self._uuid = unique_identifier if unique_identifier else hex(id(self))[2:]
        try:
            with tf.variable_scope(self._uuid):
                with tf.variable_scope(self._uuid):
                    tf.get_variable("initialized",shape=[1])
        except:
            raise NameError("Error: {} is an invalid unique identifier.".format(self._uuid))
#Structure
    def add(self, name, function, *args, **kwargs):
        if hasattr(self, name):
            raise AttributeError("Error: {} already defined.".format(name))
        current = set(tf.all_variables())
        #sanitize
        sanitized_args = [self.sanitize(arg) for arg in args]
        sanitized_kwargs = {key:self.sanitize(value) for key,value in kwargs.items()}
        #Test.
        try:
            pkl.dumps([sanitized_args,sanitized_kwargs])
        except:
            raise ValueError("Error: Unable to sanitize.")
        #unsanitize
        unsanitized_args = [self.unsanitize(arg) for arg in args]
        unsanitized_kwargs = {key:self.unsanitize(value) for key,value in kwargs.items()}
        with tf.variable_scope(self._uuid):
            with tf.variable_scope(name):
                setattr(self, name, function(*unsanitized_args, **unsanitized_kwargs))
        self._vars |= set(tf.all_variables()) - current
        self._var_pkl.append([name, function, sanitized_args, sanitized_kwargs])
        return self
#Sanitization
    def sanitize(self, obj):
        try:
            pkl.dumps(i)
            return i
        except:
            start = len(self._uuid)
            restart = i.name[start:].find('/') + start + 1
            return _tf_temp(i.name[restart:i.name.rfind(":")])
    def unsanitize(self, obj):
        if isinstance(obj, _tf_temp):
            try:
                return tf.get_variable(obj.name)
            except:
                end = obj.name.find('/')
                return getattr(self, obj.name[:end])
        else:
            return obj
#Features
    def setFile(self, save_file_name):
        self._file_name = save_file_name
#Values
    def save(self, save_file_name = None):
        variables = []
        start = len(self._uuid)
        for i in self._vars:
            restart = i.name[start:].find('/') + start + 1
            if tf.is_variable_initialized(i).eval():
                try:
                    variables.append((i.name[restart:i.name.rfind(":")],i.value().eval()))
                except:
                    #TODO: Don't do this. Limit exceptions to known expected ones.
                    pass
        with open(self._file_name, "wb") as file:
            pkl.dump(variables, file)
    def load(self, save_file_name = None):
        try:
            with open(self._file_name, "rb") as file:
                variables = pkl.load(file)
            with tf.variable_scope(self._uuid, reuse=True):
                for i in variables:
                    tf.get_variable(i[0]).assign(i[1]).eval(session=self._sess)
        except:
            #TODO: Don't do this. Limit exceptions to known expected ones.
            pass
#Serialization
    def __reduce__(self):
        return (CruiseControl, (self._file_name,), self._var_pkl)
    def __setstate__(self, state):
        for i in state:
            self.add(i[0],i[1],*i[2],**i[3])
        return self
#Functionality
    def __enter__(self):
        self._sess = tf.Session()
        uninitialized = tf.report_uninitialized_variables(self._vars)
        self._sess.run(tf.initialize_variables(uninitialized))
        self.load()
        return self._sess.__enter__()
    def __exit__(self, *args):
        self.save()
        return self._sess.__exit__(*args)