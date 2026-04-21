import importlib


def find_class_using_name(class_name, module_name):
    """Import the module, then find the class, and return the class.

    This is a helper function for the function "find_dataset_using_name()"
    in data/__init__.py, and "find_model_using_name()" in model/__init__.py.

    Parameters:
        class_name (str) -- the class name to look for
        module_name (str) -- the module name to look for the class in

    Returns:
        the class of the model, which is a subclass of BaseModel
    """
    modulelib = importlib.import_module(module_name)
    for name, cls in modulelib.__dict__.items():
        if name.lower() == class_name.lower():
            return cls
    raise NotImplementedError("In %s module, there should be a class with class name that matches %s in lowercase." % (
        module_name, class_name))